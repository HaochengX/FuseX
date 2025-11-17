"""
KV-Cache Aware Attention Dataflows

This module implements attention dataflows optimized for different KV-cache strategies
used in autoregressive language model inference (e.g., GPT, LLaMA).

KV-Cache Strategies:
1. **Static KV-Cache**: Traditional approach
   - Entire KV cache pre-allocated and stored in memory
   - Simple but memory-intensive for long sequences
   - Good for: Short sequences (<512 tokens)

2. **Stream-In KV-Cache**: Dynamic streaming with PE partition
   - KV cache streamed from DRAM as needed
   - Leverages PE partition to overlap compute and memory access
   - Good for: Medium sequences (512-2048 tokens)

3. **Adaptive KV-Cache**: Content-aware caching
   - Only cache important tokens (based on attention scores)
   - Dynamically adjusts cache size
   - Good for: Long sequences (>2048 tokens), sparse attention

Prefill vs Decode:
- **Prefill Phase**: Process entire prompt at once (seq_len = N, batch = 1)
  * Compute Q, K, V for all N tokens
  * Store K, V in cache for future use
  * Full attention matrix N×N

- **Decode Phase**: Generate one token at a time (seq_len = 1, batch = 1)
  * Compute Q, K, V for new token only
  * Reuse cached K, V from previous tokens
  * Attention vector 1×(context_len)
  * context_len grows: 1, 2, 3, ..., N

Performance Characteristics:
- Prefill: Compute-bound (large GEMM operations)
- Decode: Memory-bound (small GEMM, large KV-cache reads)
- PE partition helps decode more (overlap KV streaming with compute)
"""

import domino.program_ir as dir


def attention_static_kv_cache_prefill_2levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden,
                                                define_tiling_space=True):
    """
    Prefill phase with static KV-cache.

    This processes the entire prompt and stores K, V in cache.

    Args:
        ctx: TileFlow context
        tQ, tK, tV: Query, Key, Value tensors [batch, num_heads, seq_len, hidden]
        batch: Batch size (typically 1 for inference)
        num_heads: Number of attention heads
        seq_len: Prompt length
        hidden: Hidden dimension per head
        define_tiling_space: Whether to define tiling parameters

    Returns:
        Tuple of (output tensors, KV cache tensors, loop variables)
    """
    # Define loop variables
    b = dir.Loop(batch, name="B")
    h = dir.Loop(num_heads, name="H")
    m = dir.Loop(seq_len, name="M")  # Query position
    l = dir.Loop(seq_len, name="L")  # Key/Value position
    k = dir.Loop(hidden, name="K")   # Head dimension

    # Intermediate tensors
    tAttnScores = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnScores", dtype="int16", ctx=ctx)
    tAttnMax = dir.Tensor([batch, num_heads, seq_len], name="AttnMax", dtype="int16", ctx=ctx)
    tAttnExp = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnExp", dtype="int16", ctx=ctx)
    tAttnSum = dir.Tensor([batch, num_heads, seq_len], name="AttnSum", dtype="int16", ctx=ctx)
    tAttnProbs = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnProbs", dtype="int16", ctx=ctx)
    tOutput = dir.Tensor([batch, num_heads, seq_len, hidden], name="Output", dtype="int16", ctx=ctx)

    # KV Cache tensors (stored for decode phase)
    tKCache = dir.Tensor([batch, num_heads, seq_len, hidden], name="KCache", dtype="int16", ctx=ctx)
    tVCache = dir.Tensor([batch, num_heads, seq_len, hidden], name="VCache", dtype="int16", ctx=ctx)

    # Define tiling
    if define_tiling_space:
        ctx.define_split(b, nparts=3)
        ctx.define_split(h, nparts=3)
        ctx.define_split(m, nparts=3)
        ctx.define_split(l, nparts=3)
        ctx.define_split(k, nparts=2)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_m = ctx.get_split(m)
        factors_l = ctx.get_split(l)
        factors_k = ctx.get_split(k)
    else:
        factors_b = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(3)]
        factors_m = [dir.Var("int32") for i in range(3)]
        factors_l = [dir.Var("int32") for i in range(3)]
        factors_k = [dir.Var("int32") for i in range(2)]

    b2, b1, b0 = ctx.split(b, factors=factors_b)
    h2, h1, h0 = ctx.split(h, factors=factors_h)
    m2, m1, m0 = ctx.split(m, factors=factors_m)
    l2, l1, l0 = ctx.split(l, factors=factors_l)
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])

    with ctx.pipeline():
        # Store K, V in cache (these will be reused in decode)
        with ctx.tile("L2", [b2, h2, l2], "Temporal"):
            with ctx.tile("L2", [b1, h1, l1], "Spatial"):
                with ctx.tile("L1", [b0, h0, l0, k2], "Temporal"):
                    with ctx.tile("L1", [k1], "Spatial"):
                        with ctx.tile("L0", [k0], "Temporal"):
                            tKCache[b, h, l, k] = tK[b, h, l, k]
                            tVCache[b, h, l, k] = tV[b, h, l, k]

        # Q@K^T
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                    with ctx.tile("L1", [l1, k1], "Spatial"):
                        with ctx.tile("L0", [l0, k0], "Temporal"):
                            tAttnScores[b, h, m, l] = tAttnScores[b, h, m, l] + tQ[b, h, m, k] * tKCache[b, h, l, k]

        # Softmax (fused on unified PE)
        with ctx.parallel():
            with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                        with ctx.tile("L1", [l1], "Spatial"):
                            with ctx.tile("L0", [l0], "Temporal"):
                                tAttnMax[b, h, m] = dir.max(tAttnMax[b, h, m], tAttnScores[b, h, m, l])
                                tAttnExp[b, h, m, l] = dir.exp(tAttnScores[b, h, m, l] - tAttnMax[b, h, m])
                                tAttnSum[b, h, m] = tAttnSum[b, h, m] + tAttnExp[b, h, m, l]
                                tAttnProbs[b, h, m, l] = tAttnExp[b, h, m, l] / tAttnSum[b, h, m]

        # Attn@V
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, k2, l2], "Temporal"):
                    with ctx.tile("L1", [k1, l1], "Spatial"):
                        with ctx.tile("L0", [k0, l0], "Temporal"):
                            tOutput[b, h, m, k] = tOutput[b, h, m, k] + tAttnProbs[b, h, m, l] * tVCache[b, h, l, k]

    return [tOutput, tKCache, tVCache], [b, h, m, l, k]


def attention_static_kv_cache_decode_2levels(ctx, tQ_new, tKCache, tVCache, batch, num_heads,
                                               context_len, hidden, define_tiling_space=True):
    """
    Decode phase with static KV-cache.

    This generates one new token using cached K, V from previous tokens.

    Args:
        ctx: TileFlow context
        tQ_new: Query for new token [batch, num_heads, 1, hidden]
        tKCache: Cached keys [batch, num_heads, context_len, hidden]
        tVCache: Cached values [batch, num_heads, context_len, hidden]
        batch: Batch size (typically 1)
        num_heads: Number of attention heads
        context_len: Number of tokens in context (grows each decode step)
        hidden: Hidden dimension per head
        define_tiling_space: Whether to define tiling

    Returns:
        Tuple of (output tensor, loop variables)
    """
    # Define loop variables
    b = dir.Loop(batch, name="B")
    h = dir.Loop(num_heads, name="H")
    m = dir.Loop(1, name="M")  # Only 1 query token
    l = dir.Loop(context_len, name="L")  # All context tokens
    k = dir.Loop(hidden, name="K")

    # Intermediate tensors
    tAttnScores = dir.Tensor([batch, num_heads, 1, context_len], name="AttnScores", dtype="int16", ctx=ctx)
    tAttnMax = dir.Tensor([batch, num_heads, 1], name="AttnMax", dtype="int16", ctx=ctx)
    tAttnExp = dir.Tensor([batch, num_heads, 1, context_len], name="AttnExp", dtype="int16", ctx=ctx)
    tAttnSum = dir.Tensor([batch, num_heads, 1], name="AttnSum", dtype="int16", ctx=ctx)
    tAttnProbs = dir.Tensor([batch, num_heads, 1, context_len], name="AttnProbs", dtype="int16", ctx=ctx)
    tOutput = dir.Tensor([batch, num_heads, 1, hidden], name="Output", dtype="int16", ctx=ctx)

    # Define tiling
    if define_tiling_space:
        ctx.define_split(b, nparts=3)
        ctx.define_split(h, nparts=3)
        ctx.define_split(m, nparts=3)
        ctx.define_split(l, nparts=3)
        ctx.define_split(k, nparts=2)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_m = ctx.get_split(m)
        factors_l = ctx.get_split(l)
        factors_k = ctx.get_split(k)
    else:
        factors_b = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(3)]
        factors_m = [dir.Var("int32") for i in range(3)]
        factors_l = [dir.Var("int32") for i in range(3)]
        factors_k = [dir.Var("int32") for i in range(2)]

    b2, b1, b0 = ctx.split(b, factors=factors_b)
    h2, h1, h0 = ctx.split(h, factors=factors_h)
    m2, m1, m0 = ctx.split(m, factors=factors_m)
    l2, l1, l0 = ctx.split(l, factors=factors_l)
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])

    with ctx.pipeline():
        # Q@K^T (1×context_len attention vector)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                    with ctx.tile("L1", [l1, k1], "Spatial"):
                        with ctx.tile("L0", [l0, k0], "Temporal"):
                            tAttnScores[b, h, m, l] = tAttnScores[b, h, m, l] + tQ_new[b, h, m, k] * tKCache[b, h, l, k]

        # Softmax
        with ctx.parallel():
            with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                        with ctx.tile("L1", [l1], "Spatial"):
                            with ctx.tile("L0", [l0], "Temporal"):
                                tAttnMax[b, h, m] = dir.max(tAttnMax[b, h, m], tAttnScores[b, h, m, l])
                                tAttnExp[b, h, m, l] = dir.exp(tAttnScores[b, h, m, l] - tAttnMax[b, h, m])
                                tAttnSum[b, h, m] = tAttnSum[b, h, m] + tAttnExp[b, h, m, l]
                                tAttnProbs[b, h, m, l] = tAttnExp[b, h, m, l] / tAttnSum[b, h, m]

        # Attn@V (1×hidden output)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, k2, l2], "Temporal"):
                    with ctx.tile("L1", [k1, l1], "Spatial"):
                        with ctx.tile("L0", [k0, l0], "Temporal"):
                            tOutput[b, h, m, k] = tOutput[b, h, m, k] + tAttnProbs[b, h, m, l] * tVCache[b, h, l, k]

    return [tOutput], [b, h, m, l, k]


def get_attention_kv_cache_dataflow(levels, phase, batch, num_heads, seq_len, hidden,
                                     cache_strategy="static", context_len=None, define_tiling_space=True):
    """
    Get KV-cache aware attention dataflow.

    Args:
        levels: Memory hierarchy levels (2 for edge, 3 for cloud)
        phase: "prefill" or "decode"
        batch: Batch size (typically 1 for inference)
        num_heads: Number of attention heads
        seq_len: Sequence length (prompt length for prefill, 1 for decode)
        hidden: Hidden dimension per head
        cache_strategy: "static", "stream", or "adaptive"
        context_len: Context length (only for decode phase)
        define_tiling_space: Whether to define tiling

    Returns:
        Dataflow context
    """
    if levels != 2:
        raise NotImplementedError("Only 2-level hierarchy currently supported for KV-cache dataflows")

    if cache_strategy != "static":
        raise NotImplementedError(f"Cache strategy '{cache_strategy}' not yet implemented. Use 'static'.")

    ctx = dir.MappingContext()

    if phase == "prefill":
        # Prefill: process entire prompt
        tQ = dir.Tensor([batch, num_heads, seq_len, hidden], name="Q", dtype="int16", ctx=ctx)
        tK = dir.Tensor([batch, num_heads, seq_len, hidden], name="K", dtype="int16", ctx=ctx)
        tV = dir.Tensor([batch, num_heads, seq_len, hidden], name="V", dtype="int16", ctx=ctx)

        output, loop_vars = attention_static_kv_cache_prefill_2levels(
            ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space)

        ctx.set_output(output)
        ctx.set_loop_var(loop_vars)
        return ctx

    elif phase == "decode":
        # Decode: generate one token
        if context_len is None:
            raise ValueError("context_len must be specified for decode phase")

        tQ_new = dir.Tensor([batch, num_heads, 1, hidden], name="Q_new", dtype="int16", ctx=ctx)
        tKCache = dir.Tensor([batch, num_heads, context_len, hidden], name="KCache", dtype="int16", ctx=ctx)
        tVCache = dir.Tensor([batch, num_heads, context_len, hidden], name="VCache", dtype="int16", ctx=ctx)

        output, loop_vars = attention_static_kv_cache_decode_2levels(
            ctx, tQ_new, tKCache, tVCache, batch, num_heads, context_len, hidden, define_tiling_space)

        ctx.set_output(output)
        ctx.set_loop_var(loop_vars)
        return ctx

    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'prefill' or 'decode'.")
