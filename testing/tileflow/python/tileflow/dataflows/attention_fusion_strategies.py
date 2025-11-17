"""
Attention Dataflow with Configurable Fusion Strategies

This module provides self-attention dataflows with different fusion strategies
to model three types of accelerator architectures:

1. Baseline (no_fusion): Traditional systolic array + CPU
   - GEMM operations on systolic array
   - Non-GEMM operations sent to CPU via DRAM (no fusion)
   - Each non-GEMM op is a separate stage

2. TPU-style (partial_fusion): Systolic array + on-chip VPU
   - GEMM operations on systolic array
   - Non-GEMM operations pipelined on VPU (partial fusion)
   - Operations can overlap but not fully fused

3. UniFlow (full_fusion): Unified PE supporting both GEMM and non-GEMM
   - GEMM operations on unified PEs
   - All non-GEMM operations fully fused on same PEs
   - Maximum fusion and data locality

Usage:
    dataflow = get_attention_fusion_dataflow(
        levels=2,
        fusion_strategy='full_fusion',  # or 'no_fusion', 'partial_fusion'
        batch=1,
        num_heads=12,
        seq_len=512,
        hidden=768
    )
"""

import domino.program_ir as dir


def attention_no_fusion_2levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """
    Baseline: No fusion - each non-GEMM operation is separate.
    Models baseline accelerator where non-GEMM ops go to CPU via DRAM.

    Operations executed sequentially without fusion:
    1. A = Q @ K^T (GEMM)
    2. B = max(A) (separate)
    3. C = A - B (separate)
    4. D = exp(C) (separate)
    5. E = sum(D) (separate)
    6. F = D / E (separate)
    7. G = F @ V (GEMM)
    """
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, num_heads, seq_len],
                    name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([batch, num_heads, seq_len],
                    name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        # Use nparts=2 for 2-level hierarchy to match 3 split levels (l2, l1, l0)
        # nparts controls how many intermediate factorizations are sampled
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=2)

        factors_l = ctx.get_split(l)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
    else:
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]

    sub_b = ctx.split(b, factors=[1, 1, batch])
    sub_h = ctx.split(h, factors=[1, 1, num_heads])
    sub_m = ctx.split(m, factors=[1, 1, seq_len])
    sub_n = ctx.split(n, factors=[*factors_n, 1])
    sub_k = ctx.split(k, factors=[*factors_k, 1])
    sub_l = ctx.split(l, factors=[*factors_l, 1])

    b2, b1, b0 = sub_b
    h2, h1, h0 = sub_h
    m2, m1, m0 = sub_m
    n2, n1, n0 = sub_n
    k2, k1, k0 = sub_k
    l2, l1, l0 = sub_l

    # No fusion - each operation is separate (sequential)
    with ctx.sequential():
        # Stage 1: GEMM - Q @ K^T
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                    with ctx.tile("L1", [l1, k1], "Spatial"):
                        with ctx.tile("L0", [l0, k0], "Temporal"):
                            tA[b, h, m, l] = tA[b, h, m, l] + \
                                tQ[b, h, m, k] * tK[b, h, k, l]

        # Stage 2: Max reduction (separate - models CPU access)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tB[b, h, m] = dir.max(
                                tB[b, h, m], tA[b, h, m, l])

        # Stage 3: Subtraction (separate - models CPU access)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tC[b, h, m, l] = tA[b, h, m, l] - tB[b, h, m]

        # Stage 4: Exponentiation (separate - models CPU access)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tD[b, h, m, l] = dir.exp(tC[b, h, m, l])

        # Stage 5: Sum reduction (separate - models CPU access)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tE[b, h, m] = tE[b, h, m] + tD[b, h, m, l]

        # Stage 6: Division (separate - models CPU access)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tF[b, h, m, l] = tD[b, h, m, l] / tE[b, h, m]

        # Stage 7: GEMM - Softmax @ V
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                    with ctx.tile("L1", [n1, l1], "Spatial"):
                        with ctx.tile("L0", [n0, l0], "Temporal"):
                            tG[b, h, m, n] = tG[b, h, m, n] + \
                                tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def attention_partial_fusion_2levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """
    TPU-style: Partial fusion - non-GEMM operations can be pipelined on VPU.
    Models TPU-like accelerator with on-chip VPU for non-GEMM operations.

    Operations are pipelined but not fully fused:
    1. A = Q @ K^T (GEMM on systolic)
    2-6. Non-GEMM ops pipelined on VPU (partial fusion via pipeline)
    7. G = F @ V (GEMM on systolic)
    """
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, num_heads, seq_len],
                    name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([batch, num_heads, seq_len],
                    name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        # Reduced split levels for 2-level hierarchy to avoid mapping conflicts
        ctx.define_split(b, nparts=2)
        ctx.define_split(h, nparts=2)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(m, nparts=2)
        ctx.define_split(l, nparts=2)

        factors_m = ctx.get_split(m)
        factors_l = ctx.get_split(l)
        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
    else:
        # Use 2 factors for 2-level hierarchy (creates 3 split levels with appended 1)
        factors_b = [dir.Var("int32") for i in range(2)]
        factors_h = [dir.Var("int32") for i in range(2)]
        factors_m = [dir.Var("int32") for i in range(2)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]

    sub_b = ctx.split(b, factors=factors_b)
    sub_h = ctx.split(h, factors=factors_h)
    sub_m = ctx.split(m, factors=factors_m)
    sub_n = ctx.split(n, factors=[*factors_n, 1])
    sub_k = ctx.split(k, factors=[*factors_k, 1])
    sub_l = ctx.split(l, factors=[*factors_l, 1])

    b2, b1, b0 = sub_b
    h2, h1, h0 = sub_h
    m2, m1, m0 = sub_m
    n2, n1, n0 = sub_n
    k2, k1, k0 = sub_k
    l2, l1, l0 = sub_l  # Only 3 split levels now

    # Partial fusion via pipelining (models VPU behavior)
    with ctx.tile("L2", [b2, h2, m2, l2], "Temporal"):
        with ctx.pipeline():
            # Stage 1: GEMM on systolic array
            with ctx.pipeline():
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                        with ctx.tile("L1", [l1, k1], "Spatial"):
                            with ctx.tile("L0", [l0, k0], "Temporal"):
                                tA[b, h, m, l] = tA[b, h, m, l] + \
                                    tQ[b, h, m, k] * tK[b, h, k, l]

                # Stage 2-6: Non-GEMM ops pipelined on VPU (partial fusion)
                with ctx.pipeline():
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tB[b, h, m] = dir.max(
                                        tB[b, h, m], tA[b, h, m, l])
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tC[b, h, m, l] = tA[b, h, m, l] - \
                                        tB[b, h, m]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tD[b, h, m, l] = dir.exp(
                                        tC[b, h, m, l])
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tE[b, h, m] = tE[b, h, m] + \
                                        tD[b, h, m, l]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tF[b, h, m, l] = tD[b, h, m, l] / \
                                        tE[b, h, m]

            # Stage 7: GEMM on systolic array
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                    with ctx.tile("L1", [n1, l1], "Spatial"):
                        with ctx.tile("L0", [n0, l0], "Temporal"):
                            tG[b, h, m, n] = tG[b, h, m, n] + \
                                tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def attention_full_fusion_2levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """
    UniFlow: Full fusion - all non-GEMM operations fully fused on unified PEs.
    Models UniFlow accelerator with unified PEs supporting both GEMM and non-GEMM.

    Operations are maximally fused:
    1. A = Q @ K^T (GEMM on unified PE)
    2-6. All non-GEMM ops fused together on same unified PE
    7. G = F @ V (GEMM on unified PE)
    """
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB_max = dir.Tensor([batch, num_heads, seq_len],
                        name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE_sum = dir.Tensor([batch, num_heads, seq_len],
                        name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        # Reduced split levels for 2-level hierarchy to avoid mapping conflicts
        ctx.define_split(b, nparts=2)
        ctx.define_split(h, nparts=2)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(m, nparts=2)
        ctx.define_split(l, nparts=2)

        factors_m = ctx.get_split(m)
        factors_l = ctx.get_split(l)
        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
    else:
        # Use 2 factors for 2-level hierarchy (creates 3 split levels with appended 1)
        factors_b = [dir.Var("int32") for i in range(2)]
        factors_h = [dir.Var("int32") for i in range(2)]
        factors_m = [dir.Var("int32") for i in range(2)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]

    sub_b = ctx.split(b, factors=factors_b)
    sub_h = ctx.split(h, factors=factors_h)
    sub_m = ctx.split(m, factors=factors_m)
    sub_n = ctx.split(n, factors=[*factors_n, 1])
    sub_k = ctx.split(k, factors=[*factors_k, 1])
    sub_l = ctx.split(l, factors=[*factors_l, 1])

    b2, b1, b0 = sub_b
    h2, h1, h0 = sub_h
    m2, m1, m0 = sub_m
    n2, n1, n0 = sub_n
    k2, k1, k0 = sub_k
    l2, l1, l0 = sub_l  # Only 3 split levels now

    # Full fusion (models unified PE behavior)
    with ctx.tile("L2", [b2, h2, m2, l2], "Temporal"):
        with ctx.pipeline():
            # GEMM and non-GEMM can run in parallel on different PEs
            with ctx.parallel():
                # Stage 1: GEMM on unified PE
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                        with ctx.tile("L1", [l1, k1], "Spatial"):
                            with ctx.tile("L0", [l0, k0], "Temporal"):
                                tA[b, h, m, l] = tA[b, h, m, l] + \
                                    tQ[b, h, m, k] * tK[b, h, k, l]

                # Stage 2-6: All non-GEMM ops fully fused on unified PE
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                        with ctx.tile("L1", [b0, h0, m0], "Spatial"):
                            with ctx.tile("L0", [l0], "Temporal"):
                                # All operations fused in single scope
                                tB_max[b, h, m] = dir.max(tB_max[b, h, m], tA[b, h, m, l])
                                tC[b, h, m, l] = tA[b, h, m, l] - tB_max[b, h, m]
                                tD[b, h, m, l] = dir.exp(tC[b, h, m, l])
                                tE_sum[b, h, m] = tE_sum[b, h, m] + tD[b, h, m, l]
                                tF[b, h, m, l] = tD[b, h, m, l] / tE_sum[b, h, m]

            # Stage 7: GEMM on unified PE
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                    with ctx.tile("L1", [n1, l1], "Spatial"):
                        with ctx.tile("L0", [n0, l0], "Temporal"):
                            tG[b, h, m, n] = tG[b, h, m, n] + \
                                tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


# Similar implementations for 3-level memory hierarchy
def attention_no_fusion_3levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """3-level version of no fusion (baseline for cloud accelerators)"""
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, num_heads, seq_len],
                    name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([batch, num_heads, seq_len],
                    name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        # Use nparts=2 for 3-level hierarchy to match 3 split levels (l2, l1, l0)
        # nparts controls how many intermediate factorizations are sampled
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=2)

        factors_l = ctx.get_split(l)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
    else:
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]

    b4, b3, b2, b1, b0 = ctx.split(b, factors=[1, 1, 1, 1, batch])
    h4, h3, h2, h1, h0 = ctx.split(h, factors=[1, 1, 1, 1, num_heads])
    m4, m3, m2, m1, m0 = ctx.split(m, factors=[1, 1, 1, 1, seq_len])
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])

    # No fusion - each operation separate
    with ctx.sequential():
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                            with ctx.tile("L1", [l1, k1], "Spatial"):
                                with ctx.tile("L0", [l0, k0], "Temporal"):
                                    tA[b, h, m, l] = tA[b, h, m, l] + \
                                        tQ[b, h, m, k] * tK[b, h, k, l]
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tB[b, h, m] = dir.max(
                                        tB[b, h, m], tA[b, h, m, l])
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tC[b, h, m, l] = tA[b, h,
                                                        m, l] - tB[b, h, m]
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tD[b, h, m, l] = dir.exp(
                                        tC[b, h, m, l])
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tE[b, h, m] = tE[b, h, m] + \
                                        tD[b, h, m, l]
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tF[b, h, m, l] = tD[b, h,
                                                        m, l] / tE[b, h, m]
        with ctx.tile("L3", [b4, h4, m4], "Temporal"):
            with ctx.tile("L3", [b3, h3, m3], "Spatial"):
                with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                            with ctx.tile("L1", [n1, l1], "Spatial"):
                                with ctx.tile("L0", [n0, l0], "Temporal"):
                                    tG[b, h, m, n] = tG[b, h, m, n] + \
                                        tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def attention_partial_fusion_3levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """3-level version of partial fusion (TPU-style for cloud)"""
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, num_heads, seq_len],
                    name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([batch, num_heads, seq_len],
                    name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(b, nparts=5)
        ctx.define_split(h, nparts=5)
        ctx.define_split(m, nparts=5)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=4)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_m = ctx.get_split(m)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        factors_l = ctx.get_split(l)
    else:
        factors_b = [dir.Var("int32") for i in range(5)]
        factors_h = [dir.Var("int32") for i in range(5)]
        factors_m = [dir.Var("int32") for i in range(5)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(4)]

    b4, b3, b2, b1, b0 = ctx.split(b, factors=factors_b)
    h4, h3, h2, h1, h0 = ctx.split(h, factors=factors_h)
    m4, m3, m2, m1, m0 = ctx.split(m, factors=factors_m)
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l4, l3, l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])

    # Partial fusion via pipeline
    with ctx.tile("L3", [b4, h4, m4, l4], "Temporal"):
        with ctx.tile("L3", [b3, h3, m3], "Spatial"):
            with ctx.tile("L2", [b2, h2, m2, l3], "Temporal"):
                with ctx.pipeline():
                    with ctx.pipeline():
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                                with ctx.tile("L1", [l1, k1], "Spatial"):
                                    with ctx.tile("L0", [l0, k0], "Temporal"):
                                        tA[b, h, m, l] = tA[b, h, m, l] + \
                                            tQ[b, h, m, k] * tK[b, h, k, l]
                        with ctx.pipeline():
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tB[b, h, m] = dir.max(
                                                tB[b, h, m], tA[b, h, m, l])
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tC[b, h, m, l] = tA[b, h,
                                                                m, l] - tB[b, h, m]
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tD[b, h, m, l] = dir.exp(
                                                tC[b, h, m, l])
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tE[b, h, m] = tE[b, h, m] + \
                                                tD[b, h, m, l]
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tF[b, h, m, l] = tD[b, h,
                                                                m, l] / tE[b, h, m]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                            with ctx.tile("L1", [n1, l1], "Spatial"):
                                with ctx.tile("L0", [n0, l0], "Temporal"):
                                    tG[b, h, m, n] = tG[b, h, m, n] + \
                                        tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def attention_full_fusion_3levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """3-level version of full fusion (UniFlow for cloud)"""
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB_max = dir.Tensor([batch, num_heads, seq_len],
                        name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE_sum = dir.Tensor([batch, num_heads, seq_len],
                        name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(b, nparts=5)
        ctx.define_split(h, nparts=5)
        ctx.define_split(m, nparts=5)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=4)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_m = ctx.get_split(m)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        factors_l = ctx.get_split(l)
    else:
        factors_b = [dir.Var("int32") for i in range(5)]
        factors_h = [dir.Var("int32") for i in range(5)]
        factors_m = [dir.Var("int32") for i in range(5)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(4)]

    b4, b3, b2, b1, b0 = ctx.split(b, factors=factors_b)
    h4, h3, h2, h1, h0 = ctx.split(h, factors=factors_h)
    m4, m3, m2, m1, m0 = ctx.split(m, factors=factors_m)
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l4, l3, l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])

    # Full fusion
    with ctx.tile("L3", [b4, h4, m4, l4], "Temporal"):
        with ctx.tile("L3", [b3, h3, m3], "Spatial"):
            with ctx.tile("L2", [b2, h2, m2, l3], "Temporal"):
                with ctx.pipeline():
                    with ctx.parallel():
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                                with ctx.tile("L1", [l1, k1], "Spatial"):
                                    with ctx.tile("L0", [l0, k0], "Temporal"):
                                        tA[b, h, m, l] = tA[b, h, m, l] + \
                                            tQ[b, h, m, k] * tK[b, h, k, l]
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                with ctx.tile("L1", [b0, h0, m0], "Spatial"):
                                    with ctx.tile("L0", [l0], "Temporal"):
                                        tB_max[b, h, m] = dir.max(tB_max[b, h, m], tA[b, h, m, l])
                                        tC[b, h, m, l] = tA[b, h, m, l] - tB_max[b, h, m]
                                        tD[b, h, m, l] = dir.exp(tC[b, h, m, l])
                                        tE_sum[b, h, m] = tE_sum[b, h, m] + tD[b, h, m, l]
                                        tF[b, h, m, l] = tD[b, h, m, l] / tE_sum[b, h, m]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                            with ctx.tile("L1", [n1, l1], "Spatial"):
                                with ctx.tile("L0", [n0, l0], "Temporal"):
                                    tG[b, h, m, n] = tG[b, h, m, n] + \
                                        tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def get_attention_fusion_dataflow(levels, fusion_strategy, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """
    Get attention dataflow with specified fusion strategy.

    Args:
        levels: Memory hierarchy levels (2 for edge, 3 for cloud)
        fusion_strategy: One of:
            - 'no_fusion': Baseline (systolic array + CPU)
            - 'partial_fusion': TPU-style (systolic array + VPU)
            - 'full_fusion': UniFlow (unified PE)
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        hidden: Hidden dimension
        define_tiling_space: Whether to define tiling space for tuning

    Returns:
        Dataflow function for TileFlow
    """
    def static_self_attention(ctx):
        B = batch
        H = num_heads
        M = seq_len
        N = hidden
        with dir.NameScope(only_capital=True):
            tQ = dir.Tensor([B, H, M, N//H], name="Q", dtype="int16", ctx=ctx)
            tK = dir.Tensor([B, H, N//H, M], name="K", dtype="int16", ctx=ctx)
            tV = dir.Tensor([B, H, M, N//H], name="V", dtype="int16", ctx=ctx)

            # Select appropriate dataflow based on levels and fusion strategy
            if levels == 2:
                if fusion_strategy == 'no_fusion':
                    [tF], loops = attention_no_fusion_2levels(
                        ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
                elif fusion_strategy == 'partial_fusion':
                    [tF], loops = attention_partial_fusion_2levels(
                        ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
                elif fusion_strategy == 'full_fusion':
                    [tF], loops = attention_full_fusion_2levels(
                        ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
                else:
                    raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            elif levels == 3:
                if fusion_strategy == 'no_fusion':
                    [tF], loops = attention_no_fusion_3levels(
                        ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
                elif fusion_strategy == 'partial_fusion':
                    [tF], loops = attention_partial_fusion_3levels(
                        ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
                elif fusion_strategy == 'full_fusion':
                    [tF], loops = attention_full_fusion_3levels(
                        ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
                else:
                    raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            else:
                raise NotImplementedError(f"Levels {levels} not supported")

            return [tQ, tK, tV], [tF], loops

    return static_self_attention
