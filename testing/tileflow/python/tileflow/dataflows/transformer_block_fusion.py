"""
Full Transformer Block Dataflows with Extended Fusion

This module models complete transformer blocks including:
- Multi-head attention (Q@K^T, Softmax, Softmax@V)
- LayerNorm (mean, variance, normalize)
- GELU activation (non-GEMM elementwise)
- Feed-forward network (Linear + GELU + Linear)
- Residual connections

Fusion strategies demonstrate the advantage of unified architectures:
1. no_fusion: All operations separate (baseline - CPU for non-GEMM)
2. attention_only_fusion: Only softmax fused (previous work, TPU-style)
3. full_fusion: All non-GEMM fused (UniFlow - maximum advantage)

This shows that UniFlow can fuse not just attention softmax, but also
LayerNorm, GELU, and other non-GEMM operations for maximum performance.
"""

import domino.program_ir as dir


def transformer_block_no_fusion_2levels(ctx, tX, batch, num_heads, seq_len, hidden, ff_dim, define_tiling_space=True):
    """
    Transformer block with NO fusion - all operations separate.

    Models baseline accelerator where ALL non-GEMM ops go to CPU via DRAM.

    Operations (all separate):
    1. LayerNorm1 (mean, var, normalize)
    2. Attention QKV projection (GEMM)
    3. Attention scores (Q@K^T) (GEMM)
    4. Attention softmax (max, sub, exp, sum, div)
    5. Attention output (Softmax@V) (GEMM)
    6. Attention output projection (GEMM)
    7. Residual add
    8. LayerNorm2 (mean, var, normalize)
    9. FFN layer 1 (GEMM)
    10. GELU activation
    11. FFN layer 2 (GEMM)
    12. Residual add

    Total: 6 GEMM operations + 6 non-GEMM stages (all CPU)
    """
    model_k = hidden // num_heads

    # Loop variables
    b = dir.Loop(batch, name="B")
    m = dir.Loop(seq_len, name="M")
    n = dir.Loop(hidden, name="N")
    h = dir.Loop(num_heads, name="H")
    k = dir.Loop(model_k, name="K")
    l = dir.Loop(seq_len, name="L")
    f = dir.Loop(ff_dim, name="F")

    # Intermediate tensors
    # LayerNorm1
    tMean1 = dir.Tensor([batch, seq_len], name="Mean1", dtype="int16", ctx=ctx)
    tVar1 = dir.Tensor([batch, seq_len], name="Var1", dtype="int16", ctx=ctx)
    tNorm1 = dir.Tensor([batch, seq_len, hidden], name="Norm1", dtype="int16", ctx=ctx)

    # Attention
    tQ = dir.Tensor([batch, num_heads, seq_len, model_k], name="Q", dtype="int16", ctx=ctx)
    tK = dir.Tensor([batch, num_heads, model_k, seq_len], name="K", dtype="int16", ctx=ctx)
    tV = dir.Tensor([batch, num_heads, seq_len, model_k], name="V", dtype="int16", ctx=ctx)
    tAttnScores = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnScores", dtype="int16", ctx=ctx)
    tAttnMax = dir.Tensor([batch, num_heads, seq_len], name="AttnMax", dtype="int16", ctx=ctx)
    tAttnExp = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnExp", dtype="int16", ctx=ctx)
    tAttnSum = dir.Tensor([batch, num_heads, seq_len], name="AttnSum", dtype="int16", ctx=ctx)
    tAttnProbs = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnProbs", dtype="int16", ctx=ctx)
    tAttnOut = dir.Tensor([batch, num_heads, seq_len, model_k], name="AttnOut", dtype="int16", ctx=ctx)
    tAttnProj = dir.Tensor([batch, seq_len, hidden], name="AttnProj", dtype="int16", ctx=ctx)
    tResid1 = dir.Tensor([batch, seq_len, hidden], name="Resid1", dtype="int16", ctx=ctx)

    # LayerNorm2
    tMean2 = dir.Tensor([batch, seq_len], name="Mean2", dtype="int16", ctx=ctx)
    tVar2 = dir.Tensor([batch, seq_len], name="Var2", dtype="int16", ctx=ctx)
    tNorm2 = dir.Tensor([batch, seq_len, hidden], name="Norm2", dtype="int16", ctx=ctx)

    # FFN
    tFFN1 = dir.Tensor([batch, seq_len, ff_dim], name="FFN1", dtype="int16", ctx=ctx)
    tGELU = dir.Tensor([batch, seq_len, ff_dim], name="GELU", dtype="int16", ctx=ctx)
    tFFN2 = dir.Tensor([batch, seq_len, hidden], name="FFN2", dtype="int16", ctx=ctx)
    tResid2 = dir.Tensor([batch, seq_len, hidden], name="Resid2", dtype="int16", ctx=ctx)

    # Weight tensors (assumed to exist)
    # QKV weights are reshaped to multi-head format [hidden, num_heads, model_k]
    tWQ = dir.Tensor([hidden, num_heads, model_k], name="WQ", dtype="int16", ctx=ctx)
    tWK = dir.Tensor([hidden, num_heads, model_k], name="WK", dtype="int16", ctx=ctx)
    tWV = dir.Tensor([hidden, num_heads, model_k], name="WV", dtype="int16", ctx=ctx)
    # Output projection weights from multi-head format back to hidden
    tWO = dir.Tensor([num_heads, model_k, hidden], name="WO", dtype="int16", ctx=ctx)
    tW1 = dir.Tensor([hidden, ff_dim], name="W1", dtype="int16", ctx=ctx)
    tW2 = dir.Tensor([ff_dim, hidden], name="W2", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=2)
        ctx.define_split(f, nparts=2)

        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        factors_l = ctx.get_split(l)
        factors_f = ctx.get_split(f)
    else:
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]
        factors_f = [dir.Var("int32") for i in range(2)]

    b2, b1, b0 = ctx.split(b, factors=[1, 1, batch])
    m2, m1, m0 = ctx.split(m, factors=[1, 1, seq_len])
    h2, h1, h0 = ctx.split(h, factors=[1, 1, num_heads])
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])
    f2, f1, f0 = ctx.split(f, factors=[*factors_f, 1])

    # NO FUSION - all operations completely separate
    with ctx.sequential():
        # Stage 1: LayerNorm1 - mean (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tMean1[b, m] = tMean1[b, m] + tX[b, m, n]

        # Stage 2: LayerNorm1 - variance (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tVar1[b, m] = tVar1[b, m] + (tX[b, m, n] - tMean1[b, m]) * (tX[b, m, n] - tMean1[b, m])

        # Stage 3: LayerNorm1 - normalize (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tNorm1[b, m, n] = (tX[b, m, n] - tMean1[b, m]) / dir.sqrt(tVar1[b, m])

        # Stage 4-6: Attention QKV projection (GEMM on systolic)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, h2, k2, n2], "Temporal"):
                    with ctx.tile("L1", [h1, k1, n1], "Spatial"):
                        with ctx.tile("L0", [h0, k0, n0], "Temporal"):
                            tQ[b, h, m, k] = tQ[b, h, m, k] + tNorm1[b, m, n] * tWQ[n, h, k]
                            tK[b, h, k, m] = tK[b, h, k, m] + tNorm1[b, m, n] * tWK[n, h, k]
                            tV[b, h, m, k] = tV[b, h, m, k] + tNorm1[b, m, n] * tWV[n, h, k]

        # Stage 7: Attention scores Q@K^T (GEMM on systolic)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                    with ctx.tile("L1", [l1, k1], "Spatial"):
                        with ctx.tile("L0", [l0, k0], "Temporal"):
                            tAttnScores[b, h, m, l] = tAttnScores[b, h, m, l] + tQ[b, h, m, k] * tK[b, h, k, l]

        # Stage 8: Softmax - max (CPU)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tAttnMax[b, h, m] = dir.max(tAttnMax[b, h, m], tAttnScores[b, h, m, l])

        # Stage 9: Softmax - exp (CPU)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tAttnExp[b, h, m, l] = dir.exp(tAttnScores[b, h, m, l] - tAttnMax[b, h, m])

        # Stage 10: Softmax - sum (CPU)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tAttnSum[b, h, m] = tAttnSum[b, h, m] + tAttnExp[b, h, m, l]

        # Stage 11: Softmax - normalize (CPU)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                    with ctx.tile("L1", [l1], "Spatial"):
                        with ctx.tile("L0", [l0], "Temporal"):
                            tAttnProbs[b, h, m, l] = tAttnExp[b, h, m, l] / tAttnSum[b, h, m]

        # Stage 12: Attention output Softmax@V (GEMM on systolic)
        with ctx.tile("L2", [b2, h2, m2], "Temporal"):
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, k2, l2], "Temporal"):
                    with ctx.tile("L1", [k1, l1], "Spatial"):
                        with ctx.tile("L0", [k0, l0], "Temporal"):
                            tAttnOut[b, h, m, k] = tAttnOut[b, h, m, k] + tAttnProbs[b, h, m, l] * tV[b, h, l, k]

        # Stage 13: Attention output projection (GEMM on systolic)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2, h2, k2], "Temporal"):
                    with ctx.tile("L1", [n1, h1, k1], "Spatial"):
                        with ctx.tile("L0", [n0, h0, k0], "Temporal"):
                            tAttnProj[b, m, n] = tAttnProj[b, m, n] + tAttnOut[b, h, m, k] * tWO[h, k, n]

        # Stage 14: Residual connection 1 (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tResid1[b, m, n] = tX[b, m, n] + tAttnProj[b, m, n]

        # Stage 15: LayerNorm2 - mean (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tMean2[b, m] = tMean2[b, m] + tResid1[b, m, n]

        # Stage 16: LayerNorm2 - variance (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tVar2[b, m] = tVar2[b, m] + (tResid1[b, m, n] - tMean2[b, m]) * (tResid1[b, m, n] - tMean2[b, m])

        # Stage 17: LayerNorm2 - normalize (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tNorm2[b, m, n] = (tResid1[b, m, n] - tMean2[b, m]) / dir.sqrt(tVar2[b, m])

        # Stage 18: FFN layer 1 (GEMM on systolic)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, f2, n2], "Temporal"):
                    with ctx.tile("L1", [f1, n1], "Spatial"):
                        with ctx.tile("L0", [f0, n0], "Temporal"):
                            tFFN1[b, m, f] = tFFN1[b, m, f] + tNorm2[b, m, n] * tW1[n, f]

        # Stage 19: GELU activation (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, f2], "Temporal"):
                    with ctx.tile("L1", [f1], "Spatial"):
                        with ctx.tile("L0", [f0], "Temporal"):
                            tGELU[b, m, f] = tFFN1[b, m, f] * dir.sigmoid(1.702 * tFFN1[b, m, f])

        # Stage 20: FFN layer 2 (GEMM on systolic)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2, f2], "Temporal"):
                    with ctx.tile("L1", [n1, f1], "Spatial"):
                        with ctx.tile("L0", [n0, f0], "Temporal"):
                            tFFN2[b, m, n] = tFFN2[b, m, n] + tGELU[b, m, f] * tW2[f, n]

        # Stage 21: Residual connection 2 (CPU)
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tResid2[b, m, n] = tResid1[b, m, n] + tFFN2[b, m, n]

    return [tResid2], [b, m, n, h, k, l, f]


def transformer_block_attention_only_fusion_2levels(ctx, tX, batch, num_heads, seq_len, hidden, ff_dim, define_tiling_space=True):
    """
    Transformer block with TPU-style execution (previous work).

    Models TPU-like accelerator where:
    - ALL non-GEMM operations execute on on-chip VPU (not CPU!)
    - Operations are PIPELINED but NOT fully fused
    - Each operation is still a separate stage (writes to intermediate buffers)

    Key difference from baseline: On-chip VPU vs off-chip CPU
    Key difference from UniFlow: Separate stages vs fully fused

    This represents the state-of-the-art before UniFlow:
    - Better than baseline (on-chip VPU vs off-chip CPU)
    - Worse than UniFlow (separate stages vs fully fused)
    """
    model_k = hidden // num_heads

    # [Similar tensor definitions as no_fusion...]
    b = dir.Loop(batch, name="B")
    m = dir.Loop(seq_len, name="M")
    n = dir.Loop(hidden, name="N")
    h = dir.Loop(num_heads, name="H")
    k = dir.Loop(model_k, name="K")
    l = dir.Loop(seq_len, name="L")
    f = dir.Loop(ff_dim, name="F")

    # [Same tensors as no_fusion - omitted for brevity]
    tMean1 = dir.Tensor([batch, seq_len], name="Mean1", dtype="int16", ctx=ctx)
    tVar1 = dir.Tensor([batch, seq_len], name="Var1", dtype="int16", ctx=ctx)
    tNorm1 = dir.Tensor([batch, seq_len, hidden], name="Norm1", dtype="int16", ctx=ctx)
    tQ = dir.Tensor([batch, num_heads, seq_len, model_k], name="Q", dtype="int16", ctx=ctx)
    tK = dir.Tensor([batch, num_heads, model_k, seq_len], name="K", dtype="int16", ctx=ctx)
    tV = dir.Tensor([batch, num_heads, seq_len, model_k], name="V", dtype="int16", ctx=ctx)
    tAttnScores = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnScores", dtype="int16", ctx=ctx)
    tAttnMax = dir.Tensor([batch, num_heads, seq_len], name="AttnMax", dtype="int16", ctx=ctx)
    tAttnExp = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnExp", dtype="int16", ctx=ctx)
    tAttnSum = dir.Tensor([batch, num_heads, seq_len], name="AttnSum", dtype="int16", ctx=ctx)
    tAttnProbs = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnProbs", dtype="int16", ctx=ctx)
    tAttnOut = dir.Tensor([batch, num_heads, seq_len, model_k], name="AttnOut", dtype="int16", ctx=ctx)
    tAttnProj = dir.Tensor([batch, seq_len, hidden], name="AttnProj", dtype="int16", ctx=ctx)
    tResid1 = dir.Tensor([batch, seq_len, hidden], name="Resid1", dtype="int16", ctx=ctx)
    tMean2 = dir.Tensor([batch, seq_len], name="Mean2", dtype="int16", ctx=ctx)
    tVar2 = dir.Tensor([batch, seq_len], name="Var2", dtype="int16", ctx=ctx)
    tNorm2 = dir.Tensor([batch, seq_len, hidden], name="Norm2", dtype="int16", ctx=ctx)
    tFFN1 = dir.Tensor([batch, seq_len, ff_dim], name="FFN1", dtype="int16", ctx=ctx)
    tGELU = dir.Tensor([batch, seq_len, ff_dim], name="GELU", dtype="int16", ctx=ctx)
    tFFN2 = dir.Tensor([batch, seq_len, hidden], name="FFN2", dtype="int16", ctx=ctx)
    tResid2 = dir.Tensor([batch, seq_len, hidden], name="Resid2", dtype="int16", ctx=ctx)

    # QKV weights are reshaped to multi-head format [hidden, num_heads, model_k]
    tWQ = dir.Tensor([hidden, num_heads, model_k], name="WQ", dtype="int16", ctx=ctx)
    tWK = dir.Tensor([hidden, num_heads, model_k], name="WK", dtype="int16", ctx=ctx)
    tWV = dir.Tensor([hidden, num_heads, model_k], name="WV", dtype="int16", ctx=ctx)
    # Output projection weights from multi-head format back to hidden
    tWO = dir.Tensor([num_heads, model_k, hidden], name="WO", dtype="int16", ctx=ctx)
    tW1 = dir.Tensor([hidden, ff_dim], name="W1", dtype="int16", ctx=ctx)
    tW2 = dir.Tensor([ff_dim, hidden], name="W2", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(b, nparts=3)
        ctx.define_split(m, nparts=3)
        ctx.define_split(h, nparts=3)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=3)
        ctx.define_split(f, nparts=2)

        factors_b = ctx.get_split(b)
        factors_m = ctx.get_split(m)
        factors_h = ctx.get_split(h)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        factors_l = ctx.get_split(l)
        factors_f = ctx.get_split(f)
    else:
        factors_b = [dir.Var("int32") for i in range(3)]
        factors_m = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(3)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(3)]
        factors_f = [dir.Var("int32") for i in range(2)]

    b2, b1, b0 = ctx.split(b, factors=factors_b)
    m2, m1, m0 = ctx.split(m, factors=factors_m)
    h2, h1, h0 = ctx.split(h, factors=factors_h)
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l3, l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])
    f2, f1, f0 = ctx.split(f, factors=[*factors_f, 1])

    # TPU-STYLE: All non-GEMM on VPU (on-chip), but pipelined not fused
    with ctx.sequential():
        # LayerNorm1 - On VPU (on-chip), but separate stages (pipelined)
        with ctx.pipeline():
            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [n1], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tMean1[b, m] = tMean1[b, m] + tX[b, m, n]

            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [n1], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tVar1[b, m] = tVar1[b, m] + (tX[b, m, n] - tMean1[b, m]) * (tX[b, m, n] - tMean1[b, m])

            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [n1], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tNorm1[b, m, n] = (tX[b, m, n] - tMean1[b, m]) / dir.sqrt(tVar1[b, m])

        # Attention with SOFTMAX FUSION (this is the improvement)
        with ctx.tile("L2", [b2, h2, m2, l3], "Temporal"):
            with ctx.pipeline():
                # QKV projection
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, k2, n2], "Temporal"):
                        with ctx.tile("L1", [k1, n1], "Spatial"):
                            with ctx.tile("L0", [k0, n0], "Temporal"):
                                tQ[b, h, m, k] = tQ[b, h, m, k] + tNorm1[b, m, n] * tWQ[n, h, k]
                                tK[b, h, k, m] = tK[b, h, k, m] + tNorm1[b, m, n] * tWK[n, h, k]
                                tV[b, h, m, k] = tV[b, h, m, k] + tNorm1[b, m, n] * tWV[n, h, k]

                # Attention scores
                with ctx.pipeline():
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                            with ctx.tile("L1", [l1, k1], "Spatial"):
                                with ctx.tile("L0", [l0, k0], "Temporal"):
                                    tAttnScores[b, h, m, l] = tAttnScores[b, h, m, l] + tQ[b, h, m, k] * tK[b, h, k, l]

                    # Softmax FUSED (pipelined on VPU)
                    with ctx.pipeline():
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                with ctx.tile("L1", [l1], "Spatial"):
                                    with ctx.tile("L0", [l0], "Temporal"):
                                        tAttnMax[b, h, m] = dir.max(tAttnMax[b, h, m], tAttnScores[b, h, m, l])

                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                with ctx.tile("L1", [l1], "Spatial"):
                                    with ctx.tile("L0", [l0], "Temporal"):
                                        tAttnExp[b, h, m, l] = dir.exp(tAttnScores[b, h, m, l] - tAttnMax[b, h, m])

                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                with ctx.tile("L1", [l1], "Spatial"):
                                    with ctx.tile("L0", [l0], "Temporal"):
                                        tAttnSum[b, h, m] = tAttnSum[b, h, m] + tAttnExp[b, h, m, l]

                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                with ctx.tile("L1", [l1], "Spatial"):
                                    with ctx.tile("L0", [l0], "Temporal"):
                                        tAttnProbs[b, h, m, l] = tAttnExp[b, h, m, l] / tAttnSum[b, h, m]

                # Attention output
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, k2, l2], "Temporal"):
                        with ctx.tile("L1", [k1, l1], "Spatial"):
                            with ctx.tile("L0", [k0, l0], "Temporal"):
                                tAttnOut[b, h, m, k] = tAttnOut[b, h, m, k] + tAttnProbs[b, h, m, l] * tV[b, h, l, k]

        # Output projection
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2, h2, k2], "Temporal"):
                    with ctx.tile("L1", [n1, h1, k1], "Spatial"):
                        with ctx.tile("L0", [n0, h0, k0], "Temporal"):
                            tAttnProj[b, m, n] = tAttnProj[b, m, n] + tAttnOut[b, h, m, k] * tWO[h, k, n]

        # Residual - On VPU (on-chip), separate stage
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tResid1[b, m, n] = tX[b, m, n] + tAttnProj[b, m, n]

        # LayerNorm2 - On VPU (on-chip), but pipelined separate stages
        with ctx.pipeline():
            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [n1], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tMean2[b, m] = tMean2[b, m] + tResid1[b, m, n]

            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [n1], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tVar2[b, m] = tVar2[b, m] + (tResid1[b, m, n] - tMean2[b, m]) * (tResid1[b, m, n] - tMean2[b, m])

            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [n1], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tNorm2[b, m, n] = (tResid1[b, m, n] - tMean2[b, m]) / dir.sqrt(tVar2[b, m])

        # FFN layer 1
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, f2, n2], "Temporal"):
                    with ctx.tile("L1", [f1, n1], "Spatial"):
                        with ctx.tile("L0", [f0, n0], "Temporal"):
                            tFFN1[b, m, f] = tFFN1[b, m, f] + tNorm2[b, m, n] * tW1[n, f]

        # GELU - On VPU (on-chip), separate stage
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, f2], "Temporal"):
                    with ctx.tile("L1", [f1], "Spatial"):
                        with ctx.tile("L0", [f0], "Temporal"):
                            tGELU[b, m, f] = tFFN1[b, m, f] * dir.sigmoid(1.702 * tFFN1[b, m, f])

        # FFN layer 2
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2, f2], "Temporal"):
                    with ctx.tile("L1", [n1, f1], "Spatial"):
                        with ctx.tile("L0", [n0, f0], "Temporal"):
                            tFFN2[b, m, n] = tFFN2[b, m, n] + tGELU[b, m, f] * tW2[f, n]

        # Residual - On VPU (on-chip), separate stage
        with ctx.tile("L2", [b2, m2], "Temporal"):
            with ctx.tile("L2", [b1, m1], "Spatial"):
                with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                    with ctx.tile("L1", [n1], "Spatial"):
                        with ctx.tile("L0", [n0], "Temporal"):
                            tResid2[b, m, n] = tResid1[b, m, n] + tFFN2[b, m, n]

    return [tResid2], [b, m, n, h, k, l, f]


def transformer_block_full_fusion_2levels(ctx, tX, batch, num_heads, seq_len, hidden, ff_dim, define_tiling_space=True):
    """
    Transformer block with FULL FUSION (UniFlow).

    Models UniFlow accelerator where ALL non-GEMM operations are fused:
    - LayerNorm (mean + var + normalize) fully fused
    - Attention softmax (max + exp + sum + div) fully fused
    - GELU activation fused
    - Residual connections fused

    This is the key advantage of UniFlow:
    - Not just attention softmax (like TPU)
    - But ALL non-GEMM operations throughout the transformer
    - Maximum data locality and minimal data movement
    """
    model_k = hidden // num_heads

    b = dir.Loop(batch, name="B")
    m = dir.Loop(seq_len, name="M")
    n = dir.Loop(hidden, name="N")
    h = dir.Loop(num_heads, name="H")
    k = dir.Loop(model_k, name="K")
    l = dir.Loop(seq_len, name="L")
    f = dir.Loop(ff_dim, name="F")

    # [Same tensor definitions - omitted for brevity]
    tMean1 = dir.Tensor([batch, seq_len], name="Mean1", dtype="int16", ctx=ctx)
    tVar1 = dir.Tensor([batch, seq_len], name="Var1", dtype="int16", ctx=ctx)
    tNorm1 = dir.Tensor([batch, seq_len, hidden], name="Norm1", dtype="int16", ctx=ctx)
    tQ = dir.Tensor([batch, num_heads, seq_len, model_k], name="Q", dtype="int16", ctx=ctx)
    tK = dir.Tensor([batch, num_heads, model_k, seq_len], name="K", dtype="int16", ctx=ctx)
    tV = dir.Tensor([batch, num_heads, seq_len, model_k], name="V", dtype="int16", ctx=ctx)
    tAttnScores = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnScores", dtype="int16", ctx=ctx)
    tAttnMax = dir.Tensor([batch, num_heads, seq_len], name="AttnMax", dtype="int16", ctx=ctx)
    tAttnExp = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnExp", dtype="int16", ctx=ctx)
    tAttnSum = dir.Tensor([batch, num_heads, seq_len], name="AttnSum", dtype="int16", ctx=ctx)
    tAttnProbs = dir.Tensor([batch, num_heads, seq_len, seq_len], name="AttnProbs", dtype="int16", ctx=ctx)
    tAttnOut = dir.Tensor([batch, num_heads, seq_len, model_k], name="AttnOut", dtype="int16", ctx=ctx)
    tAttnProj = dir.Tensor([batch, seq_len, hidden], name="AttnProj", dtype="int16", ctx=ctx)
    tResid1 = dir.Tensor([batch, seq_len, hidden], name="Resid1", dtype="int16", ctx=ctx)
    tMean2 = dir.Tensor([batch, seq_len], name="Mean2", dtype="int16", ctx=ctx)
    tVar2 = dir.Tensor([batch, seq_len], name="Var2", dtype="int16", ctx=ctx)
    tNorm2 = dir.Tensor([batch, seq_len, hidden], name="Norm2", dtype="int16", ctx=ctx)
    tFFN1 = dir.Tensor([batch, seq_len, ff_dim], name="FFN1", dtype="int16", ctx=ctx)
    tGELU = dir.Tensor([batch, seq_len, ff_dim], name="GELU", dtype="int16", ctx=ctx)
    tFFN2 = dir.Tensor([batch, seq_len, hidden], name="FFN2", dtype="int16", ctx=ctx)
    tResid2 = dir.Tensor([batch, seq_len, hidden], name="Resid2", dtype="int16", ctx=ctx)

    # QKV weights are reshaped to multi-head format [hidden, num_heads, model_k]
    tWQ = dir.Tensor([hidden, num_heads, model_k], name="WQ", dtype="int16", ctx=ctx)
    tWK = dir.Tensor([hidden, num_heads, model_k], name="WK", dtype="int16", ctx=ctx)
    tWV = dir.Tensor([hidden, num_heads, model_k], name="WV", dtype="int16", ctx=ctx)
    # Output projection weights from multi-head format back to hidden
    tWO = dir.Tensor([num_heads, model_k, hidden], name="WO", dtype="int16", ctx=ctx)
    tW1 = dir.Tensor([hidden, ff_dim], name="W1", dtype="int16", ctx=ctx)
    tW2 = dir.Tensor([ff_dim, hidden], name="W2", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(b, nparts=3)
        ctx.define_split(m, nparts=3)
        ctx.define_split(h, nparts=3)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=3)
        ctx.define_split(f, nparts=2)

        factors_b = ctx.get_split(b)
        factors_m = ctx.get_split(m)
        factors_h = ctx.get_split(h)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        factors_l = ctx.get_split(l)
        factors_f = ctx.get_split(f)
    else:
        factors_b = [dir.Var("int32") for i in range(3)]
        factors_m = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(3)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(3)]
        factors_f = [dir.Var("int32") for i in range(2)]

    b2, b1, b0 = ctx.split(b, factors=factors_b)
    m2, m1, m0 = ctx.split(m, factors=factors_m)
    h2, h1, h0 = ctx.split(h, factors=factors_h)
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l3, l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])
    f2, f1, f0 = ctx.split(f, factors=[*factors_f, 1])

    # FULL FUSION - all non-GEMM operations fused
    with ctx.pipeline():
        # LayerNorm1 FULLY FUSED (all 3 ops in one scope)
        with ctx.parallel():
            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [b0, m0], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                # All LayerNorm ops fused together
                                tMean1[b, m] = tMean1[b, m] + tX[b, m, n]
                                tVar1[b, m] = tVar1[b, m] + (tX[b, m, n] - tMean1[b, m]) * (tX[b, m, n] - tMean1[b, m])
                                tNorm1[b, m, n] = (tX[b, m, n] - tMean1[b, m]) / dir.sqrt(tVar1[b, m])

            # Attention with FULL FUSION
            with ctx.tile("L2", [b2, h2, m2, l3], "Temporal"):
                with ctx.pipeline():
                    # QKV projection
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, k2, n2], "Temporal"):
                            with ctx.tile("L1", [k1, n1], "Spatial"):
                                with ctx.tile("L0", [k0, n0], "Temporal"):
                                    tQ[b, h, m, k] = tQ[b, h, m, k] + tNorm1[b, m, n] * tWQ[n, h, k]
                                    tK[b, h, k, m] = tK[b, h, k, m] + tNorm1[b, m, n] * tWK[n, h, k]
                                    tV[b, h, m, k] = tV[b, h, m, k] + tNorm1[b, m, n] * tWV[n, h, k]

                    # Attention with softmax FULLY FUSED
                    with ctx.parallel():
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                                with ctx.tile("L1", [l1, k1], "Spatial"):
                                    with ctx.tile("L0", [l0, k0], "Temporal"):
                                        tAttnScores[b, h, m, l] = tAttnScores[b, h, m, l] + tQ[b, h, m, k] * tK[b, h, k, l]

                        # Softmax FULLY FUSED (all ops in one scope)
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                with ctx.tile("L1", [b0, h0, m0], "Spatial"):
                                    with ctx.tile("L0", [l0], "Temporal"):
                                        tAttnMax[b, h, m] = dir.max(tAttnMax[b, h, m], tAttnScores[b, h, m, l])
                                        tAttnExp[b, h, m, l] = dir.exp(tAttnScores[b, h, m, l] - tAttnMax[b, h, m])
                                        tAttnSum[b, h, m] = tAttnSum[b, h, m] + tAttnExp[b, h, m, l]
                                        tAttnProbs[b, h, m, l] = tAttnExp[b, h, m, l] / tAttnSum[b, h, m]

                    # Attention output + projection
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, k2, l2], "Temporal"):
                            with ctx.tile("L1", [k1, l1], "Spatial"):
                                with ctx.tile("L0", [k0, l0], "Temporal"):
                                    tAttnOut[b, h, m, k] = tAttnOut[b, h, m, k] + tAttnProbs[b, h, m, l] * tV[b, h, l, k]

        # Output projection + residual FUSED
        with ctx.parallel():
            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2, h2, k2], "Temporal"):
                        with ctx.tile("L1", [n1, h1, k1], "Spatial"):
                            with ctx.tile("L0", [n0, h0, k0], "Temporal"):
                                tAttnProj[b, m, n] = tAttnProj[b, m, n] + tAttnOut[b, h, m, k] * tWO[h, k, n]
                                tResid1[b, m, n] = tX[b, m, n] + tAttnProj[b, m, n]

            # LayerNorm2 FULLY FUSED
            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2], "Temporal"):
                        with ctx.tile("L1", [b0, m0], "Spatial"):
                            with ctx.tile("L0", [n0], "Temporal"):
                                tMean2[b, m] = tMean2[b, m] + tResid1[b, m, n]
                                tVar2[b, m] = tVar2[b, m] + (tResid1[b, m, n] - tMean2[b, m]) * (tResid1[b, m, n] - tMean2[b, m])
                                tNorm2[b, m, n] = (tResid1[b, m, n] - tMean2[b, m]) / dir.sqrt(tVar2[b, m])

        # FFN with GELU FUSED
        with ctx.parallel():
            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, f2, n2], "Temporal"):
                        with ctx.tile("L1", [f1, n1], "Spatial"):
                            with ctx.tile("L0", [f0, n0], "Temporal"):
                                tFFN1[b, m, f] = tFFN1[b, m, f] + tNorm2[b, m, n] * tW1[n, f]
                                # GELU FUSED with FFN1
                                tGELU[b, m, f] = tFFN1[b, m, f] * dir.sigmoid(1.702 * tFFN1[b, m, f])

            with ctx.tile("L2", [b2, m2], "Temporal"):
                with ctx.tile("L2", [b1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, m0, n2, f2], "Temporal"):
                        with ctx.tile("L1", [n1, f1], "Spatial"):
                            with ctx.tile("L0", [n0, f0], "Temporal"):
                                tFFN2[b, m, n] = tFFN2[b, m, n] + tGELU[b, m, f] * tW2[f, n]
                                # Residual FUSED with FFN2
                                tResid2[b, m, n] = tResid1[b, m, n] + tFFN2[b, m, n]

    return [tResid2], [b, m, n, h, k, l, f]


def get_transformer_block_dataflow(levels, fusion_strategy, batch, num_heads, seq_len, hidden, ff_dim, define_tiling_space=True):
    """
    Get full transformer block dataflow with specified fusion strategy.

    Args:
        levels: Memory hierarchy levels (2 for edge, 3 for cloud)
        fusion_strategy: One of:
            - 'no_fusion': Baseline (all non-GEMM to CPU)
            - 'attention_only_fusion': Previous work (only attention softmax fused)
            - 'full_fusion': UniFlow (all non-GEMM fused)
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        hidden: Hidden dimension
        ff_dim: Feed-forward intermediate dimension (typically 4*hidden)
        define_tiling_space: Whether to define tiling space for tuning

    Returns:
        Dataflow function for TileFlow
    """
    def static_transformer_block(ctx):
        B = batch
        H = num_heads
        M = seq_len
        N = hidden
        F = ff_dim

        # Note: Not using only_capital=True because transformer blocks have 35+ unique tensor names
        # which exceeds the 26 capital letter limit (A-Z)
        with dir.NameScope():
            tX = dir.Tensor([B, M, N], name="X", dtype="int16", ctx=ctx)

            if levels == 2:
                if fusion_strategy == 'no_fusion':
                    [tOut], loops = transformer_block_no_fusion_2levels(
                        ctx, tX, B, H, M, N, F, define_tiling_space=define_tiling_space)
                elif fusion_strategy == 'attention_only_fusion':
                    [tOut], loops = transformer_block_attention_only_fusion_2levels(
                        ctx, tX, B, H, M, N, F, define_tiling_space=define_tiling_space)
                elif fusion_strategy == 'full_fusion':
                    [tOut], loops = transformer_block_full_fusion_2levels(
                        ctx, tX, B, H, M, N, F, define_tiling_space=define_tiling_space)
                else:
                    raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            else:
                raise NotImplementedError(f"3-level transformer blocks not yet implemented")

            return [tX], [tOut], loops

    return static_transformer_block
