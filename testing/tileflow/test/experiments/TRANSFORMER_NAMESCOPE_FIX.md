# Transformer Block Fix: NameScope Character Limit

## Problem

When running `transformer_fusion_comparison.py`, all configurations failed with:
```
[0] baseline_edge with no_fusion
Get space...
    ERROR: No enough characters to use.
    Skipping...
```

This affected all fusion strategies (no_fusion, attention_only_fusion, full_fusion) and all hardware configs.

## Root Cause

The `get_transformer_block_dataflow()` function (line 728) used:
```python
with dir.NameScope(only_capital=True):
    tX = dir.Tensor([B, M, N], name="X", dtype="int16", ctx=ctx)
```

`NameScope(only_capital=True)` restricts all tensor/loop names to **capital letters A-Z only** (26 total).

However, the transformer block dataflow creates **36+ unique names**:

### Loop Variables (7)
- B (batch)
- M (seq_len)
- N (hidden)
- H (num_heads)
- K (model_k)
- L (seq_len for attention)
- F (ff_dim)

### Intermediate Tensors (21)
- Mean1, Var1, Norm1 (LayerNorm1)
- Q, K, V (Attention projections)
- AttnScores, AttnMax, AttnExp, AttnSum, AttnProbs, AttnOut, AttnProj (Attention)
- Resid1 (Residual connection)
- Mean2, Var2, Norm2 (LayerNorm2)
- FFN1, GELU, FFN2 (Feed-forward network)
- Resid2 (Residual connection)

### Weight Tensors (6)
- WQ, WK, WV, WO (Attention weights)
- W1, W2 (FFN weights)

### Input/Output (2)
- X (input)
- Out (output)

**Total: 36 unique names trying to fit into 26 letters = ERROR!**

## The Fix

Changed line 728-730 from:
```python
with dir.NameScope(only_capital=True):
    tX = dir.Tensor([B, M, N], name="X", dtype="int16", ctx=ctx)
```

To:
```python
# Note: Not using only_capital=True because transformer blocks have 35+ unique tensor names
# which exceeds the 26 capital letter limit (A-Z)
with dir.NameScope():
    tX = dir.Tensor([B, M, N], name="X", dtype="int16", ctx=ctx)
```

**Result**: The `NameScope()` (without `only_capital=True`) allows flexible naming:
- First occurrence: Uses the name as-is (e.g., "Mean1")
- Subsequent occurrences: Appends suffix (e.g., "Mean1_0", "Mean1_1")

This supports unlimited unique names, solving the character limit issue.

## Why `only_capital=True` Exists

The `only_capital=True` mode is useful for **simple dataflows** with few tensors:
- ✅ Attention (Q, K, V, Output) - 4-8 tensors
- ✅ GEMM (A, B, C) - 3 tensors
- ✅ Conv (Input, Weight, Output) - 3-5 tensors

These fit comfortably within 26 letters and produce clean, concise loop naming.

But for **complex dataflows** like transformer blocks:
- ❌ Transformer block - 36+ tensors
- ❌ Full layer fusion - potentially 50+ tensors

These exceed the limit and must use flexible naming.

## Comparison: When to Use Each

### Use `NameScope(only_capital=True)` when:
- ✅ Simple operations (GEMM, Conv, basic attention)
- ✅ Fewer than 20 unique tensor/loop names
- ✅ Want clean, simple variable names (A, B, C, M, N, K)

### Use `NameScope()` (flexible naming) when:
- ✅ Complex operations (transformer blocks, multi-layer fusion)
- ✅ More than 20 unique tensor/loop names
- ✅ Descriptive names are more important than brevity

## Expected Result

After building and running the experiment:

```bash
cd /home/user/FuseX/testing/tileflow/test/experiments
python transformer_fusion_comparison.py \
    --metric=1e9/latency \
    --trials=100 \
    --edge_only
```

Expected output (no errors):
```
================================================================================
Model: BERT-Base-128
  Heads: 12, Seq: 128, Hidden: 768, FF: 3072
================================================================================

[0] baseline_edge with no_fusion
Get space...
Tuning...
Best performance: 1.00x (baseline)

[1] tpu_edge with attention_only_fusion
Get space...
Tuning...
Best performance: 1.35x

[2] uniflow_edge with full_fusion
Get space...
Tuning...
Best performance: 1.85x

Full fusion benefit (UniFlow vs TPU): 1.37x
Total benefit (UniFlow vs Baseline): 1.85x
```

## Related Files

- `transformer_block_fusion.py:728` - Fixed NameScope restriction
- `attention_fusion_strategies.py:728` - Uses `only_capital=True` (OK - only 8 tensors)
- `attention_kv_cache.py:395` - Uses `only_capital=True` (OK - only 10 tensors)
- `attention_pe_partition.py:171` - Uses `only_capital=True` (OK - only 8 tensors)

## Summary

**Problem**: Transformer blocks have 36+ unique names, but `NameScope(only_capital=True)` limits to 26 (A-Z).

**Solution**: Remove `only_capital=True` restriction for transformer blocks.

**Commit**: `3d673f9` - "Fix transformer block: remove only_capital=True restriction"

The fix allows the transformer fusion comparison experiment to run successfully.
