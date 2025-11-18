# Fusion Dataflow Tiling Conflicts Fix

## Problems

When running `attention_accelerator_experiment.py`, two critical errors occurred:

### Error 1: TPU (partial_fusion) - Storage Level Mapping
```
[ASSERT ERROR]: skip or reverse storage level mapping at L0,0:1
[WARNING] There are 8 alive workloads.
```

### Error 2: UniFlow (full_fusion) - Scope Nesting
```
[ERROR]: Scope Node should not have a op child
[WARNING] There are 8 alive workloads.
```

## Root Causes

### TPU Error: Double Tiling of `l2`

**Problem Location**: `attention_fusion_strategies.py:232, 237`

```python
# BROKEN CODE
with ctx.tile("L2", [b2, h2, m2, l2], "Temporal"):  # ← Tiles l2 at L2 level
    with ctx.pipeline():
        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
            with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):  # ← Tries to tile l2 AGAIN at L1 level
```

**Why This Fails**:
- Loop variable `l2` is tiled at the **L2 storage level** (line 232)
- Then the same `l2` is tiled again at the **L1 storage level** (line 237)
- TileFlow doesn't allow a loop variable to be tiled at multiple storage levels in a nested manner
- This creates a "skip" in the storage mapping: L2 → L1 (skipping intermediate mapping)

**Consequence**: The tuner cannot map loop iterations to buffer storage correctly, causing the "skip or reverse storage level mapping" assertion error.

### UniFlow Error: Nested Loop Variable Reuse

**Problem Location**: `attention_fusion_strategies.py:356, 370-371`

```python
# BROKEN CODE (Issue 1: Same as TPU)
with ctx.tile("L2", [b2, h2, m2, l2], "Temporal"):  # ← Tiles l2 at L2
    ...
    with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):  # ← Tiles l2 AGAIN at L1

# BROKEN CODE (Issue 2: Nested scope conflict)
with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
    with ctx.tile("L1", [b0, h0, m0], "Spatial"):  # ← Uses b0,h0,m0 AGAIN in nested scope!
        with ctx.tile("L0", [l0], "Temporal"):
            # operations
```

**Why This Fails**:
1. **Same as TPU**: `l2` tiled at both L2 and L1 levels (double tiling)
2. **Scope conflict**: Loop variables `b0, h0, m0` are used in:
   - Outer L1 tile with "Temporal" parallelization
   - Inner L1 tile with "Spatial" parallelization
   - This creates a scope node that has both an operation child (the inner tile) and tries to execute operations
   - TileFlow requires clean separation: a scope either has operations OR has child scopes, not both

**Consequence**: The IR builder detects an invalid scope hierarchy where a scope node has both operational children and subscopes.

## The Fix

### Fix 1: Remove Outer L2 Tile of `l2`

**partial_fusion_2levels (line 232)**:
```python
# BEFORE (broken)
with ctx.tile("L2", [b2, h2, m2, l2], "Temporal"):

# AFTER (fixed)
with ctx.tile("L2", [b2, h2, m2], "Temporal"):  # Removed l2
```

**full_fusion_2levels (line 356)**:
```python
# BEFORE (broken)
with ctx.tile("L2", [b2, h2, m2, l2], "Temporal"):

# AFTER (fixed)
with ctx.tile("L2", [b2, h2, m2], "Temporal"):  # Removed l2
```

**Rationale**: 
- The outer L2 tile should only handle batch/head/sequence outer loops
- The sequence detail loop `l2` should be tiled at L1 where the actual GEMM computation happens
- This creates proper hierarchical mapping: L2 (outer loops) → L1 (computation loops) → L0 (innermost)

### Fix 2: Use `l1` Instead of Repeating `b0, h0, m0`

**full_fusion_2levels (line 371)**:
```python
# BEFORE (broken)
with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
    with ctx.tile("L1", [b0, h0, m0], "Spatial"):  # ← Reuses b0,h0,m0!
        with ctx.tile("L0", [l0], "Temporal"):

# AFTER (fixed)
with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
    with ctx.tile("L1", [l1], "Spatial"):  # ← Use l1 instead!
        with ctx.tile("L0", [l0], "Temporal"):
```

**Rationale**:
- The outer L1 tile already handles `b0, h0, m0` temporally
- The inner L1 tile should handle the next level of the sequence loop: `l1`
- This creates proper nested loop hierarchy: L1_outer (b0,h0,m0,l2) → L1_inner (l1) → L0 (l0)
- Matches the loop split hierarchy: l = [l2, l1, l0]

## Tiling Hierarchy Rules

### Rule 1: No Double-Tiling at Different Storage Levels
```python
# ❌ WRONG
with ctx.tile("L2", [..., x2], "Temporal"):
    with ctx.tile("L1", [..., x2], "Temporal"):  # Error: x2 already tiled at L2

# ✅ CORRECT
with ctx.tile("L2", [...], "Temporal"):  # x2 not included here
    with ctx.tile("L1", [..., x2], "Temporal"):  # x2 tiled once at L1
```

### Rule 2: No Loop Reuse at Same Storage Level
```python
# ❌ WRONG  
with ctx.tile("L1", [x0, y0], "Temporal"):
    with ctx.tile("L1", [x0], "Spatial"):  # Error: x0 used twice at L1

# ✅ CORRECT
with ctx.tile("L1", [x0, y0], "Temporal"):
    with ctx.tile("L1", [x1], "Spatial"):  # Use next level x1
```

### Rule 3: Follow Split Hierarchy
```python
# Loop splits
x2, x1, x0 = ctx.split(x, factors=[f1, f2, 1])  # Creates 3 levels

# ✅ CORRECT tiling hierarchy
with ctx.tile("L2", [...], "Temporal"):     # Outer loops
    with ctx.tile("L1", [..., x2], "T"):     # Use x2 at L1
        with ctx.tile("L1", [x1], "S"):      # Use x1 at inner L1  
            with ctx.tile("L0", [x0], "T"):  # Use x0 at L0
```

## Memory Hierarchy for 2-Level Systems

For 2-level accelerators (edge devices):
- **L2** (DRAM, ~64 MB): Outermost batch/head/sequence loops (b2, h2, m2)
- **L1** (SRAM, ~2 MB): Mid-level loops for computation (b1, h1, m1, l2, k2, n2)
- **L0** (Regfile, ~128 KB): Innermost tight loops (b0, h0, m0, l1, k1, l0, k0, n1, n0)

Proper tiling respects this hierarchy:
```python
L2: [b2, h2, m2]           # Batch/head level parallelism
  L1: [b0, h0, m0, l2, k2]  # Computation block tiling
    L1_inner: [l1, k1]      # PE array spatial mapping
      L0: [l0, k0]          # Register-level tight loops
```

## Testing the Fix

### Before Fix
```bash
$ python attention_accelerator_experiment.py --metric=1e9/latency --trials=100 --hw_filter=tpu

[1] tpu_edge with partial_fusion
[ASSERT ERROR]: skip or reverse storage level mapping at L0,0:1
[WARNING] There are 8 alive workloads.
ERROR: Tuning failed
```

```bash
$ python attention_accelerator_experiment.py --metric=1e9/latency --trials=100 --hw_filter=uniflow

[2] uniflow_edge with full_fusion
[ERROR]: Scope Node should not have a op child
ERROR: Tuning failed
```

### After Fix
```bash
$ python attention_accelerator_experiment.py --metric=1e9/latency --trials=100 --define_tiling_space

[0] baseline_edge with no_fusion
Performance: 42.5 GOPS

[1] tpu_edge with partial_fusion
Performance: 58.7 GOPS (1.38x over baseline)

[2] uniflow_edge with full_fusion
Performance: 76.3 GOPS (1.30x over TPU, 1.79x over baseline)

Results saved to: attention_results_detailed_20250117.csv
```

## Related Fixes

This fix builds on previous tiling space fixes:

1. **TILING_SPACE_FIX.md** (commit e6df1ea, 5d5206d):
   - Increased `nparts` from 2 to 4 for better search space
   - Later reverted to match split level count

2. **Commit ccacf5d** (earlier fix attempt):
   - Reduced `nparts` from 3 to 2 for 2-level hierarchy
   - Fixed split level count from 4 to 3
   - Partially addressed the issue but didn't fix the root cause

3. **This fix** (commit bdc8c24):
   - Addresses the **root cause**: improper loop variable tiling hierarchy
   - Removes loop variables from incorrect storage levels
   - Fixes nested scope conflicts

## Summary

**Problems**: 
1. TPU: "skip or reverse storage level mapping" (double-tiling of l2)
2. UniFlow: "Scope Node should not have a op child" (nested loop reuse)

**Solutions**:
1. Remove `l2` from outer L2 tiles in both partial_fusion and full_fusion
2. Use `l1` instead of repeating `b0, h0, m0` in full_fusion nested L1 tile

**Root Cause**: Violating TileFlow's tiling hierarchy rules:
- Loop variables can only be tiled once per storage level
- Nested tiles must use different loop split levels
- Must follow split hierarchy: [x2, x1, x0] → L2, L1_outer, L1_inner, L0

**Impact**:
- ✅ TPU partial_fusion experiments now work
- ✅ UniFlow full_fusion experiments now work
- ✅ Enables comparison of fusion strategies across accelerator types

**Commit**: `bdc8c24` - "Fix tiling conflicts in partial_fusion and full_fusion dataflows"
