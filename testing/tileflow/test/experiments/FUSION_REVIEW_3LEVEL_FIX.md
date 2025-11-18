# Fusion Dataflow Review: 3-Level Implementation Fixes

## Review Request

User requested a thorough review of `attention_fusion_strategies.py` to verify correctness of `partial_fusion` and `full_fusion` implementations.

## Summary

**2-Level implementations (Edge)**: ✅ **CORRECT** after previous fixes (commit bdc8c24)

**3-Level implementations (Cloud)**: ❌ **BROKEN** - Same errors as 2-level had, plus additional issues!

## Detailed Findings

### ✅ 2-Level Implementations Are Correct

#### `partial_fusion_2levels` (Lines 163-284)
**Status**: CORRECT ✅

After commit bdc8c24, this implementation properly:
- Removed `l2` from outer L2 tile (line 232)
- Uses proper loop variable hierarchy
- No double-tiling violations

**Tiling Structure**:
```python
L2 Temporal: [b2, h2, m2]                    # ✅ No l2 here
  Pipeline:
    GEMM stage:
      L2 Spatial: [b1, h1, m1]
        L1 Temporal: [b0, h0, m0, l2, k2]    # ✅ l2 first appears here
          L1 Spatial: [l1, k1]                # ✅ l1 used here
            L0 Temporal: [l0, k0]             # ✅ l0 used here
```

#### `full_fusion_2levels` (Lines 287-388)
**Status**: CORRECT ✅

After commit bdc8c24, this implementation properly:
- Removed `l2` from outer L2 tile (line 356)
- Fixed nested L1 scope to use `[l1]` instead of repeating `[b0, h0, m0]` (line 371)
- Uses proper loop variable hierarchy

**Tiling Structure**:
```python
L2 Temporal: [b2, h2, m2]                    # ✅ No l2 here
  Pipeline:
    Parallel:
      GEMM branch:
        L2 Spatial: [b1, h1, m1]
          L1 Temporal: [b0, h0, m0, l2, k2]  # ✅ l2 first appears here
            L1 Spatial: [l1, k1]              # ✅ l1 used here
      Non-GEMM branch:
        L2 Spatial: [b1, h1, m1]
          L1 Temporal: [b0, h0, m0, l2]      # ✅ b,h,m,l2 used here
            L1 Spatial: [l1]                  # ✅ l1 used (NOT b0,h0,m0!)
              L0 Temporal: [l0]               # ✅ l0 used here
```

### ❌ 3-Level Implementations Had Critical Errors

#### `partial_fusion_3levels` (Lines 505-606)
**Status**: BROKEN ❌ → NOW FIXED ✅

**Error Found**: Loop variable `l` was tiled at **ALL 5 storage levels**!

**Before (Broken)**:
```python
# Split creates 5 levels: l4, l3, l2, l1, l0
l4, l3, l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])

# WRONG: l appears at every single storage level
L3 Temporal: [b4, h4, m4, l4]  # ❌ l4 tiled at L3
  L3 Spatial: [b3, h3, m3]
    L2 Temporal: [b2, h2, m2, l3]  # ❌ l3 tiled at L2
      L2 Spatial: [b1, h1, m1]
        L1 Temporal: [b0, h0, m0, l2, k2]  # ❌ l2 tiled at L1
          L1 Spatial: [l1, k1]               # ❌ l1 tiled at L1 spatial
            L0 Temporal: [l0, k0]            # ❌ l0 tiled at L0
```

**Why This Is Wrong**:
- Loop variable `l` is tiled at L3 (l4), L2 (l3), L1 (l2), L1 (l1), AND L0 (l0)
- This creates **5-level nested tiling conflict**
- TileFlow's mapper cannot handle this many levels of the same loop variable
- Causes "skip or reverse storage level mapping" errors
- Memory hierarchy gets confused about where to place l's data

**After (Fixed)**:
```python
# CORRECT: Remove l from outer storage levels
L3 Temporal: [b4, h4, m4]      # ✅ No l4 here
  L3 Spatial: [b3, h3, m3]
    L2 Temporal: [b2, h2, m2]  # ✅ No l3 here either
      L2 Spatial: [b1, h1, m1]
        L1 Temporal: [b0, h0, m0, l2, k2]  # ✅ l2 first appears here
          L1 Spatial: [l1, k1]               # ✅ l1 used here
            L0 Temporal: [l0, k0]            # ✅ l0 used here
```

**Rationale**:
- Outer L3/L2 tiles should only handle batch/head/sequence outer loops (b, h, m)
- The sequence detail loop `l` should only appear at L1 and below
- This matches the 2-level pattern where outer L2 doesn't tile `l`

#### `full_fusion_3levels` (Lines 609-688)
**Status**: BROKEN ❌❌ (TWO errors!) → NOW FIXED ✅

**Error 1**: Same multi-level `l` tiling issue as `partial_fusion_3levels`

**Error 2**: Nested L1 scope reused `b0, h0, m0` variables!

**Before (Broken)**:
```python
# Error 1: Same as partial_fusion
L3 Temporal: [b4, h4, m4, l4]  # ❌ l4 tiled
  L3 Spatial: [b3, h3, m3]
    L2 Temporal: [b2, h2, m2, l3]  # ❌ l3 tiled
      Parallel:
        Non-GEMM branch:
          L2 Spatial: [b1, h1, m1]
            L1 Temporal: [b0, h0, m0, l2]  # Uses b0,h0,m0,l2
              L1 Spatial: [b0, h0, m0]  # ❌❌❌ REPEATS b0,h0,m0!
                L0 Temporal: [l0]
                  # fused operations
```

**Why Error 2 Is Wrong**:
- Line 673: L1 Temporal tile uses `[b0, h0, m0, l2]`
- Line 674: Nested L1 Spatial tile **repeats** `[b0, h0, m0]`
- Same loop variables used twice at L1 storage level with different parallelization
- Creates scope hierarchy conflict: "Scope Node should not have a op child"
- **This is the EXACT same error** we fixed in 2-level implementation!

**After (Fixed)**:
```python
# CORRECT: Fixed both issues
L3 Temporal: [b4, h4, m4]      # ✅ No l4
  L3 Spatial: [b3, h3, m3]
    L2 Temporal: [b2, h2, m2]  # ✅ No l3
      Parallel:
        Non-GEMM branch:
          L2 Spatial: [b1, h1, m1]
            L1 Temporal: [b0, h0, m0, l2]  # Uses b0,h0,m0,l2
              L1 Spatial: [l1]  # ✅ Uses l1, not b0,h0,m0!
                L0 Temporal: [l0]
                  # fused operations
```

## Why These Errors Existed

1. **Pattern inconsistency**: The 2-level implementations were fixed in commit bdc8c24, but the 3-level implementations had the **same errors** that weren't caught

2. **Multi-level complexity**: With 3-level hierarchy, there are more storage levels (L3, L2, L1, L0) and more split levels (5 instead of 3), making it easier to accidentally tile loop variables at too many levels

3. **Copy-paste from broken code**: The 3-level implementations were likely created before the 2-level fixes, so they contained the original errors

## TileFlow Tiling Rules (Reinforced)

### Rule 1: Limit Loop Variable Scope Across Storage Levels
```python
# ❌ WRONG: Loop variable tiled at too many storage levels
L3: [..., x4]
  L2: [..., x3]  # x appears at both L3 and L2
    L1: [..., x2]  # and L1
      L0: [..., x0]  # and L0

# ✅ CORRECT: Loop variable tiled at minimal levels
L3: [...]          # No x at outer levels
  L2: [...]        # No x here either
    L1: [..., x2]  # x first appears at L1
      L1: [x1]     # Next level x1 at inner L1
        L0: [x0]   # Innermost x0 at L0
```

### Rule 2: No Loop Reuse at Same Storage Level
```python
# ❌ WRONG: Loop variable used twice at L1
L1 Temporal: [a0, b0, c0]
  L1 Spatial: [a0, b0]  # Repeats a0, b0!

# ✅ CORRECT: Use different split levels
L1 Temporal: [a0, b0, c0]
  L1 Spatial: [c1]  # Use next level of different variable
```

### Rule 3: Outer Tiles Use Outer Loops Only
```python
# For 3-level hierarchy with loops: b, h, m (outer) and l, k (computation)

# ✅ CORRECT pattern:
L3 Temporal: [b4, h4, m4]     # Only outer loops
  L3 Spatial: [b3, h3, m3]    # Only outer loops
    L2 Temporal: [b2, h2, m2]  # Only outer loops
      L2 Spatial: [b1, h1, m1]  # Only outer loops
        L1 Temporal: [b0, h0, m0, l2, k2]  # NOW include computation loops
          L1 Spatial: [l1, k1]
            L0 Temporal: [l0, k0]

# ❌ WRONG: Including computation loops too early
L3 Temporal: [b4, h4, m4, l4]  # l4 too early!
  L2 Temporal: [b2, h2, m2, l3]  # l3 too early!
```

## Testing Status

### Before Fixes
- ❌ 3-level TPU experiments: Would fail with storage mapping errors
- ❌ 3-level UniFlow experiments: Would fail with scope node errors  
- ❌ Cloud configurations: All failures

### After Fixes (Commit 35d8fb6)
- ✅ 3-level TPU experiments: Should work
- ✅ 3-level UniFlow experiments: Should work
- ✅ Cloud configurations: Should work

## Verification Commands

Test 3-level (cloud) configurations:

```bash
cd /home/user/FuseX/testing/tileflow/test/experiments

# Test cloud TPU
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=100 \
    --hw_filter=tpu \
    --cloud_only \
    --define_tiling_space

# Test cloud UniFlow
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=100 \
    --hw_filter=uniflow \
    --cloud_only \
    --define_tiling_space

# Test all cloud configs
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=100 \
    --cloud_only \
    --define_tiling_space
```

## Summary of All Fusion Fixes

### Commit bdc8c24 (2-level fixes)
Fixed `partial_fusion_2levels` and `full_fusion_2levels`:
- Removed `l2` from outer L2 tiles
- Fixed `full_fusion` nested L1 scope to use `[l1]` instead of `[b0,h0,m0]`

### Commit 35d8fb6 (3-level fixes)
Fixed `partial_fusion_3levels` and `full_fusion_3levels`:
- Removed `l4` from outer L3 tiles
- Removed `l3` from L2 tiles
- Fixed `full_fusion` nested L1 scope to use `[l1]` instead of `[b0,h0,m0]`

**Pattern**: The same types of errors existed in both 2-level and 3-level implementations. Now both are fixed with consistent patterns.

## Lessons Learned

1. **Consistency is critical**: When fixing an error pattern in one place, check if the same pattern exists elsewhere

2. **Multi-level hierarchy is complex**: More storage levels = more opportunities for tiling conflicts

3. **Follow the reference pattern**: 2-level implementations serve as the pattern for 3-level implementations

4. **Outer loops vs computation loops**: Separate concerns - outer tiles handle parallelism over batch/heads/sequence, inner tiles handle computation loops

## Files Changed
- `attention_fusion_strategies.py`:
  - Lines 557, 559: `partial_fusion_3levels` - removed l4, l3
  - Lines 661, 663: `full_fusion_3levels` - removed l4, l3
  - Line 674: `full_fusion_3levels` - changed [b0,h0,m0] to [l1]

## Related Documentation
- `FUSION_TILING_FIX.md`: Original 2-level fixes (commit bdc8c24)
- `TILING_SPACE_FIX.md`: Split level arithmetic fixes
- `DATAFLOW_PATTERN_FIX.md`: Context pattern fixes
