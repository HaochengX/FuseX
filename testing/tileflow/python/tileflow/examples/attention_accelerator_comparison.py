"""
Example: Comparing Different Accelerator Architectures for Self-Attention

This script demonstrates how to model and compare three different accelerator
architectures for self-attention operations:

1. Baseline: Systolic array + CPU for non-GEMM operations
2. TPU-style: Systolic array + on-chip VPU for non-GEMM operations
3. UniFlow: Unified PE supporting both GEMM and non-GEMM operations

The self-attention operation consists of:
- GEMM operations: Q @ K^T, softmax(A) @ V
- Non-GEMM operations: max (reduce), exp, subtraction, division, sum (reduce)

Each architecture handles these operations differently, leading to different
performance characteristics in terms of:
- Data movement (on-chip vs off-chip)
- Execution latency
- Energy consumption
- Hardware complexity

Usage:
    python attention_accelerator_comparison.py

Author: FuseX Team
Date: 2025
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tileflow import accelerator
from tileflow.dataflows import self_attention_no_fuse
import domino.program_ir as dir


def compare_attention_accelerators():
    """
    Compare three different accelerator architectures for self-attention.

    This function demonstrates how to set up and use the three different
    accelerator types for modeling attention operations.
    """

    # Attention workload parameters
    batch_size = 1
    num_heads = 12
    seq_len = 512
    hidden_dim = 768

    print("="*80)
    print("Self-Attention Accelerator Architecture Comparison")
    print("="*80)
    print(f"\nWorkload Parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Head dimension: {hidden_dim // num_heads}")
    print()

    # ========================================================================
    # Architecture 1: Baseline (Systolic Array + CPU)
    # ========================================================================
    print("\n" + "="*80)
    print("1. Baseline Architecture: Systolic Array + CPU")
    print("="*80)
    print("\nCharacteristics:")
    print("  - 32x32 systolic array for GEMM operations (Q@K^T, softmax@V)")
    print("  - Non-GEMM ops (max, exp, div) sent to CPU via DRAM")
    print("  - High data movement penalty for non-GEMM operations")
    print("  - Lower hardware complexity")
    print("\nMemory Hierarchy:")
    print("  L0 (Reg):  6 entries/PE, 3 GB/s R/W")
    print("  L1 (SRAM): 4000 KB, 500 GB/s read, 200 GB/s write")
    print("  L2 (DRAM): 1.6 TB, 25 GB/s read, 10 GB/s write (CPU access)")

    baseline_edge = accelerator.get_baseline_attention_edge(L1_BW=500, L2_BW=25)
    baseline_cloud = accelerator.get_baseline_attention_cloud(
        L1_BW=4000, L2_BW=800, L3_BW=160
    )

    print("\nAccelerator configurations created:")
    print(f"  - Baseline Edge: {baseline_edge.name}")
    print(f"  - Baseline Cloud: {baseline_cloud.name}")

    # ========================================================================
    # Architecture 2: TPU-style (Systolic Array + VPU)
    # ========================================================================
    print("\n" + "="*80)
    print("2. TPU-style Architecture: Systolic Array + VPU")
    print("="*80)
    print("\nCharacteristics:")
    print("  - 32x32 systolic array for GEMM operations")
    print("  - On-chip VPU for non-GEMM operations (max, exp, div)")
    print("  - Data stays on-chip between systolic array and VPU")
    print("  - Reduced off-chip data movement compared to baseline")
    print("  - Medium hardware complexity")
    print("\nMemory Hierarchy:")
    print("  L0 (Reg):     6 entries/PE, 3 GB/s R/W")
    print("  L1 (SRAM):    4000 KB, 500 GB/s read, 200 GB/s write (systolic)")
    print("  L1_VPU (SRAM): 2000 KB, 400 GB/s read, 160 GB/s write (VPU)")
    print("  L2 (DRAM):    1.6 TB, 25 GB/s read, 10 GB/s write")

    tpu_edge = accelerator.get_tpu_attention_edge(
        L1_BW=500, L1_VPU_BW=400, L2_BW=25
    )
    tpu_cloud = accelerator.get_tpu_attention_cloud(
        L1_BW=4000, L1_VPU_BW=3000, L2_BW=800, L3_BW=160
    )

    print("\nAccelerator configurations created:")
    print(f"  - TPU-style Edge: {tpu_edge.name}")
    print(f"  - TPU-style Cloud: {tpu_cloud.name}")

    # ========================================================================
    # Architecture 3: UniFlow (Unified PE)
    # ========================================================================
    print("\n" + "="*80)
    print("3. UniFlow Architecture: Unified PE")
    print("="*80)
    print("\nCharacteristics:")
    print("  - 32x32 unified PEs supporting BOTH GEMM and non-GEMM operations")
    print("  - Each PE contains MAC unit AND vector ALU (max, exp, div, etc.)")
    print("  - No data movement between different processing units")
    print("  - Seamless execution of entire attention pipeline on same hardware")
    print("  - Higher hardware complexity per PE, but simpler data management")
    print("\nMemory Hierarchy:")
    print("  L0 (Reg):  12 entries/PE, 4 GB/s R/W (enhanced for unified ops)")
    print("  L1 (SRAM): 4000 KB, 500 GB/s read, 200 GB/s write (unified buffer)")
    print("  L2 (DRAM): 1.6 TB, 25 GB/s read, 10 GB/s write")

    uniflow_edge = accelerator.get_uniflow_attention_edge(L1_BW=500, L2_BW=25)
    uniflow_cloud = accelerator.get_uniflow_attention_cloud(
        L1_BW=4000, L2_BW=800, L3_BW=160
    )

    print("\nAccelerator configurations created:")
    print(f"  - UniFlow Edge: {uniflow_edge.name}")
    print(f"  - UniFlow Cloud: {uniflow_cloud.name}")

    # ========================================================================
    # Dataflow Setup
    # ========================================================================
    print("\n" + "="*80)
    print("Self-Attention Dataflow Breakdown")
    print("="*80)
    print("\nThe self-attention operation consists of the following stages:")
    print("\n1. GEMM: A = Q @ K^T  [GEMM operation]")
    print("   - Dimensions: [batch, heads, seq_len, seq_len]")
    print("   - Executed on: Systolic array (all architectures)")
    print()
    print("2. Max Reduction: B = max(A, dim=-1)  [Non-GEMM]")
    print("   - Dimensions: [batch, heads, seq_len]")
    print("   - Baseline: Sent to CPU via DRAM")
    print("   - TPU: Executed on VPU (on-chip)")
    print("   - UniFlow: Executed on unified PE")
    print()
    print("3. Subtraction: C = A - B  [Non-GEMM]")
    print("   - Element-wise operation")
    print("   - Execution same as max reduction")
    print()
    print("4. Exponentiation: D = exp(C)  [Non-GEMM]")
    print("   - Element-wise operation")
    print("   - Execution same as max reduction")
    print()
    print("5. Sum Reduction: E = sum(D, dim=-1)  [Non-GEMM]")
    print("   - Dimensions: [batch, heads, seq_len]")
    print("   - Execution same as max reduction")
    print()
    print("6. Division: F = D / E  [Non-GEMM]")
    print("   - Softmax normalization")
    print("   - Execution same as max reduction")
    print()
    print("7. GEMM: G = F @ V  [GEMM operation]")
    print("   - Final attention output")
    print("   - Executed on: Systolic array (all architectures)")

    # Get dataflow for 2-level memory hierarchy (edge devices)
    dataflow_2level = self_attention_no_fuse.get_self_attention_no_fuse_dataflow(
        levels=2,
        batch=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        hidden=hidden_dim,
        define_tiling_space=True
    )

    # Get dataflow for 3-level memory hierarchy (cloud devices)
    dataflow_3level = self_attention_no_fuse.get_self_attention_no_fuse_dataflow(
        levels=3,
        batch=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        hidden=hidden_dim,
        define_tiling_space=True
    )

    print("\n" + "="*80)
    print("Summary: Key Differences Between Architectures")
    print("="*80)
    print("\n+-------------------------+------------------+------------------+-----------------+")
    print("| Metric                  | Baseline         | TPU-style        | UniFlow         |")
    print("+-------------------------+------------------+------------------+-----------------+")
    print("| GEMM execution          | Systolic array   | Systolic array   | Unified PE      |")
    print("| Non-GEMM execution      | CPU (off-chip)   | VPU (on-chip)    | Unified PE      |")
    print("| Data movement penalty   | High (DRAM)      | Low (on-chip)    | Minimal (in-PE) |")
    print("| Hardware complexity     | Low              | Medium           | High            |")
    print("| PE utilization          | Lower            | Medium           | Higher          |")
    print("| Best for                | Simple workloads | Balanced         | Attention-heavy |")
    print("+-------------------------+------------------+------------------+-----------------+")

    print("\n" + "="*80)
    print("Usage Instructions")
    print("="*80)
    print("\nTo use these accelerators in your experiments:")
    print("\n1. Import the accelerator module:")
    print("   from tileflow import accelerator")
    print("\n2. Create accelerator instance:")
    print("   acc = accelerator.get_baseline_attention_edge()")
    print("   # or get_tpu_attention_edge() / get_uniflow_attention_edge()")
    print("\n3. Define your attention dataflow:")
    print("   from tileflow.dataflows import self_attention_no_fuse")
    print("   dataflow = self_attention_no_fuse.get_self_attention_no_fuse_dataflow(...)")
    print("\n4. Run the analysis with your TileFlow toolchain")
    print("\nFor cloud-scale experiments, use the _cloud variants:")
    print("   - get_baseline_attention_cloud()")
    print("   - get_tpu_attention_cloud()")
    print("   - get_uniflow_attention_cloud()")

    print("\n" + "="*80)
    print("Expected Performance Insights")
    print("="*80)
    print("\nBased on the architecture differences, you should observe:")
    print("\n1. Baseline Architecture:")
    print("   - Highest latency due to off-chip data movement for non-GEMM ops")
    print("   - Lower energy efficiency (DRAM accesses are expensive)")
    print("   - Best case: When non-GEMM overhead is small relative to GEMM")
    print("\n2. TPU-style Architecture:")
    print("   - Medium latency with on-chip VPU processing")
    print("   - Better energy efficiency than baseline")
    print("   - Performance gap vs baseline increases with more non-GEMM ops")
    print("\n3. UniFlow Architecture:")
    print("   - Lowest latency with in-place PE execution")
    print("   - Best energy efficiency (no cross-unit data movement)")
    print("   - Highest PE utilization")
    print("   - Advantage increases with attention complexity")

    print("\n" + "="*80)
    print("Completed!")
    print("="*80)
    print()

    return {
        'baseline_edge': baseline_edge,
        'baseline_cloud': baseline_cloud,
        'tpu_edge': tpu_edge,
        'tpu_cloud': tpu_cloud,
        'uniflow_edge': uniflow_edge,
        'uniflow_cloud': uniflow_cloud,
        'dataflow_2level': dataflow_2level,
        'dataflow_3level': dataflow_3level,
    }


if __name__ == "__main__":
    # Run the comparison
    results = compare_attention_accelerators()

    print("\nAll accelerator configurations have been created successfully!")
    print("You can now use them with the TileFlow analysis framework.")
