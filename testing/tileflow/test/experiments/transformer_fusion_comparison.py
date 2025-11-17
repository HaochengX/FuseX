"""
Full Transformer Block Fusion Comparison Experiment

This experiment demonstrates the INCREMENTAL benefit of extending fusion
beyond just attention softmax to include LayerNorm, GELU, and other non-GEMM operations.

Fusion Strategies Compared:
1. no_fusion: All non-GEMM ops to CPU (Baseline architecture)
   - 21 separate stages (6 GEMM + 15 non-GEMM stages)
   - All non-GEMM operations (LayerNorm, softmax, GELU, residual) go to CPU via DRAM

2. attention_only_fusion: Only attention softmax fused (Previous work, TPU-style)
   - Softmax operations (max, exp, sum, div) pipelined on VPU
   - LayerNorm and GELU still go to CPU
   - Represents state-of-the-art before UniFlow

3. full_fusion: ALL non-GEMM fused (UniFlow)
   - LayerNorm fully fused (mean + var + normalize)
   - Attention softmax fully fused (max + exp + sum + div)
   - GELU fused with FFN
   - Residual connections fused
   - Maximum data locality

KEY INSIGHT:
UniFlow shows ADDITIONAL speedup beyond TPU-style because it can fuse
LayerNorm, GELU, and residual ops, not just attention softmax.

Expected Speedups:
- attention_only vs no_fusion: 1.3-1.8x (from softmax fusion)
- full_fusion vs attention_only: 1.2-1.5x (from LayerNorm+GELU fusion)
- full_fusion vs no_fusion: 1.5-2.7x (combined effect)

Usage:
    python transformer_fusion_comparison.py --metric=1e9/latency --trials=1000
"""

from tileflow import (
    get_baseline_attention_edge, get_baseline_attention_cloud,
    get_tpu_attention_edge, get_tpu_attention_cloud,
    get_uniflow_attention_edge, get_uniflow_attention_cloud,
    tuning, inference
)
import tileflow.dataflows as td
import domino.accelerator as acc
import argparse
import json


def run(levels, hw_config, fusion_strategy, batch, num_heads, seq_len, hidden, ff_dim, trials, metric_type,
        debug=False, resource_check=True, define_tiling_space=False):
    """Run tuning for full transformer block"""
    dataflow = td.get_transformer_block_dataflow(
        levels, fusion_strategy, batch, num_heads, seq_len, hidden, ff_dim,
        define_tiling_space=define_tiling_space)

    best_perf, best_config_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type,
        sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_config_key, best_config


# Hardware configurations with appropriate fusion strategies
hw_configs = [
    # Baseline: can only do no_fusion
    (2, get_baseline_attention_edge(), "baseline_edge", "no_fusion"),

    # TPU: can do attention_only_fusion
    (2, get_tpu_attention_edge(), "tpu_edge", "attention_only_fusion"),

    # UniFlow: can do full_fusion
    (2, get_uniflow_attention_edge(), "uniflow_edge", "full_fusion"),

    # Cloud variants
    (3, get_baseline_attention_cloud(), "baseline_cloud", "no_fusion"),
    (3, get_tpu_attention_cloud(), "tpu_cloud", "attention_only_fusion"),
    (3, get_uniflow_attention_cloud(), "uniflow_cloud", "full_fusion"),
]


# Transformer model shapes: (num_heads, seq_len, hidden, ff_dim)
shapes = [
    # BERT variants
    (12, 128, 768, 3072),   # BERT-Base (short)
    (12, 512, 768, 3072),   # BERT-Base
    (16, 512, 1024, 4096),  # BERT-Large

    # GPT-2 variants
    (12, 256, 768, 3072),   # GPT-2 Small
    (24, 512, 1024, 4096),  # GPT-2 Medium

    # LLaMA variants
    (32, 128, 4096, 11008), # LLaMA (short)
    (32, 512, 4096, 11008), # LLaMA (medium)
]


shape_names = {
    (12, 128, 768, 3072): 'BERT-Base-128',
    (12, 512, 768, 3072): 'BERT-Base-512',
    (16, 512, 1024, 4096): 'BERT-Large-512',
    (12, 256, 768, 3072): 'GPT-2-Small-256',
    (24, 512, 1024, 4096): 'GPT-2-Medium-512',
    (32, 128, 4096, 11008): 'LLaMA-128',
    (32, 512, 4096, 11008): 'LLaMA-512',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Fusion Strategy Comparison')
    parser.add_argument("--metric", type=str, default="1e9/latency",
                        help="Evaluation metric [1e9/latency, 1e9/energy]")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--begin", type=int, default=0,
                        help="Shape begin index")
    parser.add_argument("--number", type=int, default=1,
                        help="Number of shapes to evaluate")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Tuning trials")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--check_resource", default=False, action="store_true")
    parser.add_argument("--define_tiling_space", default=False, action="store_true")
    parser.add_argument("--edge_only", default=False, action="store_true")
    parser.add_argument("--cloud_only", default=False, action="store_true")

    args = parser.parse_args()

    # Filter hardware
    hw_filtered = hw_configs
    if args.edge_only:
        hw_filtered = [h for h in hw_configs if 'edge' in h[2]]
    elif args.cloud_only:
        hw_filtered = [h for h in hw_configs if 'cloud' in h[2]]

    print("="*80)
    print("Transformer Block Fusion Strategy Comparison")
    print("="*80)
    print(f"Metric: {args.metric}")
    print(f"Batch: {args.batch}")
    print(f"Trials: {args.trials}")
    print()

    print("Fusion Strategies:")
    print("  1. no_fusion:            All non-GEMM to CPU (Baseline)")
    print("  2. attention_only_fusion: Only softmax fused (Previous work)")
    print("  3. full_fusion:          All non-GEMM fused (UniFlow)")
    print()

    print("Hardware Configurations:")
    for i, (levels, hw_acc, name, fusion) in enumerate(hw_filtered):
        print(f"  [{i}] {name:20s} - {fusion:25s}")
    print()

    results_for_shape = []

    for shape in shapes[args.begin:args.begin+args.number]:
        num_heads, seq_len, hidden, ff_dim = shape
        shape_name = shape_names.get(shape, f"{num_heads}h_{seq_len}s_{hidden}h_{ff_dim}ff")

        print("="*80)
        print(f"Model: {shape_name}")
        print(f"  Heads: {num_heads}, Seq: {seq_len}, Hidden: {hidden}, FF: {ff_dim}")
        print("="*80)

        results = []

        for levels, hw_acc, hw_name, fusion_strategy in hw_filtered:
            hw_id = hw_configs.index((levels, hw_acc, hw_name, fusion_strategy))

            print(f"\n[{hw_id}] {hw_name} with {fusion_strategy}")

            hw_config = acc.tileflow_accelerator_generator(hw_acc)

            try:
                perf, key, config = run(
                    levels, hw_config, fusion_strategy, args.batch, num_heads, seq_len, hidden, ff_dim,
                    args.trials, metric_type=args.metric, debug=args.debug,
                    resource_check=args.check_resource, define_tiling_space=args.define_tiling_space)

                if perf is None:
                    print(f"    WARNING: No valid configuration found! Skipping...")
                    continue

                results.append((perf, key, config, hw_name, fusion_strategy, hw_id))
                print(f"    Performance: {perf:.2f} {args.metric}")

            except Exception as e:
                print(f"    ERROR: {e}")
                print(f"    Skipping...")
                continue

        results_for_shape.append((shape, shape_name, results))

    # Check results
    total_results = sum(len(results) for _, _, results in results_for_shape)
    if total_results == 0:
        print("\n" + "="*80)
        print("ERROR: No valid results collected!")
        print("="*80)
        print("\nTry: --trials=100 or higher")
        exit(1)

    # Print results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print("batch,seq_len,num_heads,hidden,ff_dim,metric,hw_name,fusion_strategy,hw_id,perf")

    for shape, shape_name, results in results_for_shape:
        if len(results) == 0:
            continue
        for res in results:
            perf, key, config, hw_name, fusion_strategy, hw_id = res
            print(
                f"{args.batch},{shape[1]},{shape[0]},{shape[2]},{shape[3]},"
                f"{args.metric},{hw_name},{fusion_strategy},{hw_id},{perf}")

    # Print fusion benefit analysis
    print("\n" + "="*80)
    print("FUSION BENEFIT ANALYSIS")
    print("="*80)

    for shape, shape_name, results in results_for_shape:
        if len(results) == 0:
            continue

        print(f"\nModel: {shape_name}")
        print("-" * 80)
        print(f"{'Strategy':<30} {'Hardware':<20} {'Perf':<15} {'vs Baseline':<15} {'vs Attn-Only'}")
        print("-" * 80)

        # Find baseline and attention-only performance
        no_fusion_perf = None
        attention_only_perf = None
        full_fusion_perf = None

        perf_map = {}
        for res in results:
            perf, key, config, hw_name, fusion_strategy, hw_id = res
            perf_map[fusion_strategy] = perf

        no_fusion_perf = perf_map.get('no_fusion')
        attention_only_perf = perf_map.get('attention_only_fusion')
        full_fusion_perf = perf_map.get('full_fusion')

        for res in results:
            perf, key, config, hw_name, fusion_strategy, hw_id = res

            vs_baseline = perf / no_fusion_perf if no_fusion_perf and no_fusion_perf > 0 else 1.0
            vs_attn = perf / attention_only_perf if attention_only_perf and attention_only_perf > 0 else 1.0

            print(f"{fusion_strategy:<30} {hw_name:<20} {perf:<15.2f} {vs_baseline:<15.2f}x {vs_attn:.2f}x")

        # Print key insights
        if no_fusion_perf and attention_only_perf and full_fusion_perf:
            print()
            print("KEY INSIGHTS:")
            attn_benefit = attention_only_perf / no_fusion_perf
            layernorm_gelu_benefit = full_fusion_perf / attention_only_perf
            total_benefit = full_fusion_perf / no_fusion_perf

            print(f"  1. Softmax fusion benefit:           {attn_benefit:.2f}x")
            print(f"  2. LayerNorm+GELU fusion benefit:    {layernorm_gelu_benefit:.2f}x")
            print(f"  3. Total UniFlow benefit:            {total_benefit:.2f}x")
            print()
            print(f"  UniFlow advantage = {layernorm_gelu_benefit:.2f}x beyond previous work!")

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)
