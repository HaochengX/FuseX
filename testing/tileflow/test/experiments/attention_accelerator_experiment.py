"""
Attention Accelerator Architecture Comparison Experiment

This experiment compares three accelerator architectures for self-attention:
1. Baseline: Systolic array + CPU (no fusion)
2. TPU-style: Systolic array + VPU (partial fusion)
3. UniFlow: Unified PE (full fusion)

Each architecture is tested with edge and cloud configurations, and
evaluated across different transformer model sizes.

Usage:
    python attention_accelerator_experiment.py --metric=1e9/latency --trials=1000 --batch=1

Results format:
    batch,seq_len,num_heads,hidden,metric,hw_name,fusion_strategy,hw_id,key,config,perf
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
import csv
from datetime import datetime


def run(levels, hw_config, fusion_strategy, batch, num_heads, seq_len, hidden, trials, metric_type,
        debug=False, resource_check=True, define_tiling_space=False):
    """Run tuning for a specific configuration"""
    dataflow = td.get_attention_fusion_dataflow(
        levels, fusion_strategy, batch, num_heads, seq_len, hidden,
        define_tiling_space=define_tiling_space)

    best_perf, best_config_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type,
        sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_config_key, best_config


def replay_config(config, hw_id, fusion_strategy, batch, num_heads, seq_len, hidden, metric_type,
                  debug=False, resource_check=True, define_tiling_space=True):
    """Replay a specific configuration"""
    levels = hw[hw_id][0]
    hw_config = acc.tileflow_accelerator_generator(hw[hw_id][1])
    dataflow = td.get_attention_fusion_dataflow(
        levels, fusion_strategy, batch, num_heads, seq_len, hidden,
        define_tiling_space=define_tiling_space)

    perf, _, _ = inference(hw_config, dataflow, [], config, metric_type,
                          resource_check=resource_check, debug=debug)
    return perf


# Hardware configurations: (levels, accelerator, name, fusion_strategy)
# Each accelerator has an associated fusion strategy that it can support
hw = [
    # Edge configurations
    (2, get_baseline_attention_edge(), "baseline_edge", "no_fusion"),
    (2, get_tpu_attention_edge(), "tpu_edge", "partial_fusion"),
    (2, get_uniflow_attention_edge(), "uniflow_edge", "full_fusion"),

    # Cloud configurations
    (3, get_baseline_attention_cloud(), "baseline_cloud", "no_fusion"),
    (3, get_tpu_attention_cloud(), "tpu_cloud", "partial_fusion"),
    (3, get_uniflow_attention_cloud(), "uniflow_cloud", "full_fusion"),
]


# Model shapes: (num_heads, seq_len, hidden)
shapes = [
    # BERT variants
    (8, 512, 512),    # BERT-Small
    (12, 512, 768),   # BERT-Base
    (16, 512, 1024),  # BERT-Large

    # Vision Transformers
    (12, 256, 768),   # ViT-Base/14
    (16, 256, 1024),  # ViT-Large/14
    (16, 256, 1280),  # ViT-Huge/14
    (12, 196, 768),   # ViT-Base/16
    (16, 196, 1024),  # ViT-Large/16
    (16, 196, 1280),  # ViT-Huge/16

    # LLaMA variants
    (32, 128, 4096),  # LLaMA-128
    (32, 512, 4096),  # LLaMA-512
    (32, 1024, 4096), # LLaMA-1024

    # GPT-3 variants
    (24, 128, 1024),  # GPT-3 Medium-128
    (24, 512, 1024),  # GPT-3 Medium-512
    (24, 1024, 1024), # GPT-3 Medium-1024
]


shape_names = {
    (8, 512, 512): 'BERT-Small',
    (12, 512, 768): 'BERT-Base',
    (16, 512, 1024): 'BERT-Large',
    (12, 256, 768): 'ViT-Base/14',
    (16, 256, 1024): 'ViT-Large/14',
    (16, 256, 1280): 'ViT-Huge/14',
    (12, 196, 768): 'ViT-Base/16',
    (16, 196, 1024): 'ViT-Large/16',
    (16, 196, 1280): 'ViT-Huge/16',
    (32, 128, 4096): 'LLaMA-128',
    (32, 512, 4096): 'LLaMA-512',
    (32, 1024, 4096): 'LLaMA-1024',
    (24, 128, 1024): 'GPT-3-Medium-128',
    (24, 512, 1024): 'GPT-3-Medium-512',
    (24, 1024, 1024): 'GPT-3-Medium-1024',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention Accelerator Architecture Comparison')
    parser.add_argument(
        "--metric", type=str,
        help="Evaluation metric type [1e9/latency, 1e9/energy, Utilization_L0, Utilization_L1, Utilization_L2, Utilization_L3]",
        default="1e9/latency")
    parser.add_argument("--batch", type=int,
                        help="Self attention batch size", default=1)
    parser.add_argument("--begin", type=int,
                        help="Shape begin index [0]", default=0)
    parser.add_argument("--number", type=int,
                        help="Number of shapes evaluated [1]", default=1)
    parser.add_argument("--trials", type=int,
                        help="Tuning trials [1000]", default=1000)
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--check_resource", default=False, action="store_true",
                        help="Enable resource checking")
    parser.add_argument("--define_tiling_space", default=False, action="store_true",
                        help="Define tiling space for tuning")
    parser.add_argument("--inference", default=False, action="store_true",
                        help="Run inference mode instead of tuning")
    parser.add_argument("--config_key", type=str, default="",
                        help="Manual config key for inference")
    parser.add_argument("--edge_only", default=False, action="store_true",
                        help="Only run edge configurations")
    parser.add_argument("--cloud_only", default=False, action="store_true",
                        help="Only run cloud configurations")
    parser.add_argument("--hw_filter", type=str, default="",
                        help="Filter hardware by name (baseline, tpu, uniflow)")

    args = parser.parse_args()
    batch = args.batch
    trials = args.trials
    metric_type = args.metric

    # Filter hardware configurations based on arguments
    hw_filtered = hw
    if args.edge_only:
        hw_filtered = [h for h in hw if 'edge' in h[2]]
    elif args.cloud_only:
        hw_filtered = [h for h in hw if 'cloud' in h[2]]

    if args.hw_filter:
        hw_filtered = [h for h in hw_filtered if args.hw_filter in h[2]]

    if args.inference:
        # Inference mode (not implemented yet)
        print("Inference mode not yet implemented. Use tuning mode.")
    else:
        # Tuning mode
        print("="*80)
        print("Attention Accelerator Architecture Comparison Experiment")
        print("="*80)
        print(f"Metric: {metric_type}")
        print(f"Batch size: {batch}")
        print(f"Trials per configuration: {trials}")
        print(f"Shapes to evaluate: {args.number} starting from index {args.begin}")
        print(f"Hardware configurations: {len(hw_filtered)}")
        print()

        # Print hardware configuration summary
        print("Hardware Configurations:")
        for i, (levels, hw_acc, name, fusion) in enumerate(hw_filtered):
            original_idx = hw.index((levels, hw_acc, name, fusion))
            print(f"  [{original_idx}] {name:20s} - {fusion:20s} ({levels} levels)")
        print()

        results_for_shape = []
        for shape in shapes[args.begin:args.begin+args.number]:
            results = []
            num_heads = shape[0]
            seq_len = shape[1]
            hidden = shape[2]
            shape_name = shape_names.get(shape, f"{num_heads}h_{seq_len}s_{hidden}d")

            print("="*80)
            print(f"Evaluating: {shape_name} (heads={num_heads}, seq_len={seq_len}, hidden={hidden})")
            print("="*80)

            for levels, hw_acc, hw_name, fusion_strategy in hw_filtered:
                # Get original index for logging
                hw_id = hw.index((levels, hw_acc, hw_name, fusion_strategy))

                print(f"\n[{hw_id}] Running: {hw_name} with {fusion_strategy}")
                print(f"    Architecture: {hw_name}")
                print(f"    Fusion: {fusion_strategy}")
                print(f"    Levels: {levels}")

                hw_config = acc.tileflow_accelerator_generator(hw_acc)

                try:
                    perf, key, config = run(
                        levels, hw_config, fusion_strategy, batch, num_heads, seq_len, hidden,
                        trials, metric_type=metric_type, debug=args.debug,
                        resource_check=args.check_resource, define_tiling_space=args.define_tiling_space)

                    if perf is None:
                        print(f"    WARNING: No valid configuration found! Skipping...")
                        continue

                    results.append((perf, key, config, hw_name, fusion_strategy, hw_id))

                    print(f"    Performance: {perf:.2f} {metric_type}")
                    print(f"    Config: {key}")

                except Exception as e:
                    print(f"    ERROR: Tuning failed with error: {e}")
                    print(f"    Skipping this configuration...")
                    continue

            results_for_shape.append((shape, shape_name, results))

        # Check if we have any results
        total_results = sum(len(results) for _, _, results in results_for_shape)
        if total_results == 0:
            print("\n" + "="*80)
            print("ERROR: No valid results were collected!")
            print("="*80)
            print("\nPossible causes:")
            print("  1. Trials too low (try --trials=100 or higher)")
            print("  2. Resource constraints violated (try --check_resource=False)")
            print("  3. Dataflow import issues")
            print("  4. Hardware configuration mismatch")
            print("\nTry running with --debug flag for more information.")
            exit(1)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to CSV
        detailed_csv = f"attention_results_detailed_{timestamp}.csv"
        with open(detailed_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'seq_len', 'num_heads', 'hidden', 'metric',
                           'hw_name', 'fusion_strategy', 'hw_id', 'config_key', 'performance'])

            for shape, shape_name, results in results_for_shape:
                if len(results) == 0:
                    continue
                for res in results:
                    perf, key, config, hw_name, fusion_strategy, hw_id = res
                    # Handle perf if it's a dict (extract numeric value)
                    perf_value = perf if isinstance(perf, (int, float)) else str(perf)
                    writer.writerow([
                        batch, shape[1], shape[0], shape[2], metric_type,
                        hw_name, fusion_strategy, hw_id, key, perf_value
                    ])

        print(f"\n✓ Detailed results saved to: {detailed_csv}")

        # Save comparison summary to CSV
        summary_csv = f"attention_results_summary_{timestamp}.csv"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name', 'num_heads', 'seq_len', 'hidden',
                           'hw_name', 'fusion_strategy', 'performance', 'speedup_vs_baseline'])

            for shape, shape_name, results in results_for_shape:
                if len(results) == 0:
                    continue

                # Find baseline performance for comparison
                baseline_perf = None
                for res in results:
                    perf, key, config, hw_name, fusion_strategy, hw_id = res
                    if 'baseline' in hw_name:
                        # Handle perf if it's a dict
                        baseline_perf = perf if isinstance(perf, (int, float)) else None
                        break

                for res in results:
                    perf, key, config, hw_name, fusion_strategy, hw_id = res
                    # Handle perf if it's a dict
                    perf_value = perf if isinstance(perf, (int, float)) else None

                    if perf_value is not None and baseline_perf is not None and baseline_perf > 0:
                        speedup = perf_value / baseline_perf
                    else:
                        speedup = 1.0

                    writer.writerow([
                        shape_name, shape[0], shape[1], shape[2],
                        hw_name, fusion_strategy, perf_value if perf_value else 'N/A', speedup
                    ])

        print(f"✓ Summary results saved to: {summary_csv}")

        # Print quick summary to console
        print("\n" + "="*80)
        print("QUICK SUMMARY")
        print("="*80)

        for shape, shape_name, results in results_for_shape:
            if len(results) == 0:
                print(f"\n{shape_name}: No valid results")
                continue

            print(f"\n{shape_name}:")
            for res in results:
                perf, key, config, hw_name, fusion_strategy, hw_id = res
                perf_value = perf if isinstance(perf, (int, float)) else 'N/A'
                print(f"  {hw_name:25s} {fusion_strategy:20s} {perf_value}")

        print("\n" + "="*80)
        print("Experiment completed successfully!")
        print(f"Results saved to:")
        print(f"  - {detailed_csv}")
        print(f"  - {summary_csv}")
        print("="*80)
