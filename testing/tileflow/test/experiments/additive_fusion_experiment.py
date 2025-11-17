"""
Additive Fusion Benefits Experiment

This experiment demonstrates the STACKABLE benefits of three orthogonal optimizations:
1. **Base Fusion**: Fusing non-GEMM operations (UniFlow vs TPU vs Baseline)
2. **PE Partition**: Spatial partitioning for GEMM/non-GEMM overlap
3. **KV-Aware Fusion**: Cache-aware dataflows for prefill/decode

Key Research Questions:
- How much does each optimization contribute independently?
- Are the benefits additive/multiplicative?
- How do benefits vary across prefill vs decode phases?
- Which optimization matters most for different sequence lengths?

Experimental Matrix:
- Architectures: 4 variants
  * Baseline (no fusion, no partition)
  * TPU (partial fusion, no partition)
  * UniFlow (full fusion, no partition)
  * UniFlow+Partition (full fusion + PE partition)

- Models: LLaMA-3 and GPT-3 variants
  * Sequence lengths: 128, 512, 1024, 2048, 4096

- Phases: Prefill and Decode
  * Prefill: Process full prompt (compute-bound)
  * Decode: Generate tokens incrementally (memory-bound)

Expected Speedup Stack:
- Base: 1.0x (baseline)
- +Fusion: 1.5-1.8x (TPU partial fusion)
- +Full Fusion: 1.8-2.2x (UniFlow full fusion)
- +PE Partition: 2.2-2.8x (spatial pipelining)
- +KV-Aware: 2.8-3.5x (cache-optimized decode)

Output: CSV files showing incremental and cumulative benefits
"""

import sys
import csv
import argparse
from datetime import datetime
from tileflow import (
    get_baseline_attention_edge, get_baseline_attention_cloud,
    get_tpu_attention_edge, get_tpu_attention_cloud,
    get_uniflow_attention_edge, get_uniflow_attention_cloud,
    get_uniflow_partition_edge, get_uniflow_partition_cloud,
    tuning, inference
)
import tileflow.dataflows as td
import domino.accelerator as acc


def run_attention_fusion(levels, hw_config, fusion_strategy, batch, num_heads, seq_len, hidden,
                          trials, metric_type, debug=False, resource_check=True, define_tiling_space=False):
    """Run standard attention with fusion strategy"""
    dataflow = td.get_attention_fusion_dataflow(
        levels, fusion_strategy, batch, num_heads, seq_len, hidden, define_tiling_space=define_tiling_space)

    best_perf, best_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type,
        sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_key, best_config


def run_attention_partition(levels, hw_config, batch, num_heads, seq_len, hidden,
                             trials, metric_type, debug=False, resource_check=True, define_tiling_space=False):
    """Run attention with PE partition"""
    dataflow = td.get_attention_partition_dataflow(
        levels, batch, num_heads, seq_len, hidden, define_tiling_space=define_tiling_space)

    best_perf, best_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type,
        sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_key, best_config


def run_attention_kv_cache(levels, hw_config, phase, batch, num_heads, seq_len, hidden,
                            context_len, cache_strategy, trials, metric_type, debug=False, resource_check=True,
                            define_tiling_space=False):
    """Run attention with KV-cache awareness"""
    dataflow = td.get_attention_kv_cache_dataflow(
        levels, phase, batch, num_heads, seq_len, hidden,
        cache_strategy=cache_strategy, context_len=context_len, define_tiling_space=define_tiling_space)

    best_perf, best_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type,
        sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_key, best_config


# Architecture configurations
# Each entry: (levels, hw_fn, name, supports_partition, fusion_strategy)
hw_configs = [
    # Baseline: No fusion, no partition
    (2, get_baseline_attention_edge, "baseline_edge", False, "no_fusion"),
    (3, get_baseline_attention_cloud, "baseline_cloud", False, "no_fusion"),

    # TPU: Partial fusion, no partition
    (2, get_tpu_attention_edge, "tpu_edge", False, "partial_fusion"),
    (3, get_tpu_attention_cloud, "tpu_cloud", False, "partial_fusion"),

    # UniFlow: Full fusion, no partition
    (2, get_uniflow_attention_edge, "uniflow_edge", False, "full_fusion"),
    (3, get_uniflow_attention_cloud, "uniflow_cloud", False, "full_fusion"),

    # UniFlow+Partition: Full fusion WITH partition
    (2, get_uniflow_partition_edge, "uniflow_partition_edge", True, "full_fusion"),
    (3, get_uniflow_partition_cloud, "uniflow_partition_cloud", True, "full_fusion"),
]


# LLaMA-3 model configurations: (num_heads, hidden, ff_dim, model_name)
llama3_models = [
    (32, 4096, 11008, "LLaMA-3-8B"),   # 8B parameter model
    (40, 5120, 13824, "LLaMA-3-70B"),  # 70B parameter model (if supported)
]


# GPT-3 model configurations: (num_heads, hidden, ff_dim, model_name)
gpt3_models = [
    (12, 768, 3072, "GPT-3-125M"),    # GPT-3 Small
    (24, 1024, 4096, "GPT-3-1.3B"),   # GPT-3 Medium
    (96, 12288, 49152, "GPT-3-175B"), # GPT-3 (if supported)
]


# Sequence lengths to evaluate
sequence_lengths = [128, 512, 1024, 2048, 4096]


def run_experiments(args):
    """Run the full experimental sweep"""

    # Select models
    if args.model_family == "llama3":
        models = llama3_models
    elif args.model_family == "gpt3":
        models = gpt3_models
    elif args.model_family == "both":
        models = llama3_models + gpt3_models
    else:
        raise ValueError(f"Unknown model family: {args.model_family}")

    # Filter hardware
    hw_filtered = hw_configs
    if args.edge_only:
        hw_filtered = [h for h in hw_configs if 'edge' in h[2]]
    elif args.cloud_only:
        hw_filtered = [h for h in hw_configs if 'cloud' in h[2]]

    print("="*80)
    print("ADDITIVE FUSION BENEFITS EXPERIMENT")
    print("="*80)
    print(f"Model Family: {args.model_family}")
    print(f"Sequence Lengths: {sequence_lengths}")
    print(f"Metric: {args.metric}")
    print(f"Batch: {args.batch}")
    print(f"Trials: {args.trials}")
    print(f"Phase: {args.phase}")
    print()

    # Results storage
    results = []

    for model_config in models:
        num_heads, hidden, ff_dim, model_name = model_config
        print(f"\n{'='*80}")
        print(f"Model: {model_name} (heads={num_heads}, hidden={hidden})")
        print(f"{'='*80}")

        for seq_len in sequence_lengths:
            print(f"\n  Sequence Length: {seq_len}")
            print(f"  {'-'*76}")

            for levels, hw_fn, hw_name, supports_partition, fusion_strategy in hw_filtered:
                print(f"\n    [{hw_name}]")

                hw_config = acc.tileflow_accelerator_generator(hw_fn())

                try:
                    if args.phase == "prefill":
                        # Prefill: Full attention over entire sequence
                        if supports_partition:
                            # Use PE partition dataflow
                            perf, key, config = run_attention_partition(
                                levels, hw_config, args.batch, num_heads, seq_len, hidden,
                                args.trials, args.metric, args.debug, args.check_resource, args.define_tiling_space)
                            variant = "partition"
                        else:
                            # Use fusion dataflow
                            perf, key, config = run_attention_fusion(
                                levels, hw_config, fusion_strategy, args.batch, num_heads, seq_len, hidden,
                                args.trials, args.metric, args.debug, args.check_resource, args.define_tiling_space)
                            variant = "fusion_only"

                        phase_info = f"prefill_seq{seq_len}"

                    elif args.phase == "decode":
                        # Decode: Generate one token with KV-cache
                        context_len = seq_len  # Assume we've already generated seq_len tokens

                        # Use appropriate KV-cache strategy based on partition support
                        cache_strategy = "stream" if supports_partition else "static"
                        perf, key, config = run_attention_kv_cache(
                            levels, hw_config, "decode", args.batch, num_heads, 1, hidden,
                            context_len, cache_strategy, args.trials, args.metric, args.debug, args.check_resource,
                            args.define_tiling_space)
                        variant = f"kv_{cache_strategy}"
                        phase_info = f"decode_{cache_strategy}_ctx{context_len}"

                    elif args.phase == "both":
                        # Run both prefill and decode
                        # Prefill
                        if supports_partition:
                            perf_prefill, key_prefill, _ = run_attention_partition(
                                levels, hw_config, args.batch, num_heads, seq_len, hidden,
                                args.trials, args.metric, args.debug, args.check_resource, args.define_tiling_space)
                        else:
                            perf_prefill, key_prefill, _ = run_attention_fusion(
                                levels, hw_config, fusion_strategy, args.batch, num_heads, seq_len, hidden,
                                args.trials, args.metric, args.debug, args.check_resource, args.define_tiling_space)

                        # Decode (with appropriate KV-cache strategy)
                        context_len = seq_len
                        cache_strategy = "stream" if supports_partition else "static"
                        perf_decode, key_decode, _ = run_attention_kv_cache(
                            levels, hw_config, "decode", args.batch, num_heads, 1, hidden,
                            context_len, cache_strategy, args.trials, args.metric, args.debug, args.check_resource,
                            args.define_tiling_space)

                        # Store both results
                        results.append({
                            'model': model_name,
                            'num_heads': num_heads,
                            'hidden': hidden,
                            'seq_len': seq_len,
                            'hw_name': hw_name,
                            'fusion_strategy': fusion_strategy,
                            'supports_partition': supports_partition,
                            'cache_strategy': 'none',  # Prefill doesn't use cache
                            'phase': 'prefill',
                            'performance': perf_prefill if isinstance(perf_prefill, (int, float)) else 'N/A',
                            'config_key': key_prefill,
                        })
                        results.append({
                            'model': model_name,
                            'num_heads': num_heads,
                            'hidden': hidden,
                            'seq_len': seq_len,
                            'hw_name': hw_name,
                            'fusion_strategy': fusion_strategy,
                            'supports_partition': supports_partition,
                            'cache_strategy': cache_strategy,  # 'stream' or 'static'
                            'phase': 'decode',
                            'performance': perf_decode if isinstance(perf_decode, (int, float)) else 'N/A',
                            'config_key': key_decode,
                        })

                        print(f"      Prefill: {perf_prefill if isinstance(perf_prefill, (int, float)) else 'N/A'}")
                        print(f"      Decode ({cache_strategy}):  {perf_decode if isinstance(perf_decode, (int, float)) else 'N/A'}")
                        continue

                    else:
                        raise ValueError(f"Invalid phase: {args.phase}")

                    if perf is None:
                        print(f"      WARNING: No valid configuration found!")
                        continue

                    perf_value = perf if isinstance(perf, (int, float)) else None

                    results.append({
                        'model': model_name,
                        'num_heads': num_heads,
                        'hidden': hidden,
                        'seq_len': seq_len,
                        'hw_name': hw_name,
                        'fusion_strategy': fusion_strategy,
                        'supports_partition': supports_partition,
                        'phase': phase_info,
                        'performance': perf_value if perf_value else 'N/A',
                        'config_key': key,
                    })

                    print(f"      Performance: {perf_value if perf_value else 'N/A'} {args.metric}")

                except Exception as e:
                    print(f"      ERROR: {e}")
                    continue

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"additive_fusion_results_{args.phase}_{timestamp}.csv"

    with open(csv_file, 'w', newline='') as f:
        if len(results) == 0:
            print("\nERROR: No results to save!")
            return

        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_file}")
    print(f"{'='*80}")

    # Print summary analysis
    print_additive_analysis(results, args.metric)


def print_additive_analysis(results, metric):
    """Print analysis showing additive benefits"""
    print(f"\n{'='*80}")
    print("ADDITIVE BENEFIT ANALYSIS")
    print(f"{'='*80}")

    # Group by model, sequence length, and phase
    grouped = {}
    for r in results:
        phase = r.get('phase', 'unknown')
        cache_strategy = r.get('cache_strategy', 'none')
        key = (r['model'], r['seq_len'], phase)
        if key not in grouped:
            grouped[key] = {}

        # Create unique identifier that includes cache strategy for decode
        hw_key = r['hw_name']
        if phase == 'decode' and cache_strategy != 'none':
            hw_key = f"{hw_key}_{cache_strategy}"

        grouped[key][hw_key] = r['performance']

    for (model, seq_len, phase), hw_perfs in grouped.items():
        print(f"\n{model} @ seq_len={seq_len}, phase={phase}")
        print("-" * 80)

        # Get baseline performance
        baseline_perf = hw_perfs.get('baseline_edge') or hw_perfs.get('baseline_cloud')
        if not baseline_perf or baseline_perf == 'N/A':
            continue

        baseline_perf = float(baseline_perf)

        # Calculate incremental speedups
        speedups = {}
        for hw_name, perf in hw_perfs.items():
            if perf != 'N/A':
                speedups[hw_name] = float(perf) / baseline_perf

        # Print in order of increasing capability
        if phase == 'decode':
            order = ['baseline', 'tpu', 'uniflow', 'uniflow_partition_static', 'uniflow_partition_stream']
        else:
            order = ['baseline', 'tpu', 'uniflow', 'uniflow_partition']

        for prefix in order:
            for hw_name, speedup in speedups.items():
                if prefix in hw_name:
                    print(f"  {hw_name:40s}  {speedup:6.2f}x")

        # Calculate additive benefits
        tpu_perf = hw_perfs.get('tpu_edge') or hw_perfs.get('tpu_cloud')
        uniflow_perf = hw_perfs.get('uniflow_edge') or hw_perfs.get('uniflow_cloud')

        if phase == 'decode':
            partition_static_perf = hw_perfs.get('uniflow_partition_edge_static') or hw_perfs.get('uniflow_partition_cloud_static')
            partition_stream_perf = hw_perfs.get('uniflow_partition_edge_stream') or hw_perfs.get('uniflow_partition_cloud_stream')
        else:
            partition_perf = hw_perfs.get('uniflow_partition_edge') or hw_perfs.get('uniflow_partition_cloud')

        print()
        if tpu_perf and tpu_perf != 'N/A' and uniflow_perf and uniflow_perf != 'N/A':
            fusion_benefit = float(uniflow_perf) / float(tpu_perf)
            print(f"  1. Fusion benefit (UniFlow vs TPU):        {fusion_benefit:.2f}x")

        if phase == 'decode':
            if partition_static_perf and partition_static_perf != 'N/A' and uniflow_perf and uniflow_perf != 'N/A':
                partition_benefit = float(partition_static_perf) / float(uniflow_perf)
                print(f"  2. Partition benefit (vs UniFlow):         {partition_benefit:.2f}x")

            if partition_stream_perf and partition_stream_perf != 'N/A' and partition_static_perf and partition_static_perf != 'N/A':
                stream_benefit = float(partition_stream_perf) / float(partition_static_perf)
                print(f"  3. KV-Stream benefit (vs static):          {stream_benefit:.2f}x")

            if partition_stream_perf and partition_stream_perf != 'N/A':
                total_benefit = float(partition_stream_perf) / baseline_perf
                print(f"  → Total benefit (Stream vs Baseline):     {total_benefit:.2f}x")
        else:
            if partition_perf and partition_perf != 'N/A' and uniflow_perf and uniflow_perf != 'N/A':
                partition_benefit = float(partition_perf) / float(uniflow_perf)
                print(f"  2. Partition benefit (vs UniFlow):         {partition_benefit:.2f}x")

            if partition_perf and partition_perf != 'N/A':
                total_benefit = float(partition_perf) / baseline_perf
                print(f"  → Total benefit (Partition vs Baseline):  {total_benefit:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Additive Fusion Benefits Experiment')
    parser.add_argument("--model_family", type=str, default="llama3",
                        choices=["llama3", "gpt3", "both"],
                        help="Model family to evaluate")
    parser.add_argument("--metric", type=str, default="1e9/latency",
                        help="Evaluation metric [1e9/latency, 1e9/energy]")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (typically 1 for inference)")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["prefill", "decode", "both"],
                        help="Which phase to evaluate")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Tuning trials")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--check_resource", default=False, action="store_true")
    parser.add_argument("--define_tiling_space", default=False, action="store_true")
    parser.add_argument("--edge_only", default=False, action="store_true")
    parser.add_argument("--cloud_only", default=False, action="store_true")

    args = parser.parse_args()
    run_experiments(args)
