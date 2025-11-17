# Additive Fusion Benefits Experiment

This document explains the experimental framework for evaluating **stackable optimizations** in attention accelerators:
1. **Base Fusion**: Fusing non-GEMM operations (softmax, layernorm, etc.)
2. **PE Partition**: Spatial partitioning for concurrent GEMM/non-GEMM execution
3. **KV-Aware**: Cache-aware dataflows for prefill/decode phases

## Architecture Variants

### 1. Baseline
- **Hardware**: Systolic array for GEMM + CPU for non-GEMM
- **Fusion**: None (sequential execution)
- **Partition**: No
- **Performance**: 1.0x (baseline)

### 2. TPU-style
- **Hardware**: Systolic array for GEMM + on-chip VPU for non-GEMM
- **Fusion**: Partial (softmax operations pipelined)
- **Partition**: No
- **Performance**: ~1.5-1.8x vs baseline

### 3. UniFlow
- **Hardware**: Unified PEs supporting both GEMM and non-GEMM
- **Fusion**: Full (all non-GEMM ops fused)
- **Partition**: No
- **Performance**: ~1.8-2.2x vs baseline

### 4. UniFlow+Partition
- **Hardware**: Spatially partitioned PEs (75% GEMM, 25% non-GEMM)
- **Fusion**: Full
- **Partition**: Yes (concurrent GEMM and non-GEMM execution)
- **Performance**: ~2.2-2.8x vs baseline

### 5. UniFlow+Partition+KV-Aware
- **Hardware**: Same as #4
- **Fusion**: Full + KV-cache optimized dataflows
- **Partition**: Yes
- **KV-Aware**: Separate prefill/decode with cache reuse
- **Performance**: ~2.8-3.5x vs baseline (especially for decode)

## Model Configurations

### LLaMA-3
- **8B model**: 32 heads, hidden=4096, ff=11008
- **70B model**: 40 heads, hidden=5120, ff=13824

### GPT-3
- **125M model**: 12 heads, hidden=768, ff=3072
- **1.3B model**: 24 heads, hidden=1024, ff=4096
- **175B model**: 96 heads, hidden=12288, ff=49152

## Sequence Lengths
- 128 tokens (short prompts)
- 512 tokens (standard chat)
- 1024 tokens (medium context)
- 2048 tokens (extended context)
- 4096 tokens (long context)

## Prefill vs Decode

### Prefill Phase
- **Task**: Process entire prompt to generate first token
- **Characteristics**:
  - Large batch of Q, K, V computations (seq_len × seq_len)
  - Compute-bound (many GEMM operations)
  - One-time cost per prompt
- **Key Optimization**: Fusion + PE partition for overlapping GEMM and softmax
- **Performance**: Benefits most from fusion and partition

### Decode Phase
- **Task**: Generate one token at a time using cached K, V
- **Characteristics**:
  - Small Q (1×hidden), large K, V cache (context_len×hidden)
  - Memory-bound (cache reads dominate)
  - Repeated for each generated token
- **Key Optimization**: KV-cache awareness to minimize data movement
- **Performance**: Benefits most from cache-aware dataflows

## Usage Examples

### 1. Quick Test (Edge, LLaMA-3, Both Phases)
```bash
python additive_fusion_experiment.py \
  --model_family=llama3 \
  --phase=both \
  --edge_only \
  --trials=100 \
  --metric=1e9/latency
```

### 2. Full LLaMA-3 Evaluation (Cloud)
```bash
python additive_fusion_experiment.py \
  --model_family=llama3 \
  --phase=both \
  --cloud_only \
  --trials=1000 \
  --metric=1e9/latency
```

### 3. Prefill-Only Analysis
```bash
python additive_fusion_experiment.py \
  --model_family=both \
  --phase=prefill \
  --trials=1000 \
  --metric=1e9/latency
```

### 4. Decode-Only Analysis (Cache-aware)
```bash
python additive_fusion_experiment.py \
  --model_family=both \
  --phase=decode \
  --trials=1000 \
  --metric=1e9/latency
```

### 5. Energy Efficiency Study
```bash
python additive_fusion_experiment.py \
  --model_family=llama3 \
  --phase=both \
  --trials=1000 \
  --metric=1e9/energy
```

## Output Files

The experiment generates timestamped CSV files:

### `additive_fusion_results_<phase>_<timestamp>.csv`

Columns:
- `model`: Model name (e.g., "LLaMA-3-8B")
- `num_heads`: Number of attention heads
- `hidden`: Hidden dimension per head
- `seq_len`: Sequence length
- `hw_name`: Hardware configuration (e.g., "uniflow_partition_edge")
- `fusion_strategy`: Fusion strategy used
- `supports_partition`: Whether HW supports PE partition
- `phase`: Phase identifier (prefill_seq512, decode_ctx512, etc.)
- `performance`: Performance metric value
- `config_key`: Tiling configuration key

## Analyzing Results

### Key Metrics to Compare

1. **Fusion Benefit**: UniFlow vs TPU
   - Shows benefit of full fusion over partial fusion
   - Expected: 1.2-1.5x

2. **Partition Benefit**: UniFlow+Partition vs UniFlow
   - Shows benefit of spatial pipelining
   - Expected: 1.2-1.4x

3. **KV-Aware Benefit**: Decode with cache vs without
   - Shows benefit of cache-aware dataflows
   - Expected: 1.1-1.3x (decode phase only)

4. **Total Benefit**: UniFlow+Partition+KV vs Baseline
   - Shows combined effect of all optimizations
   - Expected: 2.5-3.5x

### Example Analysis (Python)

```python
import pandas as pd

# Load results
df = pd.read_csv('additive_fusion_results_both_<timestamp>.csv')

# Filter for specific model and sequence length
llama_512 = df[(df['model'] == 'LLaMA-3-8B') & (df['seq_len'] == 512)]

# Get baseline performance
baseline_prefill = llama_512[(llama_512['hw_name'].str.contains('baseline')) &
                             (llama_512['phase'] == 'prefill')]['performance'].values[0]

# Calculate speedups
llama_512['speedup_vs_baseline'] = llama_512['performance'] / baseline_prefill

# Print results
print(llama_512[['hw_name', 'phase', 'performance', 'speedup_vs_baseline']])
```

### Visualization (Example)

```python
import matplotlib.pyplot as plt

# Stacked bar chart showing additive benefits
phases = ['Baseline', 'TPU', 'UniFlow', 'UniFlow+Partition']
prefill_speedups = [1.0, 1.6, 2.0, 2.5]
decode_speedups = [1.0, 1.4, 1.8, 2.8]

x = range(len(phases))
width = 0.35

plt.bar([i - width/2 for i in x], prefill_speedups, width, label='Prefill')
plt.bar([i + width/2 for i in x], decode_speedups, width, label='Decode')

plt.xlabel('Architecture')
plt.ylabel('Speedup vs Baseline')
plt.title('Additive Fusion Benefits (LLaMA-3-8B, seq_len=512)')
plt.xticks(x, phases, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('additive_benefits.png', dpi=300)
```

## Key Insights Expected

1. **Fusion is critical for compute-bound workloads (prefill)**
   - UniFlow shows significant speedup over TPU for long sequences
   - Benefit increases with sequence length

2. **PE Partition helps both phases**
   - Prefill: Overlaps Q@K^T with softmax
   - Decode: Overlaps KV-cache reads with computation

3. **KV-aware is critical for memory-bound workloads (decode)**
   - Decode phase is memory-bound (small GEMM, large cache reads)
   - Cache-aware dataflows reduce data movement significantly

4. **Benefits are mostly additive**
   - Each optimization contributes independently
   - Total speedup ≈ product of individual speedups

## Hardware Configurations

### Edge (2-level memory hierarchy)
- **PE Array**: 32×32 = 1024 PEs
- **L1 SRAM**: 4 MB
- **L2 DRAM**: 1.6 GB
- **L1 Bandwidth**: 500 GB/s
- **L2 Bandwidth**: 25 GB/s

### Cloud (3-level memory hierarchy)
- **PE Array**: 256×256 = 65536 PEs
- **L1 SRAM**: 20 MB
- **L2 SRAM**: 40 MB
- **L3 DRAM**: 1.6 GB
- **L1 Bandwidth**: 4000 GB/s
- **L2 Bandwidth**: 800 GB/s
- **L3 Bandwidth**: 160 GB/s

### PE Partition (default 75:25 ratio)
- **GEMM Block**: 75% of PEs for matrix multiplication
- **Non-GEMM Block**: 25% of PEs for softmax, layernorm, etc.
- **Shared L1**: Both blocks access same L1 buffer for data exchange

## Troubleshooting

### No valid configurations found
- **Cause**: Too few tuning trials
- **Solution**: Increase `--trials` to 100 or higher

### Memory resource violations
- **Cause**: Sequence length too large for buffer capacity
- **Solution**:
  - Use `--check_resource=False` to disable checking
  - Or use cloud configurations with larger buffers

### Performance is NaN or None
- **Cause**: Tuning failed to find valid configuration
- **Solution**: Check dataflow compatibility with accelerator

## Related Experiments

- `attention_accelerator_experiment.py`: Basic fusion comparison (no PE partition)
- `transformer_fusion_comparison.py`: Full transformer block fusion
- Individual dataflow files:
  - `attention_fusion_strategies.py`: Base fusion strategies
  - `attention_pe_partition.py`: PE partition dataflows
  - `attention_kv_cache.py`: KV-cache aware dataflows

## Citation

If you use this experimental framework in your research, please cite:

```bibtex
@article{uniflow2025,
  title={UniFlow: Unified PE Architecture for Transformer Acceleration},
  author={Your Name},
  journal={Conference/Journal},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on the FuseX GitHub repository.
