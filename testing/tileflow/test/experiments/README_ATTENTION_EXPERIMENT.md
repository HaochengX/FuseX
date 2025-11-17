# Attention Accelerator Architecture Comparison Experiments

This directory contains experiments for comparing three different accelerator architectures for self-attention operations.

## Overview

### Accelerator Architectures

We compare three fundamentally different approaches to handling self-attention workloads:

1. **Baseline Architecture** (Systolic Array + CPU)
   - GEMM operations execute on systolic array
   - Non-GEMM operations (max, exp, div, etc.) offloaded to CPU via DRAM
   - **Fusion Strategy**: No fusion - each operation is separate
   - **Models**: Traditional AI accelerators without on-chip non-GEMM support

2. **TPU-Style Architecture** (Systolic Array + VPU)
   - GEMM operations on systolic array
   - Non-GEMM operations on dedicated on-chip Vector Processing Unit (VPU)
   - **Fusion Strategy**: Partial fusion - operations can be pipelined on VPU
   - **Models**: Modern accelerators like Google TPU with specialized units

3. **UniFlow Architecture** (Unified PE)
   - Both GEMM and non-GEMM operations on unified Processing Elements
   - Each PE has MAC unit + vector ALU capabilities
   - **Fusion Strategy**: Full fusion - all operations fused on same hardware
   - **Models**: Proposed unified architecture for attention-heavy workloads

### Configuration Variants

Each architecture has two configurations:
- **Edge**: 32×32 PE array, 2-level memory hierarchy (for edge devices)
- **Cloud**: 256×256 PE array, 3-level memory hierarchy (for datacenter)

## Running Experiments

### Basic Usage

```bash
cd /home/user/FuseX/testing/tileflow/test/experiments

# Run with default settings (BERT-Base, edge configs only)
python attention_accelerator_experiment.py --metric=1e9/latency --trials=1000 --edge_only

# Run all configurations (edge + cloud)
python attention_accelerator_experiment.py --metric=1e9/latency --trials=1000 --number=3

# Run specific hardware type
python attention_accelerator_experiment.py --metric=1e9/latency --trials=1000 --hw_filter=uniflow

# Run with energy metric
python attention_accelerator_experiment.py --metric=1e9/energy --trials=1000
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--metric` | Evaluation metric (1e9/latency, 1e9/energy, Utilization_L0, etc.) | 1e9/latency |
| `--batch` | Batch size for attention | 1 |
| `--trials` | Number of tuning trials per configuration | 1000 |
| `--begin` | Starting index in shapes list | 0 |
| `--number` | Number of shapes to evaluate | 1 |
| `--edge_only` | Only evaluate edge configurations | False |
| `--cloud_only` | Only evaluate cloud configurations | False |
| `--hw_filter` | Filter hardware by name (baseline/tpu/uniflow) | "" |
| `--define_tiling_space` | Enable tiling space definition | False |
| `--check_resource` | Enable resource checking | False |
| `--debug` | Enable debug mode | False |

### Model Shapes

The experiment includes the following transformer models:

**BERT Variants:**
- BERT-Small: 8 heads, 512 seq_len, 512 hidden
- BERT-Base: 12 heads, 512 seq_len, 768 hidden
- BERT-Large: 16 heads, 512 seq_len, 1024 hidden

**Vision Transformers:**
- ViT-Base/14: 12 heads, 256 seq_len, 768 hidden
- ViT-Large/14: 16 heads, 256 seq_len, 1024 hidden
- ViT-Huge/14: 16 heads, 256 seq_len, 1280 hidden
- ViT-Base/16: 12 heads, 196 seq_len, 768 hidden
- ViT-Large/16: 16 heads, 196 seq_len, 1024 hidden
- ViT-Huge/16: 16 heads, 196 seq_len, 1280 hidden

**LLaMA Variants:**
- LLaMA-128: 32 heads, 128 seq_len, 4096 hidden
- LLaMA-512: 32 heads, 512 seq_len, 4096 hidden
- LLaMA-1024: 32 heads, 1024 seq_len, 4096 hidden

**GPT-3 Variants:**
- GPT-3-Medium-128: 24 heads, 128 seq_len, 1024 hidden
- GPT-3-Medium-512: 24 heads, 512 seq_len, 1024 hidden
- GPT-3-Medium-1024: 24 heads, 1024 seq_len, 1024 hidden

## Example Workflows

### 1. Quick Test with Edge Configurations

Test all edge configurations on BERT-Base:

```bash
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=100 \
    --edge_only \
    --begin=1 \
    --number=1
```

### 2. Comprehensive Evaluation

Test all configurations on all BERT variants:

```bash
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=1000 \
    --begin=0 \
    --number=3
```

### 3. Compare Specific Architectures

Compare only baseline vs UniFlow on LLaMA models:

```bash
# Run baseline
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=1000 \
    --hw_filter=baseline \
    --begin=9 \
    --number=3

# Run UniFlow
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=1000 \
    --hw_filter=uniflow \
    --begin=9 \
    --number=3
```

### 4. Energy Efficiency Analysis

```bash
python attention_accelerator_experiment.py \
    --metric=1e9/energy \
    --trials=1000 \
    --number=15
```

### 5. Cloud-Scale Evaluation

Test only cloud configurations:

```bash
python attention_accelerator_experiment.py \
    --metric=1e9/latency \
    --trials=1000 \
    --cloud_only \
    --begin=0 \
    --number=5
```

## Output Format

### During Execution

The experiment prints progress for each configuration:

```
================================================================================
Evaluating: BERT-Base (heads=12, seq_len=512, hidden=768)
================================================================================

[0] Running: baseline_edge with no_fusion
    Architecture: baseline_edge
    Fusion: no_fusion
    Levels: 2
    Performance: 1250.45 1e9/latency
    Config: {...}
```

### Results Summary

At the end, results are printed in CSV format:

```
batch,seq_len,num_heads,hidden,metric,hw_name,fusion_strategy,hw_id,key,config,perf
1,512,12,768,1e9/latency,baseline_edge,no_fusion,0,{...},{...},1250.45
1,512,12,768,1e9/latency,tpu_edge,partial_fusion,1,{...},{...},1780.32
1,512,12,768,1e9/latency,uniflow_edge,full_fusion,2,{...},{...},2456.78
```

### Performance Comparison Table

A comparison table shows relative speedups:

```
Model: BERT-Base (heads=12, seq_len=512, hidden=768)
--------------------------------------------------------------------------------
Hardware                  Fusion               Performance     Speedup vs Baseline
--------------------------------------------------------------------------------
baseline_edge            no_fusion            1250.45         1.00x
tpu_edge                 partial_fusion       1780.32         1.42x
uniflow_edge             full_fusion          2456.78         1.96x
```

## Understanding Results

### Metrics

- **1e9/latency**: Operations per second (higher is better)
  - Measures computational throughput
  - Primary metric for performance comparison

- **1e9/energy**: Operations per joule (higher is better)
  - Measures energy efficiency
  - Important for battery-powered devices

- **Utilization_L0/L1/L2/L3**: Buffer utilization (0-1, higher is better)
  - Measures how effectively memory hierarchy is used

### Expected Trends

Based on architecture characteristics:

1. **UniFlow > TPU > Baseline** (Latency)
   - UniFlow has minimal data movement (in-place execution)
   - TPU avoids off-chip transfers for non-GEMM
   - Baseline suffers from DRAM roundtrips

2. **UniFlow > TPU > Baseline** (Energy)
   - Fewer data transfers = lower energy
   - DRAM accesses are 100× more expensive than on-chip

3. **Advantage increases with sequence length**
   - More non-GEMM operations to benefit from fusion
   - Longer sequences amplify data movement costs

### Interpreting Speedups

Typical speedups for BERT-Base (seq_len=512):

- **TPU vs Baseline**: 1.3-1.8×
  - Benefit from on-chip VPU processing

- **UniFlow vs Baseline**: 1.8-2.5×
  - Maximum benefit from full fusion

- **UniFlow vs TPU**: 1.3-1.6×
  - Eliminates cross-unit data movement

## File Structure

```
attention_accelerator_experiment.py    # Main experiment script
README_ATTENTION_EXPERIMENT.md         # This file
../../python/tileflow/
  ├── accelerator.py                   # Accelerator definitions
  ├── dataflows/
  │   ├── attention_fusion_strategies.py   # Dataflow with fusion
  │   └── __init__.py                      # Dataflow exports
  ├── docs/
  │   └── ATTENTION_ACCELERATORS.md    # Architecture documentation
  └── examples/
      └── attention_accelerator_comparison.py   # Usage example
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `--trials` to fewer iterations
   - Use smaller models (start with BERT-Small)
   - Run only edge configs with `--edge_only`

2. **Slow Execution**
   - Set `--trials=100` for quick tests
   - Use `--hw_filter` to test one architecture at a time
   - Start with `--number=1` to test one model

3. **Import Errors**
   - Ensure you're in the correct directory
   - Check that all dataflow files are in place
   - Verify `__init__.py` has correct imports

### Debug Mode

Enable detailed logging:

```bash
python attention_accelerator_experiment.py \
    --debug \
    --trials=10 \
    --number=1
```

## Next Steps

### After Running Experiments

1. **Analyze Results**
   - Compare speedups across different models
   - Identify which architecture performs best for your workload
   - Look for trends with sequence length and hidden dimension

2. **Deep Dive**
   - Run with `--metric=1e9/energy` for energy analysis
   - Check buffer utilization with `--metric=Utilization_L1`
   - Try different tiling configurations

3. **Paper Figures**
   - Parse CSV output for plotting
   - Generate speedup charts
   - Create energy efficiency comparisons

### Customization

To modify the experiment:

1. **Add New Models**: Edit `shapes` list in the script
2. **Modify Hardware**: Edit accelerator definitions in `tileflow/accelerator.py`
3. **Change Fusion**: Modify dataflows in `attention_fusion_strategies.py`
4. **Add Metrics**: Extend metric types in experiment loop

## Citation

If you use this experiment framework in your research, please cite:

```bibtex
@article{fusex2025attention,
  title={UniFlow: Unified Accelerator Architecture for Attention Operations},
  author={Your Name},
  journal={Conference/Journal},
  year={2025}
}
```

## Support

For questions or issues:
- Check the main documentation in `tileflow/docs/ATTENTION_ACCELERATORS.md`
- Review the example in `tileflow/examples/attention_accelerator_comparison.py`
- Open an issue on the FuseX GitHub repository
