# SOL-Baseline

SOTA GPU kernel library baselines for [sol-execbench](https://github.com/NVIDIA/sol-execbench) benchmark evaluation.

This repository provides pre-built `solution.json` files that wrap high-performance GPU kernel libraries (FlashInfer, Liger, causal-conv1d) as evaluation baselines. Each baseline has been verified to pass correctness validation across all 16 workloads per task.

## Quick Start

```bash
# 1. Install sol-execbench (see https://github.com/NVIDIA/sol-execbench)
git clone https://github.com/NVIDIA/sol-execbench.git
cd sol-execbench
uv sync --all-groups

# 2. Install baseline libraries (see docs/INSTALL.md)
./scripts/install_baseline_libs.sh

# 3. Clone this repository
git clone https://github.com/qhy991/SOL-Baseline.git baselines

# 4. Run evaluation
uv run sol-execbench data/benchmark/FlashInfer-Bench/021_rmsnorm_h128 \
    --solution baselines/baselines/flashinfer/FlashInfer-Bench/FlashInfer-Bench_021_rmsnorm_h128/solution.json \
    --json
```

## Results Summary

### 16 Verified Baseline Solutions

| Library | Task Count | Speedup Range |
|---|---|---|
| **FlashInfer** | 11 | 6.7x - 14.3x |
| **Liger** | 3 | 2.2x (1 task), slower (2 tasks) |
| **causal-conv1d** | 2 | 1.5x - 1.6x |

### Performance (SOTA vs Torch Reference)

#### FlashInfer-Bench RMSNorm
| Task | hidden_size | SOTA (ms) | Ref (ms) | Speedup |
|---|---|---|---|---|
| 001 fused_add_rmsnorm | 2048 | 0.027 | 0.196 | **7.3x** |
| 002 fused_add_rmsnorm | 4096 | 0.039 | 0.330 | **8.5x** |
| 003 fused_add_rmsnorm | 7168 | 0.049 | 0.426 | **8.7x** |
| 021 rmsnorm | 128 | 0.017 | 0.163 | **9.3x** |
| 022 rmsnorm | 512 | 0.011 | 0.073 | **7.0x** |
| 023 rmsnorm | 1536 | 0.011 | 0.112 | **10.1x** |
| 024 rmsnorm | 2048 | 0.012 | 0.146 | **12.1x** |
| 025 rmsnorm | 4096 | 0.018 | 0.243 | **13.9x** |
| 026 rmsnorm | 7168 | 0.022 | 0.312 | **14.3x** |

#### L1 Tasks
| Task | Library | SOTA (ms) | Ref (ms) | Speedup |
|---|---|---|---|---|
| L1_033 post_norm_residual | FlashInfer | 0.041 | 0.298 | **7.2x** |
| L1_069 rms_norm | FlashInfer | 0.040 | 0.267 | **6.7x** |
| L1_048 fused_gate_up_projection | Liger | 0.764 | 1.707 | **2.2x** |
| L1_005 conv_gated_projection | causal-conv1d | 0.158 | 0.256 | **1.6x** |
| L1_029 mamba_conv1d | causal-conv1d | 1.560 | 2.335 | **1.5x** |
| L1_085 geglu_activation | Liger | 0.116 | 0.084 | 0.7x |
| L1_078 group_norm_fusion | Liger | 0.342 | 0.095 | 0.3x |

## Directory Structure

```
.
├── baselines/
│   ├── flashinfer/
│   │   ├── FlashInfer-Bench/   # 9 FlashInfer-Bench RMSNorm baselines
│   │   └── L1/                 # 2 L1 RMSNorm baselines
│   ├── liger/
│   │   └── L1/                 # 3 L1 activation/norm baselines
│   └── causal_conv1d/
│       └── L1/                 # 2 L1 causal conv baselines
├── docs/
│   └── INSTALL.md              # Library installation guide
├── scripts/
│   └── benchmark.py            # Benchmark script for SOTA vs torch comparison
└── README.md
```

## Libraries Used

| Library | Version | Purpose | Baselines |
|---|---|---|---|
| **FlashInfer** | 0.6.12 | RMSNorm, Attention, Activation | 11 baselines |
| **Liger Kernel** | 0.8.0 | GEGLU, SwiGLU, GroupNorm | 3 baselines |
| **causal-conv1d** | 1.6.2 | Causal depthwise convolution | 2 baselines |

## Benchmark

Run the benchmark script to compare SOTA baselines against torch reference:

```bash
cd sol-execbench
python baselines/scripts/benchmark.py
```

The script runs each baseline and its corresponding torch reference implementation across all workloads, measuring latency and computing speedup factors.

## License

MIT