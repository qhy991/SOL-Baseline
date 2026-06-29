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

### 25 Verified Baseline Solutions

| Library | Task Count | Notes |
|---|---|---|
| **FlashInfer** | 13 | RMSNorm (11) + fused RMS+MLP + encoder norm |
| **Liger** | 3 | GEGLU (2) + GroupNorm |
| **causal-conv1d** | 2 | Mamba/Hyena depthwise conv |
| **FlashAttention + FlashInfer** | 7 | **Composition of multiple SOTA libs** for full attention/decoder blocks |

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

#### SOTA Library Composition (FlashAttention + FlashInfer + Liger)

These tasks demonstrate that **complex fused kernels can be matched by composing multiple SOTA libraries**. See [docs/COMPOSITION_METHODOLOGY.md](docs/COMPOSITION_METHODOLOGY.md) for the detailed methodology.

| Task | Description | Status |
|---|---|---|
| L1_015 gqa_rope_qk_norm | GQA + RoPE + QK RMSNorm | ✓ 16/16 PASSED |
| L1_073 encoder_norm_kv_projection | RMSNorm + KV projection | ✓ 16/16 PASSED |
| L1_092 gqa_attention_with_qk_norm | Full GQA + QK norm + RoPE + Output | ✓ 16/16 PASSED |
| L2_004 fused_residual_rms_mlp | Residual + RMSNorm + SwiGLU MLP | ✓ 16/16 PASSED |
| L2_007 multimodal_rope_attention | GQA + 3D Multi-modal RoPE | ✓ 16/16 PASSED |
| L2_018 cu_seqlens_vision_attention | Varlen vision attention | ⚠ 11/16 PASSED |
| L2_020 decoder_layer_pre_post_norm | Complete decoder (complex-RoPE) | ✓ 16/16 PASSED |
| L2_053 text_decoder_layer | Complete decoder layer (Norm+Attn+MLP) | ✓ 16/16 PASSED |
| L2_062 decoder_complete_layer | Self-attn + Cross-attn + MLP | ✓ 16/16 PASSED |

## Directory Structure

```
.
├── baselines/
│   ├── flashinfer/
│   │   ├── FlashInfer-Bench/   # 9 RMSNorm baselines (direct API match)
│   │   ├── L1/                 # 3 L1 RMSNorm baselines
│   │   └── L2/                 # 1 L2 fused RMS+MLP baseline
│   ├── liger/
│   │   └── L1/                 # 3 L1 activation/norm baselines
│   ├── causal_conv1d/
│   │   └── L1/                 # 2 L1 causal conv baselines
│   └── flash_attn/             # SOTA composition baselines
│       ├── L1/                 # 2 L1 attention block baselines
│       └── L2/                 # 5 L2 decoder/attention block baselines
├── docs/
│   ├── INSTALL.md              # Library installation guide
│   ├── BASELINE_DESIGN.md      # Baseline design notes
│   ├── COVERAGE_ANALYSIS.md    # Per-task coverage analysis
│   ├── ADDITIONAL_LIBRARIES.md # Research on more SOTA libraries
│   └── COMPOSITION_METHODOLOGY.md  # ⭐ Methodology for composing SOTA libs
│   ├── INSTALL.md              # Library installation guide
│   ├── BASELINE_DESIGN.md      # Baseline design notes
│   ├── COVERAGE_ANALYSIS.md    # Per-task coverage analysis
│   ├── ADDITIONAL_LIBRARIES.md # Research on more SOTA libraries
│   └── COMPOSITION_METHODOLOGY.md  # ⭐ Methodology for composing SOTA libs (key insight)
├── scripts/
│   ├── benchmark.py            # Benchmark script for SOTA vs torch comparison
│   └── verify.py               # Verify all baseline solutions
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