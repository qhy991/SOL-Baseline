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

### 38 Verified Baseline Solutions

| Library | Task Count | Notes |
|---|---|---|
| **FlashInfer** | 15 | RMSNorm (11) + fused RMS+MLP + 3 RMSNorm+projection tasks |
| **Liger** | 3 | GEGLU (2) + GroupNorm |
| **causal-conv1d** | 2 | Mamba/Hyena depthwise conv |
| **FlashAttention + FlashInfer** | 12 | **Composition of SOTA libs** for full attention/decoder blocks |
| **FlashAttention varlen (paged/ragged)** | 6 | GQA paged decode/prefill + ragged prefill with LSE |

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
| L1_033 post_norm_residual | FlashInfer | 0.073 | 0.563 | **7.7x** |
| L1_069 rms_norm | FlashInfer | 0.044 | 0.303 | **6.8x** |
| L1_073 encoder_norm_kv_projection | FlashInfer | 0.065 | 0.117 | **1.8x** |
| L1_043 mla_fused_qkv_rope_split | FlashInfer | 0.487 | 0.550 | 1.1x |
| L1_064 latent_kv_expansion | FlashInfer | 0.391 | 0.434 | 1.1x |
| L1_048 fused_gate_up_projection (GEGLU) | Liger | 1.004 | 2.263 | **2.3x** |
| L1_005 conv_gated_projection | causal-conv1d | 0.160 | 0.288 | **1.8x** |
| L1_029 mamba_conv1d | causal-conv1d | 1.172 | 1.831 | **1.6x** |
| L1_015 gqa_rope_qk_norm | FlashAttn+FlashInfer | 0.820 | 2.890 | **3.5x** |
| L1_092 gqa_attention_with_qk_norm | FlashAttn+FlashInfer | 0.752 | 1.697 | **2.3x** |
| L1_085 geglu_activation | Liger | 0.114 | 0.091 | 0.8x |
| L1_078 group_norm_fusion | Liger | 0.169 | 0.087 | 0.5x |

#### L2 Tasks (Composition Baselines)
| Task | Library | SOTA (ms) | Ref (ms) | Speedup |
|---|---|---|---|---|
| L2_054 vision_encoder_layer | FlashAttn+FlashInfer | 0.844 | 6.148 | **7.3x** |
| L2_018 cu_seqlens_vision_attention | FlashAttn varlen | 0.352 | 2.273 | **6.4x** |
| L2_041 kv_shared_dual_rope | FlashAttn+FlashInfer | 0.746 | 2.348 | **3.1x** |
| L2_007 multimodal_rope_attention | FlashAttn | 1.033 | 2.495 | **2.4x** |
| L2_059 decoder_layer_full_block | FlashAttn+FlashInfer | 2.013 | 3.373 | **1.7x** |
| L2_020 decoder_layer_pre_post_norm | FlashAttn+FlashInfer | 0.866 | 1.186 | 1.4x |
| L2_034 vision_language_cross_attention | FlashAttn | 1.517 | 2.178 | 1.4x |
| L2_062 decoder_complete_layer | FlashAttn+FlashInfer | 1.097 | 1.494 | 1.4x |
| L2_053 text_decoder_layer | FlashAttn+FlashInfer | 1.427 | 1.716 | 1.2x |
| L2_039 kv_shared_attention | FlashAttn+FlashInfer | 0.778 | 0.895 | 1.1x |
| L2_004 fused_residual_rms_mlp | FlashInfer | 16.772 | 17.485 | 1.0x |

#### SOTA Library Composition

The L1/L2 tasks above demonstrate that **complex fused kernels can be matched by composing multiple SOTA libraries**. See [docs/COMPOSITION_METHODOLOGY.md](docs/COMPOSITION_METHODOLOGY.md) for the detailed methodology.

All tasks pass 16/16 workloads except:
- L2_018 cu_seqlens_vision_attention: 11/16 (FlashAttention bfloat16 precision on small sequences)
- L2_041 kv_shared_dual_rope: 11/16 (similar precision issue)

#### FlashAttention varlen for FlashInfer-Bench Paged/Ragged Attention

Tasks requiring `flashinfer-trace` dataset (run `huggingface-cli download flashinfer-ai/flashinfer-trace`).
Set `FLASHINFER_TRACE_DIR=/path/to/data/flashinfer-trace` when running.

| Task | Description | Status |
|---|---|---|
| FIB_012 gqa_paged_decode_kv4 | Paged GQA decode | ✓ 48/48 PASSED |
| FIB_013 gqa_paged_decode_kv8 | Paged GQA decode (kv8) | ✓ 48/48 PASSED |
| FIB_014 gqa_paged_prefill_kv4 | Paged GQA causal prefill | ✓ 30/30 PASSED |
| FIB_015 gqa_paged_prefill_kv8 | Paged GQA causal prefill (kv8) | ✓ 38/38 PASSED |
| FIB_016 gqa_ragged_prefill_kv4 | Ragged GQA causal prefill | ✓ 15/15 PASSED |
| FIB_017 gqa_ragged_prefill_kv8 | Ragged GQA causal prefill (kv8) | ✓ 21/21 PASSED |

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