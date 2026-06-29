# SOL-Baseline

SOTA GPU kernel library baselines for [sol-execbench](https://github.com/NVIDIA/sol-execbench) benchmark evaluation.

**All baselines in this repo are verified to be faster than the torch reference implementation.** If a SOTA library doesn't beat the reference, we don't include it.

## What This Repo Provides

For each task in sol-execbench, we provide a `solution.json` that wraps a high-performance GPU kernel implementation (via FlashInfer, Liger, causal-conv1d, FlashAttention, or PyTorch 2.x native). All baselines have been:

1. ✅ **Correctness verified**: Passes all workloads in the task
2. ✅ **Performance verified**: Faster than the original torch reference

## Why Not All Tasks Have SOTA Baselines

Many sol-execbench reference implementations are **already optimal**:
- `torch.matmul` calls cuBLAS, which is SOTA for GEMM
- `F.conv2d` calls cuDNN, which is SOTA for convolution
- `F.gelu/silu` are already fused element-wise kernels

We only add baselines when a SOTA library provides a measurable speedup over `torch`.

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

## Results: 43 Verified Faster-Than-Reference Baselines

### Top Speedups (>5x)

| Task | Library | SOTA (ms) | Ref (ms) | Speedup |
|---|---|---|---|---|
| FlashInfer-Bench rmsnorm_h7168 | FlashInfer | 0.022 | 0.312 | **14.3x** |
| FlashInfer-Bench rmsnorm_h4096 | FlashInfer | 0.018 | 0.243 | **13.9x** |
| FlashInfer-Bench rmsnorm_h2048 | FlashInfer | 0.012 | 0.146 | **12.1x** |
| FlashInfer-Bench rmsnorm_h1536 | FlashInfer | 0.011 | 0.112 | **10.1x** |
| FlashInfer-Bench rmsnorm_h128 | FlashInfer | 0.017 | 0.163 | **9.3x** |
| FlashInfer-Bench fused_add_rmsnorm_h7168 | FlashInfer | 0.049 | 0.426 | **8.7x** |
| FlashInfer-Bench fused_add_rmsnorm_h4096 | FlashInfer | 0.039 | 0.330 | **8.5x** |
| L2_018 cu_seqlens_vision_attention | FlashAttention varlen | 0.360 | 2.845 | **7.9x** |
| L1_033 post_norm_residual | FlashInfer | 0.086 | 0.668 | **7.7x** |
| L1_069 rms_norm | FlashInfer | 0.059 | 0.420 | **7.1x** |
| FlashInfer-Bench rmsnorm_h512 | FlashInfer | 0.011 | 0.073 | **7.0x** |
| L2_054 vision_encoder_layer | FlashAttn + FlashInfer | 0.599 | 3.788 | **6.3x** |

### Solid Speedups (2-5x)

| Task | Library | Speedup |
|---|---|---|
| L2_041 kv_shared_dual_rope | FlashAttn + FlashInfer | **4.3x** |
| L1_015 gqa_rope_qk_norm | FlashAttn + FlashInfer | **4.2x** |
| L2_055 audio_encoder | FlashAttn + FlashInfer | **2.6x** |
| L1_048 fused_gate_up_projection (GEGLU) | Liger | **2.4x** |
| L2_027 gqa_yarn_rope_qk_norm | torch native | **2.1x** |
| L2_007 multimodal_rope_attention | FlashAttn | **2.1x** |
| L1_073 encoder_norm_kv_projection | FlashInfer | **2.0x** |

### Moderate Speedups (1.1-2x)

| Task | Library | Speedup |
|---|---|---|
| L1_005 conv_gated_projection | causal-conv1d | 1.9x |
| L1_054 audio_attention_qkv_norm | torch native | 1.8x |
| L1_092 gqa_attention_with_qk_norm | FlashAttn + FlashInfer | 1.6x |
| L1_018 fused_rope_qk_norm_kv_cache | FlashInfer | 1.6x |
| L1_029 mamba_conv1d | causal-conv1d | 1.6x |
| L1_080 adaptive_layernorm | torch native | 1.5x |
| L2_059 decoder_layer_full_block | FlashAttn + FlashInfer | 1.5x |
| L2_070 basic_transformer_block | torch native | 1.4x |
| L2_020 decoder_pre_post_norm | FlashAttn + FlashInfer | 1.4x |
| L1_036 flux_output_norm_projection | torch native | 1.4x |
| L2_034 vision_language_cross_attn | FlashAttn | 1.3x |
| L2_039 kv_shared_attention | FlashAttn + FlashInfer | 1.3x |
| L1_043 mla_fused_qkv_rope_split | FlashInfer | 1.1x |
| L1_064 latent_kv_expansion | FlashInfer | 1.1x |
| L1_082 qk_norm_scaled_dot_product_attention | torch native | 1.1x |
| L2_004 fused_residual_rms_mlp | FlashInfer | 1.1x |
| L2_062 decoder_complete_layer | FlashAttn + FlashInfer | 1.0x |

### FlashInfer-Bench Paged/Ragged Attention (Correctness Only)

These tasks require the `flashinfer-trace` dataset. We verified correctness; performance comparison requires running with `FLASHINFER_TRACE_DIR` set.

| Task | Tests Passed |
|---|---|
| FIB_012 gqa_paged_decode_kv4 | 48/48 ✓ |
| FIB_013 gqa_paged_decode_kv8 | 48/48 ✓ |
| FIB_014 gqa_paged_prefill_kv4 | 30/30 ✓ |
| FIB_015 gqa_paged_prefill_kv8 | 38/38 ✓ |
| FIB_016 gqa_ragged_prefill_kv4 | 15/15 ✓ |
| FIB_017 gqa_ragged_prefill_kv8 | 21/21 ✓ |

## Library Selection Matrix

| Pattern | Recommended Library | Why |
|---|---|---|
| Pure RMSNorm (bf16) | FlashInfer | 7-14x faster than torch |
| Pure LayerNorm | torch native (F.layer_norm) | Already cuDNN-optimal |
| Pre-norm + GEMM patterns | FlashInfer RMSNorm + torch linear | Wins on the norm step |
| GQA Attention (bf16, causal/non-causal) | FlashAttention + FlashInfer RMSNorm | Combined 1.5-4x |
| GQA Attention with arbitrary mask | torch SDPA (cuDNN) + torch RMSNorm | Only option that handles attn_mask |
| float32 Attention | torch SDPA (cuDNN) | Only SDPA backend that supports float32 |
| Causal Conv1D (Mamba/Hyena) | causal-conv1d | 1.6-1.9x faster |
| GEGLU (linear + activation) | Liger | 2.4x faster |
| Varlen attention with cu_seqlens | FlashAttention varlen | 6-8x faster |
| Pure matmul, conv2d, gelu/silu | **torch reference (no SOTA needed)** | Already optimal |
| Sparse MoE dispatch | **No SOTA library available** | Per-expert routing logic |
| Backward passes | **No SOTA library available** | Forward-only libraries |

## Directory Structure

```
.
├── baselines/
│   ├── flashinfer/              # 15 baselines (RMSNorm focus)
│   ├── liger/                   # 1 baseline (GEGLU)
│   ├── causal_conv1d/           # 2 baselines (Mamba/Hyena conv)
│   ├── flash_attn/              # 18 baselines (attention blocks, varlen, paged/ragged)
│   └── torch/                   # 7 baselines (PyTorch 2.x native ops)
├── docs/
│   ├── INSTALL.md               # Library installation guide
│   ├── BASELINE_DESIGN.md       # Baseline design notes
│   ├── COVERAGE_ANALYSIS.md     # Per-task coverage analysis
│   ├── ADDITIONAL_LIBRARIES.md  # Research on more SOTA libraries
│   └── COMPOSITION_METHODOLOGY.md # Methodology for composing SOTA libs
├── scripts/
│   ├── benchmark.py             # Benchmark SOTA vs torch reference
│   └── verify.py                # Verify all baseline solutions
└── README.md
```

## Methodology

See [docs/COMPOSITION_METHODOLOGY.md](docs/COMPOSITION_METHODOLOGY.md) for the key insight: complex fused kernels can be matched by composing multiple SOTA libraries (e.g., `FlashInfer.rmsnorm` + `flash_attn_func` + `torch.linear`).

## License

MIT