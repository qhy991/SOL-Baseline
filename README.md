# SOL-Baseline

SOTA GPU kernel library baselines for [sol-execbench](https://github.com/NVIDIA/sol-execbench) benchmark evaluation.

**All 44 baselines in this repo are verified to be faster than the torch reference implementation**, with measured speedups ranging from **1.04x to 7.21x**.

## What This Repo Provides

For each task in sol-execbench, we provide a `solution.json` that wraps a high-performance GPU kernel implementation. All baselines have been:

1. ✅ **Correctness verified**: Passes all workloads in the task
2. ✅ **Performance verified**: Faster than torch reference on average across workloads

The libraries used: **FlashInfer**, **FlashAttention**, **Liger**, **causal-conv1d**, and **PyTorch 2.x native** (F.rms_norm, F.scaled_dot_product_attention).

## Why Not All Tasks Have SOTA Baselines

Many sol-execbench reference implementations are **already optimal**:
- `torch.matmul` calls cuBLAS, which is SOTA for GEMM
- `F.conv2d` calls cuDNN, which is SOTA for convolution
- `F.gelu/silu` are already fused element-wise kernels
- Even Python for-loops can be fast when each iteration does substantial GPU work

We only add baselines when a SOTA library provides a measurable speedup over `torch`. See [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) for documented failure cases (e.g., SegmentGEMM was 33% slower than Python for-loop for sparse MoE).

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

## Results: 44 Baselines Sorted by Speedup

### Top Speedups (5x+): RMSNorm + Full Attention Blocks

| Task | Library | Speedup |
|---|---|---|
| L1_033 post_norm_residual | FlashInfer | **7.21x** |
| L1_069 rms_norm | FlashInfer | **6.62x** |
| L2_054 vision_encoder_layer | FlashAttn + FlashInfer | **6.08x** |
| L1_015 gqa_rope_qk_norm | FlashAttn + FlashInfer | **5.32x** |
| L2_018 cu_seqlens_vision_attention | FlashAttention varlen | **4.82x** |

### Solid Speedups (2-5x): Complete Decoder/Encoder Blocks

| Task | Library | Speedup |
|---|---|---|
| L1_092 gqa_attention_with_qk_norm | FlashAttn + FlashInfer | 3.47x |
| L2_059 decoder_layer_full_block | FlashAttn + FlashInfer | 3.27x |
| L2_041 kv_shared_dual_rope | FlashAttn + FlashInfer | 2.46x |
| L2_055 audio_encoder | FlashAttn + FlashInfer | 2.45x |
| L2_027 gqa_yarn_rope_qk_norm | torch native | 2.41x |
| L2_007 multimodal_rope_attention | FlashAttn | 2.28x |

### Moderate Speedups (1.5-2x)

| Task | Library | Speedup |
|---|---|---|
| L1_048 fused_gate_up_projection | Liger | 1.94x |
| L1_005 conv_gated_projection | causal-conv1d | 1.86x |
| L2_020 decoder_pre_post_norm | FlashAttn + FlashInfer | 1.81x |
| L2_034 vision_language_cross_attn | FlashAttn | 1.80x |
| L1_073 encoder_norm_kv_projection | FlashInfer | 1.72x |
| L1_018 fused_rope_qk_norm_kv_cache | FlashInfer | 1.63x |
| L1_080 adaptive_layernorm | torch native | 1.57x |
| L1_029 mamba_conv1d | causal-conv1d | 1.52x |
| **L1_021 vision_cu_seqlens_attention** | **torch SDPA adaptive** ⭐ | **1.48x** |

### Marginal Speedups (1.0-1.5x)

| Task | Library | Speedup |
|---|---|---|
| L2_062 decoder_complete_layer | FlashAttn + FlashInfer | 1.36x |
| L2_070 basic_transformer_block | torch native | 1.35x |
| L1_036 flux_output_norm_projection | torch native | 1.32x |
| L2_004 fused_residual_rms_mlp | FlashInfer | 1.27x |
| L2_039 kv_shared_attention | FlashAttn + FlashInfer | 1.18x |
| L1_064 latent_kv_expansion | FlashInfer | 1.14x |
| L1_043 mla_fused_qkv_rope_split | FlashInfer | 1.14x |
| L1_082 qk_norm_scaled_dot_product_attention | torch native | 1.13x |
| L1_054 audio_attention_qkv_norm | torch native | 1.11x |

### FlashInfer-Bench RMSNorm (separately benchmarked, see docs)

11 RMSNorm baselines with **7.0x - 14.3x** speedup, plus 6 GQA paged/ragged attention baselines verified for correctness.

## Library Selection Matrix

| Pattern | Recommended Library | When to Use |
|---|---|---|
| Pure RMSNorm (bf16, large hidden) | FlashInfer | hidden ≥ 1024, 6-14x speedup |
| Pre-norm + Attention block (bf16) | FlashInfer RMSNorm + FlashAttention | 3-7x combined speedup |
| Variable-length attention (bf16, long) | FlashAttention varlen | total_seq > 1500, 4-13x speedup |
| **Variable-length attention (bf16, short)** | **torch SDPA per-sequence loop** | total_seq < 1500 (FA bf16 inaccuracy) |
| float32 Attention | torch SDPA (cuDNN backend) | Only SDPA backend supporting float32 |
| float32 Norm + Attention | torch native (F.rms_norm + F.sdpa) | Unblocks float32 tasks |
| Custom RoPE (3D, YARN, mrope) | torch native (matches reference) + FA SDPA | Library RoPE conventions differ |
| Causal Conv1D (Mamba/Hyena) | causal-conv1d | 1.5-1.9x speedup |
| GEGLU/SwiGLU (after linear projection) | Liger GELUMul | 1.9x speedup, but **not** for pure activation |
| Sparse MoE (per-expert routing) | **No SOTA library beats torch** | See NEGATIVE_RESULTS — for-loop wins |
| Pure matmul/conv2d/gelu/silu | **torch reference (no SOTA needed)** | Already calls cuBLAS/cuDNN |
| Backward passes | **No SOTA library available** | Forward-only libraries |

## Adaptive Baselines (Per-Workload Dispatch)

For tasks where the optimal library depends on workload size, we provide adaptive baselines that switch implementations based on shape. See [`docs/ADAPTIVE_BASELINES.md`](docs/ADAPTIVE_BASELINES.md).

Example: **L1_021** uses block-diagonal mask SDPA for short sequences (1.8-2.1x) and per-sequence loop for long sequences (avoiding O(N²) mask memory). Average: 1.48x.

## Directory Structure

```
.
├── baselines/
│   ├── flashinfer/              # 15 baselines (RMSNorm + composition)
│   ├── liger/                   # 1 baseline (GEGLU after linear)
│   ├── causal_conv1d/           # 2 baselines (Mamba/Hyena conv)
│   ├── flash_attn/              # 19 baselines (attention blocks, varlen, paged/ragged)
│   └── torch/                   # 7 baselines (PyTorch 2.x F.rms_norm + F.sdpa)
├── docs/
│   ├── INSTALL.md               # Library installation guide
│   ├── BASELINE_DESIGN.md       # Baseline design notes
│   ├── COVERAGE_ANALYSIS.md     # Per-task coverage analysis
│   ├── ADDITIONAL_LIBRARIES.md  # Research on more SOTA libraries
│   ├── COMPOSITION_METHODOLOGY.md # Methodology for composing SOTA libs
│   ├── ADAPTIVE_BASELINES.md    # ⭐ Workload-aware library selection
│   └── NEGATIVE_RESULTS.md      # Cases where SOTA libs don't beat reference
├── scripts/
│   ├── benchmark.py             # Benchmark SOTA vs torch reference
│   └── verify.py                # Verify all baseline solutions
└── README.md
```

## Methodology

See [`docs/COMPOSITION_METHODOLOGY.md`](docs/COMPOSITION_METHODOLOGY.md) for the key insight: complex fused kernels can be matched by composing multiple SOTA libraries (e.g., `FlashInfer.rmsnorm` + `flash_attn_func` + `torch.linear`).

See [`docs/ADAPTIVE_BASELINES.md`](docs/ADAPTIVE_BASELINES.md) for the workload-aware dispatch pattern that maximizes average speedup.

See [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) for cases where SOTA libraries failed to beat torch reference (often because the reference itself already calls cuBLAS/cuDNN).

## License

MIT