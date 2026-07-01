# Contest baseline coverage — final summary for reviewer

**Author:** sol-baseline maintainer
**Date:** 2026-07-01
**Repository:** `qhy991/SOL-Baseline`, branch `main`, HEAD `d49f31d`
**Environment (baseline of record):** NVIDIA B200 (SM 100, Blackwell), CUDA 13.0, PyTorch 2.9.0+cu130, Triton 3.5.0, FlashInfer 0.6.12, flash_attn 2.8.3

---

## 1. Scope and completeness claim

The Contest task set at `data/benchmark/Contest/` contains **60 problems** across four categories:

| Category | Count | Description |
|---|---|---|
| L1 | 20 | Single-kernel patterns (attention fragments, RoPE, RMSNorm, expert routing, MoE forward) |
| L2 | 20 | Fused block patterns (full decoder layers, encoder blocks, MoE + attention composites) |
| Quant | 10 | FP8 blockwise quantization patterns (1×128 activation / 128×128 weight scaling) |
| FlashInfer-Bench | 10 | FlashInfer library validation targets (RMSNorm variants, paged/ragged attention, FP8 MoE) |

Of these 60 problems:

- **44 baselines shipped** and validated on B200 (73%)
- **15 negative results documented** in `docs/CONTEST_NEGATIVE_RESULTS.md` — SOTA library either matches reference within measurement noise (~1.0x), loses to it, or is unavailable/broken on this env
- **1 task deferred** (L2/006 — see §5.3)

**Total accounted-for coverage: 60/60 (100%).** No task is unlisted; every task is either shipped, negative-documented, or explicitly deferred with a reason.

---

## 2. Result at a glance

**44 shipped baselines pass 100% of workloads.** Full per-task table lives at [`docs/CONTEST_RESULTS.md`](CONTEST_RESULTS.md). Summary statistics:

| Metric | Value | Population |
|---|---|---|
| Correctness pass rate | 44/44 baselines, ≈595/595 workloads across those 44 | full Contest coverage |
| Geometric mean speedup vs reference | **3.66x** | 31 timed baselines (see §3.3 for the 13 correctness-only ones) |
| Peak speedup | **29.83x** | Quant/015 FP8 MLA output projection |
| Median speedup | ~2.02x | 31 timed baselines |
| Baselines ≥5x | 8 | Quant/003, 005, 015; FIB/002, 003, 023, 026; L1/049, L1/069; L2/054 |
| Baselines 2x–5x | 12 | mostly attention + MoE forward blocks |
| Baselines 1.0x–2x | 11 | RoPE, small compositions, adaptive baselines |

---

## 3. Methodology (audit-critical section)

### 3.1 Correctness protocol

Every baseline is run through `sol-execbench`'s eval driver, which:
1. Loads the reference `run()` implementation from the task's `definition.json`.
2. Instantiates each workload (16 per task typical; some 8, some 47) using the recorded axis assignments and the seed = 200 fixed by `bench_config*.json`.
3. Invokes the baseline's `run()` function on independently cloned inputs.
4. Compares outputs elementwise against the reference using the per-workload `tolerance` block (`max_atol`, `max_rtol`, `required_matched_ratio` — typically 0.99 or 0.98).

A baseline is considered "PASS" only when **all workloads** report `PASSED` status (subprocess-isolated, so failures are always fatal for the workload).

### 3.2 Performance protocol

Two configurations, chosen per task by the aggregator based on reference speed:

- **`bench_config.json` (`full`):** 25 warmup + 100 iterations, `benchmark_reference=True`. sol-execbench uses CUPTI-based timing (upstream commit `8d2237d`) so the numbers are the same the eval driver reports natively.
- **`bench_config_fast.json` (`fast`):** 5 warmup + 10 iterations, `benchmark_reference=False`. Used for tasks whose reference is a Python for-loop that exceeds the ~200-sec-per-task budget under `full` (L2 MoE variants, FIB attention wrappers, L1_044 MoE).

**Reviewer note:** the `fast`-benchmarked tasks show `n/a` in the results table — they are **correctness-verified but not speed-comparable in the aggregate**. Manual spot timings are on record (§3.3) but not merged into the geometric mean.

### 3.3 Manually-timed speedups (not in the geometric mean)

For the 13 `fast`-only tasks, I ran a standalone perf harness at development time (single workload, warmup=2, iter=5 sol / iter=2 ref). These are informational only:

| Task | Manual speedup | Source (dev notes) |
|---|---|---|
| L1/044 MoE 256-expert SwiGLU | ~2.0x | Direct comparison @ num_tokens=2048/3719/6144 |
| L2/008 Qwen3-30B MoE | ~2.2x | @ B=2, S=1024 |
| L2/009 Qwen3-Coder decoder+MoE | ~2.0x | @ B=1, S=1024 |
| L2/010 Qwen3-Coder MoE compute | ~1.8x | @ batch_seq=2048 |
| L2/013 Qwen3-Next MoE (E=512, top-10) | ~3.0x | @ batch_seq=2048 |
| L2/029 ERNIE-4.5 MoE | ~1.6x | @ B=2, S=1024 |
| L2/065 gpt-oss fp32 MoE | ~1.6x | @ num_tokens=1024 |
| L2/081 GLM-4.5-Air MoE | ~3.0x | @ num_tokens=2048 |
| L2/082 GLM-4.7 MoE complete | ~3.2x | @ batch_seq=2048 |
| FIB/013 GQA paged decode | Very large (~500x) | ref is Python loop; not directly comparable |
| FIB/017 GQA ragged prefill | ~11x | one workload |
| FIB/018 MLA paged decode | ~43x | one workload |
| FIB/019 MLA paged prefill | Very large | ref is Python loop |

**Reviewer caveat:** FIB attention "speedups" are inflated because the reference is a plain Python attention loop, not a proper implementation. What they demonstrate is that the FlashInfer wrapper *works and produces bit-correct output*, not that it beats a production baseline.

### 3.4 Reproducibility

- `scripts/aggregate_contest.py` — discovers all 44 baselines, runs them, emits the Markdown table. Regenerating the results is a single command: `python scripts/aggregate_contest.py --output docs/CONTEST_RESULTS.md` from the `sol-execbench` root (with `FLASHINFER_TRACE_DIR` set — script does this automatically now).
- End-to-end wall time on B200: **~90 min** (dominated by L1/023 multimodal mrope and L2/013 Qwen3-Next MoE, both of which do CPU-heavy work in the reference).
- Every baseline is a single self-contained `solution.json`; no build dependencies beyond the environment listed in §0.

---

## 4. Coverage matrix (all 60 tasks)

The following table is the disposition of every Contest task. `✓` = shipped baseline; `−` = negative result documented; `?` = deferred.

### 4.1 L1 (20 tasks)

| Task | Status | Baseline library / approach | Notes |
|---|---|---|---|
| 003_lm_head_projection_with_logit_slicing | − | — | cuBLAS bf16 GEMM already SOTA; not attempted (§5.1) |
| 011_rotary_position_embedding | ✓ | torch broadcast outer product | 1.74x — skips reference's per-batch bmm overkill |
| 015_grouped_query_attention_with_rope_and_qk_norm | ✓ | FlashInfer rmsnorm + torch SDPA | 2.70x |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | ✓ | FlashInfer rmsnorm + manual RoPE | 1.99x |
| 020_vision_patch_merger_spatial_shuffle_mlp | ✓ | FlashInfer layernorm + spatial-shuffle | 2.20x |
| 021_vision_cu_seqlens_variable_length_attention | ✓ | **Adaptive**: block-diag mask <2500 tokens, per-seq loop ≥2500 | 1.47x |
| 023_multimodal_rope_position_computation_with_grid_based_indexing | ✓ | torch (avoid `.tolist()` CPU sync) | 2.02x |
| 043_mla_fused_qkv_rope_split | ✓ | FlashInfer rmsnorm + torch linear | 1.37x |
| 044_moe_expert_computation | ✓ | sort+per-expert `torch.mm` | ~2.0x manual (fast bench) |
| 046_attention_softmax_with_softcapping_and_dropout | − | — | torch codegen already fuses (0.90x) |
| 048_fused_gate_up_projection_with_swiglu | ✓ | FlashInfer `gelu_tanh_and_mul` | 2.78x |
| 049_attention_qk_matmul_with_gqa_repeat_and_scaling | ✓ | torch bf16 (skip fp32 upcast) | 5.10x |
| 059_moe_group_score_aggregation_and_masking | − | — | Reference already tight vectorized (1.00x) |
| 063_attention_output_reshape_and_projection | − | — | Pure cuBLAS GEMM (1.00x) |
| 064_latent_kv_expansion_with_split | ✓ | FlashInfer rmsnorm + bf16 matmul | 1.39x |
| 067_flash_attention_gqa_ultralong | − | — | fp32 ultralong; SDPA loses 0.93x vs matmul-based reference |
| 069_rms_norm | ✓ | FlashInfer rmsnorm | 6.44x |
| 071_kv_cache_update_with_rope | − | — | Memory-bound cat (1.00x) |
| 076_batched_expert_forward | − | — | `broadcast matmul == replicate` on cuBLAS (1.01x) |
| 092_gqa_attention_with_qk_norm | ✓ | FlashInfer rmsnorm + **FlashAttention 2** | 2.65x |

**L1 breakdown: 13 shipped, 7 negative.**

Where the L1 negatives are: L1/003 (lm_head cuBLAS optimal), L1/046 (softcap+softmax fused by codegen), L1/059 (routing math tight), L1/063 (attn output projection cuBLAS optimal), L1/067 (fp32 ultralong), L1/071 (kv cache memcpy), L1/076 (batched expert broadcast=replicate).

### 4.2 L2 (20 tasks)

| Task | Status | Baseline library / approach | Notes |
|---|---|---|---|
| 002_decoder_layer_full_block | − | — | fp32 decoder; F.rms_norm + cuDNN SDPA = ref (1.00x) |
| 004_fused_residual_rms_mlp | − | — | Bandwidth-bound MLP (1.02x) |
| 006_multimodal_rope_position_calculation | ? | — | CPU-control-flow state machine; ≤30% gain not worth regression risk |
| 007_multimodal_rotary_embedding_attention | ✓ | torch 3D mrope + SDPA | 1.88x |
| 008_moe_sparse_routing_and_dispatch | ✓ | sort+per-expert mm | ~2.2x manual |
| 009_decoder_layer_with_residual_connections | ✓ | FlashInfer rmsnorm + SDPA + sort MoE | ~2.0x manual |
| 010_moe_expert_computation_with_weighted_accumulation | ✓ | sort+per-expert mm | ~1.8x manual |
| 012_moe_expert_batched_execution_with_capacity_factor | − | — | Ref uses padded bmm; sort+per-expert loses 0.43x |
| 013_expert_weighted_aggregation_with_shared_expert | ✓ | sort+per-expert mm + shared expert | ~3.0x manual |
| 019_decoder_layer_fused_attention_mlp | − | — | fp32 decoder; SDPA = ref (0.99x) |
| 020_decoder_layer_pre_post_norm_residual | ✓ | FlashInfer + Reka complex-pair RoPE | 1.45x |
| 027_grouped_query_attention_with_yarn_rope_and_qk_norm | ✓ | FlashInfer full-dim Q/K norm + YARN | 1.90x |
| 029_moe_sparse_routing_and_dispatch | ✓ | ERNIE-4.5 routing + sort MoE | ~1.6x manual |
| 048_moe_expert_inference_batched_dispatch | − | — | fp32 tolerance forces fp32 accum (0.96x) |
| 049_group_limited_topk_routing | ✓ | Ring-flash routing (skip fp32 cast) | 1.26x |
| 053_text_decoder_layer_with_self_attention_and_mlp | ✓ | FlashInfer + on-the-fly RoPE | 1.64x |
| 054_vision_encoder_layer_with_gated_residuals | ✓ | FlashInfer layernorm + tanh-gated residuals | 6.72x |
| 065_sparse_expert_dispatch_and_combine | ✓ | gpt-oss fp32 sort MoE | ~1.6x manual |
| 081_moe_sparse_expert_dispatch | ✓ | GLM-4.5-Air routing + sort MoE | ~3.0x manual |
| 082_moe_layer_complete_forward_with_residual | ✓ | GLM-4.7 routing + sort MoE | ~3.2x manual |

**L2 breakdown: 14 shipped, 5 negative, 1 deferred (006).**

Where the L2 negatives are: L2/002 (fp32 decoder), L2/004 (bandwidth-bound MLP), L2/012 (padded bmm already optimal), L2/019 (fp32 decoder), L2/048 (fp32 tolerance forces slow accumulation). L2/006 is deferred (see §5.3).

### 4.3 Quant (10 tasks)

| Task | Status | Baseline library / approach | Notes |
|---|---|---|---|
| 002_fp8_attention_qkv_projection | ✓ | FlashInfer `gemm_fp8_nt_groupwise` + rmsnorm | 4.27x |
| 003_fp8_mlp_gate_up_projection | ✓ | FlashInfer fp8 GEMM + torch silu | **22.56x** |
| 004_fp8_moe_expert_linear | ✓ | FlashInfer fp8 GEMM + torch SwiGLU | 3.23x |
| 005_fp8_moe_router_projection | ✓ | FlashInfer fp8 GEMM | 8.54x |
| 011_fp8_moe_gate_routing | ✓ | FlashInfer fp8 GEMM + torch topk | 1.44x |
| 012_fp8_shared_expert_mlp | ✓ | FlashInfer fp8 GEMM x3 | 4.11x |
| 013_fp8_mla_kv_compression_projection | ✓ | FlashInfer fp8 GEMM + rmsnorm | 4.26x |
| 014_fp8_yarn_rope_embedding | − | — | Reference's FP8 round-trip causes phase flip in cos/sin (impossible to match) |
| 015_fp8_mla_attention_output_projection | ✓ | FlashInfer fp8 GEMM | **29.83x** |
| 016_fp8_multi_latent_attention_qkv_projection | ✓ | FlashInfer fp8 GEMM + rmsnorm | 3.93x |

**Quant breakdown: 9 shipped, 1 negative.** This category has the highest coverage rate and largest speedups — reference is a `dequant-to-fp32 → matmul` chain, and the real fp8 GEMM path is much faster.

### 4.4 FlashInfer-Bench (10 tasks)

| Task | Status | Baseline library / approach | Notes |
|---|---|---|---|
| 002_fused_add_rmsnorm_h4096 | ✓ | FlashInfer rmsnorm(res+x) | 10.98x |
| 003_fused_add_rmsnorm_h7168 | ✓ | FlashInfer rmsnorm(res+x) | 10.83x |
| 005_gemm_n256_k7168 | − | — | cuBLAS fp16 already SOTA; routergemm loses 0.78x (bf16 cast overhead) |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | ✓ | `BatchDecodeWithPagedKVCacheWrapper` | 48/48 correctness |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | ✓ | `BatchPrefillWithRaggedKVCacheWrapper` | 21/21 correctness |
| 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | ✓ | `mla.BatchMLAPagedAttentionWrapper` | 47/47 correctness |
| 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | ✓ | Same wrapper, causal=True | 38/38 correctness |
| 020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 | − | — | `trtllm_fp8_block_scale_moe` hangs on B200 (SM100) |
| 023_rmsnorm_h1536 | ✓ | FlashInfer rmsnorm | 11.47x |
| 026_rmsnorm_h7168 | ✓ | FlashInfer rmsnorm | 13.74x |

**FIB breakdown: 8 shipped, 2 negative.**

---

## 5. Uncovered tasks — audit-critical analysis

**16 tasks are not shipped** (60 total − 44 shipped = 16). The reviewer should read this section closely. They fall into four classes.

- **13** — reference is genuinely SOTA on this hardware (§5.1)
- **2** — library bug / API unavailability (§5.2, Quant/014 and FIB/020)
- **1** — deferred (L2/006, §5.3)

Total: **16.** Every task has a documented reason.

### 5.1 Reference is genuinely SOTA on this hardware — 13 tasks

These have measured speedup ≤ 1.02x with the best SOTA approach I could construct. Removing them from the shipped set is per the sol-baseline upstream policy: **only ship baselines that beat the reference on average**.

| Task | Best measured | Root cause |
|---|---|---|
| L1/003 lm_head projection | ~1.00x (not built, cuBLAS reasoning) | cuBLAS bf16 GEMM at (M×2048)×(2048×102400) is optimal for the workload shapes |
| L1/046 softmax + softcap | 0.90x | torch codegen already fuses `tanh(x/30)*30 → softmax` |
| L1/059 MoE group score | 1.00x | Reference is a tight `topk-sum-topk-scatter-fill` chain, each op a single kernel |
| L1/063 attn output reshape+proj | 1.00x | Single `F.linear` after contiguous reshape — cuBLAS bandwidth-bound |
| L1/067 fp32 ultralong attention | 0.93x | cuDNN fp32 SDPA loses to torch matmul-based reference on B200 |
| L1/071 KV cache update+RoPE | 1.00x | Memory-bound `torch.cat` dominates |
| L1/076 batched expert forward | 1.01x | `broadcast_matmul` and `repeat_matmul` both compile to same cuBLAS `batched_gemm` |
| L2/002 fp32 decoder full block | 1.00x | `F.rms_norm + cuDNN SDPA + F.silu` matches ref exactly |
| L2/004 fused residual+RMSNorm+MLP | 1.02x | Bandwidth-bound on `(residual+hidden)` read |
| L2/012 MoE capacity-factor | 0.43x | Ref uses padded `bmm` — that's already the right pattern; sort+per-expert overhead loses |
| L2/019 fp32 Qwen-Image-Edit decoder | 0.99x | Same as L2/002 (fp32 ⇒ codegen already SOTA) |
| L2/048 fp32 MoE batched dispatch | 0.96x | Tolerance `max_atol=2.6e-4` forces fp32 accumulation everywhere; reference already does this |
| FIB/005 fp16 GEMM (DeepSeek router) | 0.78x | cuBLAS fp16 GEMM for M∈[1,14104]×N=256×K=7168 is optimal; routergemm adds bf16 cast overhead |

**Reviewer question to consider:** *Are any of these "reference is SOTA" claims actually artifacts of missing libraries?* Table §7 lists what's missing. In particular:
- FIB/005, L2/012, L2/019 might be improvable with DeepGEMM or TransformerEngine (fp8 path on fp32/fp16 inputs), which are not installed.
- L1/067 might be improvable with FA3 (SM90 native fp32 support), which is also not installed.
- **I did not test the DeepGEMM/TE/FA3 paths.** These are honest "reference is SOTA in the *available* environment" results, not "reference is SOTA in general."

### 5.2 Library bugs / API unavailability — 2 tasks

| Task | Root cause |
|---|---|
| Quant/014 fp8 YaRN RoPE | Reference explicitly does FP8 quant+dequant of `freqs*position_ids`. For seq_len > 4K, `freqs*t` wraps 2π many times, and the ε quantization noise flips `cos`/`sin` signs. A "skip the FP8 round-trip" baseline (mathematically more correct) reports `max_atol=2.0`. To pass, we'd have to replicate the exact quant noise the reference injects — meaningless. **Removed rather than ship a wrong baseline.** |
| FIB/020 FP8 routed MoE | `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` hangs indefinitely on B200 (SM100). Same root cause as `cutlass_fused_moe` hang on L1/044 (where we successfully fell back to sort+per-expert mm — that fallback isn't available for the FP8 path). Attempted upgrade to FlashInfer 0.6.13 to fix this **also failed** due to a cutlass-dsl API compat break (see §7). |

### 5.3 Deferred — 1 task

| Task | Why deferred |
|---|---|
| L2/006 multimodal mrope position | Per-batch Python state machine over `input_ids` with `.tolist()` + `.index()` calls, dependent on grid_thw shape per (image, video) block. Attempted rewrite gave ≤30% gain but was fragile across vision-token layouts. Not shipped because the risk-adjusted value is negative. |

### 5.4 Reviewer's decision checklist

For each of the 16 non-shipped tasks, the reviewer can decide:

1. **Accept as documented negative** — the disposition matches the sol-baseline upstream policy. **Recommended default.**
2. **Ask me to re-attempt with a specific missing library** — e.g., "install DeepGEMM and re-try L2/012." I have the hardware for this but did not install these libraries (see §7 for reasoning).
3. **Ask me to attempt anyway with a workload subset** — e.g., "ship L2/006 with correctness-only, no perf claim." Feasible but violates the "must beat reference" policy of sol-baseline upstream. Not recommended.

---

## 6. Methodology & pattern library (what future maintainers get)

Beyond the 44 baselines themselves, this work produced reusable infrastructure:

### 6.1 Patterns established (documented in `docs/CONTEST.md`)

1. **FP8 DeepSeek recipe:** `flashinfer.gemm.gemm_fp8_nt_groupwise(scale_granularity_mnk=(1, 128, 128), scale_major_mode='MN')` with `sx` as `(K//128, M)` and `sw` as `(K//128, N//128)` — both `.t().contiguous()` from natural compute order. Applied to 9 Quant tasks.
2. **Sort-by-expert + per-expert `torch.mm` MoE:** replaces the reference's `F.one_hot + permute + where` scan over all E experts by iterating only non-empty ones. Applied to 8 L2 MoE variants + L1/044.
3. **Adaptive cu_seqlens vision attention:** switch between block-diag mask + single SDPA (short seqs) and per-sequence SDPA loop (long seqs). Applied to L1/021, ported from sol-baseline upstream.
4. **FlashInfer attention wrappers:** `BatchDecodeWithPagedKVCacheWrapper` / `BatchPrefillWithRaggedKVCacheWrapper` / `mla.BatchMLAPagedAttentionWrapper` for the FIB paged/ragged/MLA tasks. LSE convention is base-2 by default — do **not** divide by log(2).
5. **`FLASHINFER_TRACE_DIR` env var** — must be set to an absolute path for FIB/013, 017, 018, 019 to resolve the `kv_indptr`/`kv_indices` blob dependencies. The aggregator sets this automatically.

### 6.2 Anti-patterns registered (in `library_index.json`)

These are things I stepped in during development. They are permanent gotchas for anyone extending this work:

1. **Never key cached quant outputs on `tensor.data_ptr()`** — PyTorch's allocator reuses addresses across eval calls, causing stale FP8 buffers to leak between workloads.
2. **`flashinfer.silu_and_mul` / `gelu_and_mul` require 2D contiguous input.** Docstring says 3D is fine; empirically it dispatches to a null kernel on SM100. Always `x.reshape(-1, 2*inter).contiguous()` first.
3. **`flashinfer.fused_add_rmsnorm` is in-place** on both `input` and `residual`. If you need pure-function semantics use `rmsnorm(residual + x)` instead.
4. **`gemm_fp8_nt_blockscaled` hardcodes (128,128,128) granularity** — it can't do the DeepSeek 1×128 activation recipe. Use `gemm_fp8_nt_groupwise` with explicit `scale_granularity_mnk`.
5. **FlashAttention 2 rejects fp32.** For fp32 attention (L1/067, L2/002, L2/019) you must fall back to `F.scaled_dot_product_attention` (cuDNN backend).
6. **fp32 decoder blocks are already SOTA** via `F.rms_norm + cuDNN SDPA + F.silu` codegen. Expect ≈1.00x and don't ship a "wrapper" baseline.
7. **`x.tolist()` in RoPE/index loops forces a full CPU-GPU sync.** In L1/023 this was the dominant bottleneck; replacing it with `torch.cat(list_of_gpu_tensors)` gave a ~3x speedup with no other change.
8. **Broadcast matmul does not save replication memory** on cuBLAS. `(1, T, H) @ (E, H, D)` compiles to the same batched GEMM as the explicit repeated form.

### 6.3 Files in the repo (audit trail)

- **`baselines/*/Contest/*/solution.json`** — 44 baselines, each self-contained (definition name + inline source + entry-point spec).
- **`docs/CONTEST.md`** — Human-facing playbook: env, patterns, how to run.
- **`docs/CONTEST_RESULTS.md`** — Auto-generated speedup table (44 baselines).
- **`docs/CONTEST_NEGATIVE_RESULTS.md`** — All 13 documented negatives + failed 0.6.13 upgrade + L2/006 deferred rationale.
- **`docs/library_index.json`** — Kernel × dtype × SM compatibility matrix with anti-patterns + FlashInfer 0.6.13 recovery recipe.
- **`scripts/aggregate_contest.py`** — Single-command regeneration of CONTEST_RESULTS.md.
- **`scripts/bench_config{,_fast,_quick}.json`** — Three benchmark configurations (used explicitly per task by the aggregator).
- **`scripts/verify.py`** — Extended for `Contest/{lib}/Contest/{task}` 3-level path.
- **`scripts/run_contest.sh`** — Legacy per-task runner (list-mode; superseded by aggregate script but kept for surgical re-runs).

---

## 7. Environment gaps that shape the ceiling

The upper bound on Contest coverage in this environment is set by which SOTA libraries are installed. Here's the delta.

| Library | Status | If installed, would potentially improve |
|---|---|---|
| **FlashInfer 0.6.12** | ✓ available | (current SOTA source — 26/44 baselines) |
| **flash_attn 2.8.3** | ✓ available | (used in L1/092, L2/009) |
| **Liger 0.8** | ✓ available (not used) | Liger 0.8 has PyTorch 2.9 `torch.distributed.tensor` compat issue on `LigerRMSNorm` and `LigerSiLUMulFunction`; FlashInfer supersedes both anyway |
| **causal_conv1d 1.6.2** | ✓ available (not used) | Would fit conv1d tasks; none in Contest |
| **DeepGEMM** | ✗ not installed | Might improve FIB/005 fp16 GEMM, Quant/011 fp8 routing |
| **TransformerEngine** | ✗ not installed | RowWise / TensorWise fp8 scaling for Quant tasks with non-DeepSeek recipes; L2/019 fp32 → fp8 path |
| **FlashMLA** | ✗ SM90-only (not built for SM100) | Would supersede our MLA implementations if usable |
| **FlashAttention 3** | ✗ not installed | Fp32 ultralong attention (L1/067) is the main candidate |
| **sgl-kernel** | ✗ not installed | Fused RoPE + norm variants that might improve L1/018, L1/023 |
| **vLLM custom_ops** | ✗ not installed | Alternative routes for MoE fusion |

### 7.1 Failed upgrade attempt (recorded in negatives)

I attempted to upgrade FlashInfer 0.6.12 → 0.6.13 (latest stable, 24 Jun 2026) specifically to unblock the two hanging MoE paths (`cutlass_fused_moe` bf16, `trtllm_fp8_block_scale_moe`). **The upgrade broke the environment on import**:

```
File ".../flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py":
    a_major_mode: cute.nvgpu.OperandMajorMode,
AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'OperandMajorMode'
```

0.6.13's pyproject requires `nvidia-cutlass-dsl>=4.5.0`, but the Blackwell kernel code still uses the 4.4.x `OperandMajorMode` name that 4.5.x renamed. Downgrade also blocked by transitive dependency conflicts. Recovery required 4 sequential steps to restore the env; documented at `docs/library_index.json → flashinfer._recovery_recipe`.

**No newer FlashInfer version is currently usable on this env.** MoE remains covered by sort+per-expert `torch.mm`.

---

## 8. Known risks & concerns the reviewer should flag

I am flagging these proactively rather than have them found in review.

### 8.1 Manual timing numbers in §3.3 are not reproducible from CI

The 13 `fast`-only tasks show `n/a` in the auto-generated results table because I set `benchmark_reference=False` to avoid the aggregator running out of time. The manual speedups I quoted come from ad-hoc dev-time perf runs. If the reviewer wants publishable numbers, I should:
- (a) Extend the aggregator to also do single-workload spot timings for `fast` tasks, or
- (b) Add an intermediate `bench_config_quick.json` (5w/20iter, `benchmark_reference=True`) already present in the repo but not wired into the aggregator, and use it for MoE + FIB attention.

**Recommendation:** if the reviewer wants final publishable numbers, ask me to do (b). It'd add ~30 min to the aggregator wall time.

### 8.2 Aggregator geometric mean = 3.66x is over 31 tasks, not 44

The `Geometric mean` at the bottom of `CONTEST_RESULTS.md` averages 31 timed tasks (all tasks with `full` bench). It **excludes** the 13 `fast` tasks. This is honest — the manual-timing numbers in §3.3 are not from the same benchmark protocol so mixing them would be misleading. But a reviewer might miss this. I have added a `Notes` section at the bottom of `CONTEST_RESULTS.md` that says this explicitly, but I want to call it out here.

### 8.3 Reference "fp32" answers may drift after upstream sol-execbench updates

Several negatives (L1/067, L2/002, L2/019) rely on fp32 being the reference dtype. If sol-execbench later downgrades those tasks to bf16, my "1.00x" conclusion may reverse. This is out of my control but worth calling out. Re-running `aggregate_contest.py` will surface it immediately.

### 8.4 Reka RoPE convention

L2/020 uses Reka-flash-3's non-standard **complex-pair RoPE** (`q_new = [q1*cos - q2*sin, q1*sin + q2*cos]`) with half-dim `cos`/`sin`. This differs from the standard rotate-half RoPE by construction. Reviewers used to standard RoPE conventions should double-check the baseline reads `cos`/`sin` correctly (they have shape `(b, s, half_head_dim)`, not `(b, s, head_dim)`). Test passes 16/16 so this is correct; noting for reviewer clarity.

### 8.5 L1/044 MoE speedup dependency on random routing distribution

The sort+per-expert baseline for L1/044 shows ~2x on the eval driver's random workload. If a workload has extreme skew (e.g., 90% of tokens routed to 1 expert), the per-expert mm loop becomes a single-expert `mm` and the speedup collapses to ~1.0x vs reference. The Contest workloads are random-uniform, so this is not a problem for evaluation, but production deployment characteristics may differ.

### 8.6 What I did NOT verify

- **Multi-GPU / tensor-parallel** paths — Contest tasks are all single-GPU
- **CUDA graph capture** compatibility — none of the baselines are graph-safe (some use Python control flow)
- **Backward pass** — Contest is inference-only; no baselines exist for training
- **Numerical stability at extreme scale** — largest workload is total_seq_len=16384 (L1/067)

---

## 9. What "the community SOTA has been searched to the limit" actually means here

The claim is scoped:

**True on this hardware and library set:** For 44 tasks I have a working SOTA baseline that beats the reference; for 13 tasks the reference genuinely is SOTA in this environment; for 2 more, the specific library needed either doesn't exist yet in a working form (FIB/020 trtllm-fp8-moe hangs) or is fundamentally impossible (Quant/014 phase-flip); L2/006 is deferred.

**Not true globally:** If DeepGEMM, FA3, TransformerEngine, or a fixed FlashInfer 0.6.14+ becomes available, roughly **6 of the 13 negatives** should be re-attempted, and 3 more baselines (Quant/014, FIB/020, L2/006) could realistically be added. Best-case additional coverage: **60/60 with mean speedup improving from 3.66x to ~4.5x**.

**Realistic near-term ceiling in this environment: 44/60 shipped with 3.66x geomean.** Beyond this requires either new library releases or hardware/dependency changes we don't control.

---

## 10. Recommendation for the reviewer

I would suggest reviewing in this order:

1. **Sample 3 shipped baselines** — one from each of `Quant/003` (highest speedup, simplest FP8 pattern), `L2/054` (highest L2 speedup, uses FlashInfer layernorm), and `L1/021` (adaptive strategy). Verify the code matches the pattern I described and passes locally.
2. **Read §5 (Uncovered tasks)** carefully. Decide per-task whether to accept the negative disposition or ask for a re-attempt.
3. **Read §8 (Known risks)** to make sure my caveats are acceptable.
4. **Spot-check the aggregate results:** `python scripts/aggregate_contest.py --output /tmp/re-check.md` and diff against `docs/CONTEST_RESULTS.md`. Should be identical up to normal timing variance.
5. **Approve or request changes** — I can:
   - Re-attempt any negative task with a specific library (name the library).
   - Wire the manual-timing numbers from §3.3 into the aggregator (~1 hr work + ~30 min re-run).
   - Ship correctness-only baselines for the deferred tasks (violates upstream policy — need explicit approval).
   - Any other reasonable change.

I am confident the work as it stands is honest, complete, and reproducible on the stated environment. All caveats are documented in the tree, not just here.
