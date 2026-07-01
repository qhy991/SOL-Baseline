# Contest Negative Results

Cases where we tried but the SOTA path didn't beat the reference, or the reference's own behavior makes a strict baseline impossible.

## Contest/FIB/005_gemm_n256_k7168 — cuBLAS fp16 already SOTA

`torch.matmul(A, B.t())` in fp16 is the reference; cuBLAS handles this exact shape (M=1–14104, N=256, K=7168) optimally on B200. Tried adaptive routing to `flashinfer.gemm.mm_M1_16_K7168_N256` for M≤16 (requires bf16 cast + persistent output buffer), but the cast overhead and the M=1 path's per-call autotuning give **0.78x mean speedup** (0.25x at M=1, 1.0x for M>16). Removed.

## Contest/FIB/020_moe_fp8_block_scale_ds_routing — trtllm fp8 MoE hangs on B200

`flashinfer.fused_moe.trtllm_fp8_block_scale_moe` with `routing_method_type=DeepSeekV3` either autotunes indefinitely or hits an sm_100 codegen path that doesn't return — standalone test on a synthetic 128-token workload runs >3 min without completing. Same root cause as `cutlass_fused_moe` bf16 path on B200 (see L1/044 note above): FlashInfer 0.6.12 MoE GEMM kernels aren't reliably compiled for SM 100 in this environment. Removed pending a newer FlashInfer or DeepEP build.

## Contest/L1/046_attention_softmax_with_softcapping_and_dropout — codegen already fuses

`scaled = x/30 → tanh → *30 → softmax`. Torch's elementwise codegen already fuses the three pre-softmax stages, and `softmax(..., dtype=float32)` is the SOTA implementation. Replacing with `30*tanh(x/30) → softmax` gives **0.90x** because there's nothing more to fuse. Removed.

## Contest/L1/059_moe_group_score_aggregation_and_masking — reference already tight

Pure vectorized chain (`topk(2).sum → topk(4) → scatter → expand mask → masked_fill`). Each op is a single CUDA kernel. **1.00x**. Removed.

## Contest/L1/076_batched_expert_forward — broadcast == replicate

Reference does `hidden.repeat(E, 1).view(E, T, H)` (explicit replication) then `bmm`. Tried `(1, T, H) @ (E, H, 2D)` broadcast to avoid the materialization — cuBLAS handles both identically and result is **1.01x**. Removed. Real speedup would require a sparse path (which routing_weights doesn't have here — it's dense random) or a fused expert+routing kernel like DeepEP. Not in this environment.

## Contest/L2/002_decoder_layer_full_block (fp32) — torch native already SOTA

DeepHermes-3 Llama-3-8B decoder, **fp32**. F.rms_norm + F.scaled_dot_product_attention (cuDNN, enable_gqa) + F.silu fused MLP gets **1.00x** — the reference's manual variance/rotate-half/matmul chain compiles to the same kernels via PyTorch's codegen. FlashInfer kernels are bf16-only so no FP8/normfusion fallback. Removed.

## Contest/L2/004_fused_residual_rms_mlp — 1.02x bandwidth-bound

h=16384, inter=53248. FlashInfer rmsnorm + cat-fused gate/up + silu_and_mul vs reference's straight torch chain. **1.02x mean** — reference is already memory-bound on (residual+hidden) read and the three GEMM stages dominate. Removed.

## Contest/L2/019_decoder_layer_fused_attention_mlp (fp32) — torch SDPA == reference

Qwen-Image-Edit fp32 decoder with 3D mrope. FA2 is bf16-only so we use F.scaled_dot_product_attention (cuDNN backend) — **0.99x**. Same conclusion as L2/002 (fp32 reference already calls fused cuDNN kernels). Removed.

## FlashInfer 0.6.13 upgrade attempt — broken cutlass_dsl API compat

Tried `flashinfer-python==0.6.13` (24 Jun 2026, latest stable) to see if MoE paths (`cutlass_fused_moe` bf16, `trtllm_fp8_block_scale_moe`) are fixed on SM100. Install succeeds but **any `import flashinfer` fails immediately**:

```
File ".../flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py", line 2383,
    in Sm100BlockScaledPersistentDenseGemmKernel
    a_major_mode: cute.nvgpu.OperandMajorMode,
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'OperandMajorMode'
```

The Blackwell blockscaled kernel file references `cute.nvgpu.OperandMajorMode` from cutlass-dsl 4.4.x, but 0.6.13's pyproject requires `nvidia-cutlass-dsl>=4.5.0`, and 4.5.x renamed that symbol. The two pinned deps are mutually incompatible. Downgrade also blocked: `flashinfer 0.6.12 → cutlass-dsl>=4.5.0` in metadata even though 4.4.1 is what actually works.

**Recovered by:** `uv sync --all-groups` + reinstall `flashinfer-python==0.6.12 --no-deps` + pin `nvidia-cusparselt-cu13` (0.8→0.9.1) + `nvidia-nvshmem-cu13` (3.3→3.7.1) + `pynvml`.

Wait for a 0.6.14 / 0.7.0 that resolves the cutlass-dsl split, or build FlashInfer from source pinning cutlass-dsl==4.4.1 (out of scope for this environment). MoE remains covered by sort+per-expert `torch.mm` for now.

## Contest/L2/006 — multimodal mrope position calculation: CPU-control-flow dominates

Per-batch Python state machine over input_ids: `attention_mask_bool`, `torch.argwhere`, then for each (image, video) block: `input_tokens.tolist()` + `input_tokens.index(token_id, st)` + per-grid `t.item()/h.item()/w.item()` + position-id concatenation. The reference's CPU runtime dominates the per-call cost, and a faithful rewrite still has to honor the sequential state (start-of-next-block depends on end-of-previous-block + grid_thw size). Tried; the small (≤30%) speedup from batching `.index()` lookups with a single `torch.where` doesn't justify the regression risk on diverse vision-token layouts. Deferred.

## Contest/L1/063_attention_output_reshape_and_projection — pure cuBLAS, reference optimal

`transpose(1,2).reshape(b,s,H*D) @ o_proj_weight.t()` — reference is a single `F.linear` after a contiguous reshape. **1.00x measured (0.98x–1.03x)**. No SOTA library improves on cuBLAS GEMM for this shape; the contiguous reshape is bandwidth-bound. Removed.

## Contest/L1/071_kv_cache_update_with_rope — pure memory copy + small RoPE

`rotate_half + cos/sin multiply + torch.cat`. Reference is memory-bound on the cat. **1.00x measured**. Removed.

## Contest/L2/012_moe_expert_batched_execution_with_capacity_factor — padded bmm beats per-expert loop

Reference uses 1.25x capacity and runs **three batched `bmm`** over the padded `(E, capacity, hidden)` buffer. This is genuinely fast — the bmm kernel exploits the rectangular shape and avoids the per-expert iteration. Sort+per-expert mm baseline runs at **0.43x** vs reference: the overhead of iterating 160 experts (and the per-call `mm` launch latency) outweighs avoiding the small amount of capacity-overflow work.

Conclusion: when the reference already does padded bmm and `capacity_factor` is close to the average expert load, **don't try to optimize it**. The right strategy would be DeepEP-style permuted grouped GEMM, which isn't available on this B200 environment.

## Contest/L2/048_moe_expert_inference_batched_dispatch — fp32 upcast eats all gains

Tolerance is `max_atol=2.6e-4`, which forces fp32 accumulation everywhere (cast hidden_states, all expert weights, expert outputs to fp32). Reference does this too. Result: **0.96x vs reference** — the sort+per-expert path matches the reference's `tokens_per_expert_cpu` loop almost exactly because both spend their time on the same fp32 matmuls.

There's no free lunch here unless we relax precision (we can't — tolerance won't allow it). DeepGEMM / TE FP8 paths *could* win but require libraries not in this environment.

## Quant/014 fp8_yarn_rope_embedding — phase-flip from required FP8 round-trip


**Reference does an FP8 quant→dequant of the YaRN frequency table**, then `cos/sin` of the perturbed freqs. The quant noise on `freqs * position_id` is tiny in absolute terms (~1e-3), but for `seq_len > 4K` the `freqs * t` magnitudes reach 10^4+ and wrap many times around 2π. Even ε-perturbations near a 2π boundary flip the cos/sin sign.

A "skip the FP8 round-trip" baseline (pure fp32 then `.cos()/.sin()`) is mathematically more correct but reports `max_atol=2.0` against the reference because the phases disagree. To pass we'd have to replicate the exact quant noise the reference injects — defeating the point of optimizing it.

**Removed** rather than shipping a wrong baseline.

## Contest/L1/067 flash_attention_gqa_ultralong (fp32) — torch SDPA loses to torch matmul-based reference

The reference is fp32 matmul-based attention with an explicit `triu` causal mask. We tried `F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)` (cuDNN backend) — passes 18/18 but **0.93x average** (0.85x–0.98x). The cuDNN fp32 attention kernel has worse arithmetic intensity than naked matmul-softmax-matmul on Blackwell SM100 — likely because cuDNN's fp32 SDPA backend was tuned for Ada/Hopper.

FA2 won't help (bf16 only). FA3 isn't installed. FlashMLA is SM90-only.

This task is **reference-already-SOTA in the same way as L2_002/L2_063 in sol-baseline upstream**: torch fp32 matmul already calls cuBLAS, and the for-loop overhead is amortized.

## FlashInfer `cutlass_fused_moe` bf16 path on SM100 — hangs

Tried for `Contest/L1/044_moe_expert_computation` (DeepSeek-V3 top-8/E=256). Direct call to `flashinfer.fused_moe.cutlass_fused_moe(..., quant_scales=[])` with bf16 weights blocks indefinitely (or runs an autotuning sweep that doesn't terminate within 5min). Suspect either the bf16 path isn't compiled for sm_100 in 0.6.12, or it expects a specific weight-layout reshuffle.

Fell back to sort-tokens-by-expert + per-expert `torch.mm` — passes 16/16, ~2x vs reference. Note this is unrelated to **sol-baseline upstream's L2_008 SegmentGEMM result** (which lost because reference per-expert torch.mm was *already* fast). Here we win because the reference uses a slower `where`-mask scatter loop.

## Sub-pattern findings

- **Never key cached fp8 weights on `tensor.data_ptr()`** — PyTorch's allocator reuses addresses between successive eval calls, so a stale FP8 buffer gets returned for a freshly-generated weight. Both Quant/002 and L1_048 v1 failed for this reason; both fixed by removing the cache.
- **`gemm_fp8_nt_blockscaled` is not generic blockwise.** It hardcodes (128,128,128) granularity, so it cannot handle DeepSeek-style 1×128 activation × 128×128 weight scaling. Use `gemm_fp8_nt_groupwise(scale_granularity_mnk=(1, 128, 128), scale_major_mode='MN')` instead.
- **`gemm_fp8_nt_groupwise` `out=` cannot point at a non-contiguous slice.** Writing gate/up into adjacent halves of one buffer to enable `silu_and_mul` fails silently; do two separate output tensors and `F.silu(g) * u` instead.
