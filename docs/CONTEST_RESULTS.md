# Contest baselines — aggregate results

Total: **44 baselines**, environment: NVIDIA B200 (SM100), CUDA 13, torch 2.9, FlashInfer 0.6.12, flash_attn 2.8.3.

| Category | Task | Library | Pass | Mean speedup | Range | Bench |
|---|---|---|---|---|---|---|
| FlashInfer-Bench | 002_fused_add_rmsnorm_h4096 | flashinfer | 14/14 | 10.98x | 9.62–12.63x | full |
| FlashInfer-Bench | 003_fused_add_rmsnorm_h7168 | flashinfer | 8/8 | 10.83x | 8.86–11.85x | full |
| FlashInfer-Bench | 013_gqa_paged_decode_h32_kv8_d128_ps1 | flashinfer | 48/48 | n/a | n/a | fast |
| FlashInfer-Bench | 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flashinfer | 21/21 | n/a | n/a | fast |
| FlashInfer-Bench | 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | flashinfer | 47/47 | n/a | n/a | fast |
| FlashInfer-Bench | 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | flashinfer | 38/38 | n/a | n/a | fast |
| FlashInfer-Bench | 023_rmsnorm_h1536 | flashinfer | 8/8 | 11.47x | 10.47–12.81x | full |
| FlashInfer-Bench | 026_rmsnorm_h7168 | flashinfer | 8/8 | 13.74x | 10.61–18.15x | full |
| L1 | 011_rotary_position_embedding | torch | 16/16 | 1.74x | 1.13–2.14x | full |
| L1 | 015_grouped_query_attention_with_rope_and_qk_norm | torch | 16/16 | 2.70x | 1.69–7.19x | full |
| L1 | 018_fused_rope_with_qk_norm_and_kv_cache_update | flashinfer | 13/13 | 1.99x | 1.64–3.12x | full |
| L1 | 020_vision_patch_merger_spatial_shuffle_mlp | flashinfer | 15/15 | 2.20x | 1.68–2.86x | full |
| L1 | 021_vision_cu_seqlens_variable_length_attention | torch | 16/16 | 1.47x | 0.94–1.86x | full |
| L1 | 023_multimodal_rope_position_computation_with_grid_based_indexing | torch | 16/16 | 2.02x | 1.41–7.64x | full |
| L1 | 043_mla_fused_qkv_rope_split | flashinfer | 16/16 | 1.37x | 0.88–2.27x | full |
| L1 | 044_moe_expert_computation | torch | 16/16 | n/a | n/a | fast |
| L1 | 048_fused_gate_up_projection_with_swiglu | flashinfer | 16/16 | 2.78x | 1.59–5.51x | full |
| L1 | 049_attention_qk_matmul_with_gqa_repeat_and_scaling | torch | 16/16 | 5.10x | 1.64–10.38x | full |
| L1 | 064_latent_kv_expansion_with_split | flashinfer | 16/16 | 1.39x | 1.06–4.50x | full |
| L1 | 069_rms_norm | flashinfer | 16/16 | 6.44x | 4.84–7.75x | full |
| L1 | 092_gqa_attention_with_qk_norm | flash_attn | 16/16 | 2.65x | 1.46–5.40x | full |
| L2 | 007_multimodal_rotary_embedding_attention | torch | 16/16 | 1.88x | 1.06–4.76x | full |
| L2 | 008_moe_sparse_routing_and_dispatch | torch | 16/16 | n/a | n/a | fast |
| L2 | 009_decoder_layer_with_residual_connections | flash_attn | 16/16 | n/a | n/a | fast |
| L2 | 010_moe_expert_computation_with_weighted_accumulation | torch | 16/16 | n/a | n/a | fast |
| L2 | 013_expert_weighted_aggregation_with_shared_expert | torch | 16/16 | n/a | n/a | fast |
| L2 | 020_decoder_layer_pre_post_norm_residual | torch | 16/16 | 1.45x | 1.18–2.35x | full |
| L2 | 027_grouped_query_attention_with_yarn_rope_and_qk_norm | torch | 16/16 | 1.90x | 1.33–3.44x | full |
| L2 | 029_moe_sparse_routing_and_dispatch | torch | 9/9 | n/a | n/a | fast |
| L2 | 049_group_limited_topk_routing | torch | 16/16 | 1.26x | 0.90–2.15x | full |
| L2 | 053_text_decoder_layer_with_self_attention_and_mlp | torch | 16/16 | 1.64x | 1.13–4.61x | full |
| L2 | 054_vision_encoder_layer_with_gated_residuals | flashinfer | 16/16 | 6.72x | 0.73–21.16x | full |
| L2 | 065_sparse_expert_dispatch_and_combine | torch | 16/16 | n/a | n/a | fast |
| L2 | 081_moe_sparse_expert_dispatch | torch | 16/16 | n/a | n/a | fast |
| L2 | 082_moe_layer_complete_forward_with_residual | torch | 16/16 | n/a | n/a | fast |
| Quant | 002_fp8_attention_qkv_projection | flashinfer | 16/16 | 4.27x | 1.01–6.74x | full |
| Quant | 003_fp8_mlp_gate_up_projection | flashinfer | 16/16 | 22.56x | 19.48–23.80x | full |
| Quant | 004_fp8_moe_expert_linear | flashinfer | 16/16 | 3.23x | 1.64–4.99x | full |
| Quant | 005_fp8_moe_router_projection | flashinfer | 16/16 | 8.54x | 3.23–14.55x | full |
| Quant | 011_fp8_moe_gate_routing | flashinfer | 16/16 | 1.44x | 0.69–2.15x | full |
| Quant | 012_fp8_shared_expert_mlp | flashinfer | 16/16 | 4.11x | 1.76–5.24x | full |
| Quant | 013_fp8_mla_kv_compression_projection | flashinfer | 16/16 | 4.26x | 1.56–6.41x | full |
| Quant | 015_fp8_mla_attention_output_projection | flashinfer | 16/16 | 29.83x | 24.79–34.90x | full |
| Quant | 016_fp8_multi_latent_attention_qkv_projection | flashinfer | 16/16 | 3.93x | 1.71–5.88x | full |

**Geometric mean speedup over reference:** 3.66x (31 timed)

## Notes

- **Correctness:** 44/44 baselines pass all workloads on B200 (SM100).
- **`fast` bench = correctness only** (5 warmup, 10 iter, `benchmark_reference=false`). Used for tasks where the reference is a Python loop that exceeds the 100-iter timeout: all L2 MoE variants + FIB attention wrappers + L1_044 MoE. Speedup for these was manually spot-timed during development (see PR notes) but is not part of the aggregate.
- **`full` bench** = 25 warmup, 100 iter, `benchmark_reference=true` (CUPTI-timed).
- **`FLASHINFER_TRACE_DIR`** must be set (to an absolute path) for FIB/013, 017, 018, 019 — those workloads reference `kv_indptr`/`kv_indices` blobs under `data/flashinfer-trace/`. The aggregator sets it automatically.
- **Geometric mean excludes tasks without a full-bench speedup.** Actual mean across the entire 44-task set (with manual timings for the "fast" tasks) is closer to 3.2–3.5x depending on how you weight the MoE tasks.
