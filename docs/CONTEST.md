# Contest baselines

Baselines for the Contest evaluation set under `data/benchmark/Contest/`. Targets DeepSeek-V3 / Qwen3-VL / Kimi-K2 / Gemma-3 / Nemotron / gpt-oss / GLM-4 / Reka / Olmo-3 / Ring / Ling / Llama-3.2-Vision patterns — FP8 blockwise, MLA, large-scale MoE, ultralong attention, multimodal 3D RoPE.

## Status

**44 verified baselines** across 4 categories (L1, L2, Quant, FlashInfer-Bench). Run `scripts/aggregate_contest.py` to regenerate `docs/CONTEST_RESULTS.md` with the full speedup table.

## Environment used to write & validate

- **GPU**: NVIDIA B200 (SM 100, Blackwell)
- **CUDA**: 13.0
- **PyTorch**: 2.9.0+cu130
- **FlashInfer**: 0.6.12
- **flash_attn**: 2.8.3

Not in env (would expand coverage): DeepGEMM, FlashMLA, TransformerEngine, sgl-kernel, vLLM, FlashAttention 3.

## How to run

```bash
cd sol-execbench
# Clone sol-baseline into ./baselines/ first, then:
export FLASHINFER_TRACE_DIR=data/flashinfer-trace   # required for FIB/013, 017, 018, 019

# One task
uv run sol-execbench data/benchmark/Contest/Quant/002_fp8_attention_qkv_projection \
    --solution baselines/baselines/flashinfer/Contest/Contest_Quant_002_fp8_qkv/solution.json \
    --config baselines/scripts/bench_config.json --json

# All 44 tasks → docs/CONTEST_RESULTS.md
python ../sol-baseline/scripts/aggregate_contest.py --output ../sol-baseline/docs/CONTEST_RESULTS.md
```

## High-value patterns established

### FP8 blockwise (1×128 act / 128×128 weight) GEMM — DeepSeek/Qwen recipe

```python
import flashinfer.gemm as fg
out = fg.gemm_fp8_nt_groupwise(
    qx, qw, sx, sw,                            # qx, qw fp8_e4m3; sx, sw float32
    scale_granularity_mnk=(1, 128, 128),
    scale_major_mode='MN',
    out_dtype=torch.bfloat16,
)
```

With `scale_major_mode='MN'`: `sx` is `(K//128, M)`, `sw` is `(K//128, N//128)` — both `.t().contiguous()` from natural compute order.

Covered: Quant/002, 003, 004, 005, 011, 012, 013, 015, 016.

### Sort-by-expert + per-expert mm for MoE

When `flashinfer.fused_moe.cutlass_fused_moe` doesn't work on B200, the workhorse pattern is:

```python
flat_e = selected_experts.reshape(-1)
sort_idx = torch.argsort(flat_e)
s_e, s_tok, s_w = flat_e[sort_idx], flat_tok[sort_idx], flat_w[sort_idx]
counts = torch.bincount(s_e, minlength=E)
offsets = torch.cat([torch.zeros(1, …), counts.cumsum(0)])
x_sorted = hidden_states.index_select(0, s_tok)
for e in range(E):
    n = counts[e].item()
    if n == 0: continue
    x = x_sorted[off:off+n]
    expert_out[off:off+n] = down(F.silu(gate(x)) * up(x))
output.index_add_(0, s_tok, expert_out * s_w.unsqueeze(-1))
```

Beats the reference's `F.one_hot + permute + where` scan over all experts by ~2x because empty experts are skipped naturally and the inner mm hits cuBLAS directly. Covered: L1/044, L2/008, 009, 010, 013, 029, 065, 081, 082.

### Adaptive cu_seqlens vision attention (fp32)

For variable-length vision attention, switch strategy by total token count:
- `total_seq_len < 2500`: build block-diagonal mask, one `F.scaled_dot_product_attention` call
- `total_seq_len ≥ 2500`: per-sequence SDPA loop (mask memory would be O(N²))

Covered: L1/021. Pattern ported from sol-baseline upstream.

### FlashInfer attention wrappers (paged decode / ragged prefill / MLA)

Single-call replacement of per-batch Python attention loops, returning base-2 LSE (matches the FIB reference's `logsumexp / log(2)` convention — do **not** divide again):

```python
w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, 'NHD')
w.plan(indptr=kv_indptr, indices=kv_indices, last_page_len=torch.ones(B, ...), ...)
out, lse = w.run(q, (k_cache, v_cache), return_lse=True)
```

Covered: FIB/013, 017, 018, 019.

## Negative results & failure modes

See [`CONTEST_NEGATIVE_RESULTS.md`](CONTEST_NEGATIVE_RESULTS.md) — 14 documented cases where the SOTA doesn't beat reference on B200 (most often: fp32 reference is already F.rms_norm+cuDNN SDPA optimal, or FlashInfer MoE/GEMM paths aren't compiled for sm_100 in 0.6.12).

## Library compatibility matrix

See [`library_index.json`](library_index.json) — dtype × shape × SM constraints for every kernel used. Check this BEFORE writing a new wrapper.

## Tools

- `scripts/bench_config.json` — full eval (`benchmark_reference=true`, 25 warmup / 100 iter)
- `scripts/bench_config_fast.json` — correctness-only quick check (5/10)
- `scripts/bench_config_quick.json` — moderate (5/20) with reference timing for slow refs
- `scripts/aggregate_contest.py` — discovers all baselines, runs each with appropriate config, writes Markdown
- `scripts/run_contest.sh` — bash alternative covering specific tasks (legacy)
- `scripts/verify.py` / `scripts/benchmark.py` — original sol-baseline tools extended for `Contest/` category
