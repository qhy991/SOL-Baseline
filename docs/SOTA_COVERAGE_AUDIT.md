# Community SOTA Coverage Audit

Generated: 2026-07-01 03:43:20 UTC
Target Python: `/home/qinhaiyan/sol-execbench/.venv/bin/python`

This audit checks whether candidate community SOTA libraries are actually importable in the target benchmark environment. It does not claim a library is faster; `available` means the next step is to write and benchmark a task-specific wrapper.

## Extra Source Paths

| Key | Path |
|---|---|
| `sgl_kernel` | `/home/qinhaiyan/sglang` |

## Result Matrix

| Priority | Library | Status | Probe Result | Contest re-attempt targets |
|---|---|---|---|---|
| P0 | DeepGEMM | **available** | import succeeded | FIB/005 fp16 GEMM router shape<br>FIB/020 FP8 routed MoE<br>L2/012 padded MoE grouped GEMM<br>L2/048 fp32 MoE dispatch if precision can be preserved<br>Quant/011 FP8 router projection |
| P0 | TransformerEngine | **missing** | not installed | Quant FP8 projection tasks with non-FlashInfer recipes<br>L2/019 fp32 decoder if an FP8 path is acceptable under tolerance<br>L2/048 fp32 MoE dispatch if precision can be preserved |
| P0 | FlashAttention 3 / Hopper-Blackwell attention path | **missing** | not installed | L1/067 fp32 ultralong GQA attention |
| P1 | sgl-kernel / SGLang custom ops | **missing** | not installed | L1/018 fused RoPE + QK norm + KV update<br>L1/023 multimodal RoPE position work<br>FIB/020 alternative FP8 MoE paths |
| P1 | vLLM custom ops / vllm-flash-attn | **missing** | not installed | L1/067 fp32 or long-context attention re-test<br>L1/071 KV cache update + RoPE<br>FIB attention wrappers as alternate implementation |
| P1 | MegaBlocks / grouped_gemm | **missing** | not installed | L1/044 MoE expert compute<br>L1/076 dense batched expert forward<br>L2/008/L2/010/L2/012/L2/013/L2/029 MoE variants |
| P2 | flash-linear-attention | **missing** | not installed | Not part of the 60-task Contest set; relevant to wider sol-execbench coverage. |
| P2 | NATTEN | **missing** | not installed | Potential L1/L2 vision attention variants outside current shipped Contest gaps. |
| P2 | xFormers | **missing** | not installed | L1/046 softcap/softmax sanity check<br>L1/021 varlen attention sanity check |
| P3 | AITER | **missing** | not installed | None for B200 CUDA environment unless a CUDA backend is present. |

## Details

### DeepGEMM

- Category: FP8/BF16 GEMM and MoE
- Status: available (import succeeded)
- Probe detail: `deep_gemm (2.5.0) @ /home/qinhaiyan/sol-execbench/.venv/lib/python3.12/site-packages/deep_gemm/__init__.py`
- Expected value: Best direct missing candidate for GEMM/MoE negatives.
- Blocking notes: Requires compiled deep_gemm._C extension. On SM100, FP8 scale layout may need packed UE8M0 conversion, so importability is necessary but not sufficient.

### TransformerEngine

- Category: FP8 layers
- Status: missing (not installed)
- Probe detail: `transformer_engine.pytorch: ModuleNotFoundError: No module named 'transformer_engine'`
- Expected value: Main NVIDIA library for FP8 Linear/LayerNorm/MLP experiments.
- Blocking notes: Full PyTorch extension is required; the meta package alone is not enough.

### FlashAttention 3 / Hopper-Blackwell attention path

- Category: Attention
- Status: missing (not installed)
- Probe detail: `no candidate module found on sys.path`
- Expected value: Only credible near-term path for re-testing fp32 ultralong attention.
- Blocking notes: FA2 imports are not enough; fp32 support must be verified by an actual kernel call.

### sgl-kernel / SGLang custom ops

- Category: Serving custom ops
- Status: missing (not installed)
- Probe detail: `sglang.srt._custom_ops: ModuleNotFoundError: No module named 'sglang'`
- Expected value: Useful for fused serving ops that FlashInfer 0.6.12 cannot cover on SM100.
- Blocking notes: APIs are less stable than FlashInfer; wrappers need per-op compatibility checks.

### vLLM custom ops / vllm-flash-attn

- Category: Serving attention and MoE
- Status: missing (not installed)
- Probe detail: `vllm._C: ModuleNotFoundError: No module named 'vllm'`
- Expected value: Independent serving stack for paged attention and custom CUDA ops.
- Blocking notes: Large dependency surface; importability often depends on exact torch/CUDA build.

### MegaBlocks / grouped_gemm

- Category: MoE
- Status: missing (not installed)
- Probe detail: `no candidate module found on sys.path`
- Expected value: Potential grouped-GEMM alternative to per-expert torch.mm loops.
- Blocking notes: Most useful when token routing layout matches dropless/block-sparse assumptions.

### flash-linear-attention

- Category: Linear attention / SSM
- Status: missing (not installed)
- Probe detail: `no candidate module found on sys.path`
- Expected value: Important for non-Contest Mamba2/RWKV/linear-attention tasks.
- Blocking notes: Do not count as Contest coverage until a matching Contest task exists.

### NATTEN

- Category: Vision attention
- Status: missing (not installed)
- Probe detail: `no candidate module found on sys.path`
- Expected value: Worth tracking for wider vision attention coverage.
- Blocking notes: Neighborhood attention is not a drop-in replacement for arbitrary cu_seqlens attention.

### xFormers

- Category: Attention and fused activations
- Status: missing (not installed)
- Probe detail: `no candidate module found on sys.path`
- Expected value: Useful as an independent comparison for attention/activation negatives.
- Blocking notes: Often behind PyTorch SDPA/FlashAttention on modern NVIDIA inference shapes.

### AITER

- Category: ROCm-oriented kernels
- Status: missing (not installed)
- Probe detail: `no candidate module found on sys.path`
- Expected value: Track explicitly so CUDA/B200 reviewers do not assume it was forgotten.
- Blocking notes: Generally ROCm-focused; low expected value for NVIDIA B200 Contest runs.

## Reviewer Conclusion

- Available candidates: 1
- Blocked candidates: 0
- Missing candidates: 9
- A claim that all community SOTA implementations have been exhausted is valid only when every P0/P1 candidate is either benchmarked or has a recorded task-specific blocker.
