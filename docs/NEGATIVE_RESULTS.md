# Negative Results: When SOTA Libraries Don't Help

This document records the cases where we tried to use SOTA libraries to accelerate "naive-looking" reference implementations but found they were either equal to or slower than the reference. **These are valuable signals showing that the reference is already optimal for sol-execbench's workload sizes.**

## Key Lesson

> A Python for-loop in the reference doesn't mean the reference is slow. PyTorch's optimized matmul/SDPA per iteration is often faster than fused alternatives in the workload size range.

## Case Studies

### Case 1: Sparse MoE with SegmentGEMM (L2_008)

**Reference pattern**: Python `for expert_idx in range(num_experts)` loop, processing each expert's tokens separately.

**Hypothesis**: FlashInfer `SegmentGEMMWrapper` should be faster by doing all experts in one fused kernel.

**Implementation**:
```python
# Sort tokens by expert
sort_order = torch.argsort(flat_experts)
sorted_hidden = hidden_flat[sorted_token_indices]
seg_lens = torch.bincount(sorted_experts, minlength=num_experts)
seg_indptr = torch.cat([zeros(1), seg_lens.cumsum(0)]).to(int32)

# Grouped GEMM for all experts
gate_out = seg_gemm.run(sorted_hidden, expert_gate_proj, num_experts, True, seg_indptr=seg_indptr)
up_out = seg_gemm.run(sorted_hidden, expert_up_proj, num_experts, True, seg_indptr=seg_indptr)
intermediate = (gate_out * torch.sigmoid(gate_out)) * up_out
expert_out = seg_gemm.run(intermediate, expert_down_proj, num_experts, True, seg_indptr=seg_indptr)

# Scatter back
output = torch.zeros_like(hidden_flat)
output.index_add_(0, sorted_token_indices, expert_out * sorted_weights[:, None])
```

**Result**: 0.67x (33% slower than reference)

| Workload | Tokens | Reference (Python loop) | SegmentGEMM | Speedup |
|---|---|---|---|---|
| (1, 128) | 128 | 20.1ms | 15.1ms | 1.3x ✓ |
| (1, 512) | 512 | 19.8ms | 16.5ms | 1.2x ✓ |
| (1, 2048) | 2048 | 21.7ms | 25.0ms | 0.9x ✗ |
| (4, 512) | 2048 | 20.6ms | 25.0ms | 0.8x ✗ |
| (8, 1024) | 8192 | 23.8ms | 77.8ms | 0.3x ✗ |

**Why it failed**: 
- 128 experts × top-8 means each expert gets ~64 tokens average (with batch=1 seq=2048)
- PyTorch's per-expert matmul (cuBLAS) is highly optimized
- SegmentGEMM's overhead (sorting, indirection, suboptimal kernel for short segments) exceeds its benefit
- The Python for-loop overhead is amortized across the GEMM work

### Case 2: Variable-length Vision Attention (L1_021) — Marginal Win

**Reference pattern**: Python for-loop processing each sequence separately with `.item()` calls (CPU sync).

**Implementation**: Block-diagonal mask + PyTorch SDPA in one call.

**Result**: 1.11x speedup (marginal, just enough to include)

**Why it's marginal**:
- The for-loop iterations are heavy (each does Q@K^T and attn@V)
- Removing the loop helps, but building O(seq_len²) mask adds overhead
- Net win is small for the workload sizes (avg 1500 tokens / 8 sequences)

### Case 3: Composition Baselines for Decoder Layers (L2_002, L2_019, L2_063)

**Hypothesis**: Combining torch native ops should beat the reference's similar torch implementation.

**Result**: All ~1.0x (equivalent to reference)

**Why it failed**: The reference itself uses `F.linear` + `F.softmax` + `F.silu` — these already call cuBLAS/cuDNN. Wrapping with `F.rms_norm` and `F.sdpa` saves only marginal launch overhead because the compute is matmul-bound.

### Case 4: Liger GroupNorm vs torch (L1_078)

**Hypothesis**: Liger's Triton GroupNorm should beat torch.

**Result**: 0.66x (Liger is 50% slower)

**Why it failed**: PyTorch's GroupNorm internally calls cuDNN's optimized implementation. Liger's Triton kernel is competitive on some shapes but loses on the specific shapes in this task.

### Case 5: Liger GEGLU on small tensors (L1_085)

**Hypothesis**: Liger's fused GEGLU should beat `gelu_and_mul`.

**Result**: 0.78x (Liger is 22% slower)

**Why it failed**: For pure activation (no linear), the tensor is small enough that PyTorch's elementwise kernels + JIT optimization handle it efficiently. Liger's Triton kernel launch overhead dominates.

## What This Tells Us

1. **PyTorch is already SOTA for most simple ops**: matmul (cuBLAS), conv2d (cuDNN), elementwise (codegen), softmax (codegen). Don't replace these.

2. **Python for-loops aren't automatically slow**: When each iteration does substantial GPU work, the loop overhead is amortized. The Sparse MoE case showed PyTorch's per-expert matmul beating fused SegmentGEMM.

3. **Fused kernels have setup overhead**: Sorting, building indirection arrays, segment tables — these add cost that must be repaid by faster computation. For mid-range workload sizes, the math often doesn't work out.

4. **SOTA wins are concentrated in specific patterns**:
   - **Normalization** (RMSNorm/LayerNorm): FlashInfer is 7-14x faster than naive variance computation
   - **Long-sequence attention**: FlashAttention is 1.5-7x faster than torch matmul-based attention
   - **Highly batched MLP**: Liger's fused GEMM+activation wins on large tensors

5. **Always benchmark before claiming a speedup**: Even theoretically-superior algorithms can lose on specific workload shapes.

## Removal Policy

This repository's policy: **only include baselines that beat the reference in average across the task's official workloads** (>1.0x speedup). Marginal wins (1.0-1.1x) are kept; equivalent or slower (<1.0x) are removed.

Removed baselines (with measured slowdown):
- L1_038 multi_head_rmsnorm (torch native): 0.91x — reference is similar
- L1_078 group_norm_fusion (Liger): 0.66x
- L1_085 geglu_activation (Liger): 0.78x  
- L2_002 decoder_layer_full_block (torch native): 1.00x
- L2_008 moe_sparse_routing (FlashInfer SegmentGEMM): 0.67x — see Case 1
- L2_019 decoder_qwen2vl (torch native): 0.98x
- L2_053 text_decoder_layer (FlashAttn composition): 0.89x
- L2_063 encoder_dual_norm (torch native): 0.96x

This isn't failure — it's data showing that **sol-execbench's reference quality is high** and PyTorch's built-in ops are SOTA for most patterns.