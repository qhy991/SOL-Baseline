# Adaptive Baselines: Workload-Aware Library Selection

> **Key insight from this work**: Different workload sizes have different optimal libraries. A single "best library" claim is usually wrong — the right baseline switches implementation based on input shape.

## The Problem with Single-Library Baselines

When we benchmark a single SOTA library across all of a task's workloads, we often see:

| Workload size | Library A speedup | Library B speedup |
|---|---|---|
| Small (< threshold) | 2.0x ✓ | 0.5x ✗ |
| Medium | 1.2x ✓ | 1.0x = |
| Large (> threshold) | 0.6x ✗ | 1.5x ✓ |

Picking either A or B gives a poor average. The right answer is **both**, with a heuristic switch.

## Real Example: L1_021 vision_cu_seqlens_attention

Per-workload measurement (float32 variable-length attention with 16 head, hidden=1280):

| Workload (total_seq_len) | Reference (Python loop) | Block-diag mask SDPA | Speedup |
|---|---|---|---|
| 128 | 0.281ms | 0.240ms | 1.17x |
| 211 | 0.538ms | 0.327ms | **1.65x** |
| 293 | 0.489ms | 0.263ms | **1.86x** |
| 449 | 0.643ms | 0.305ms | **2.11x** |
| 512 | 0.543ms | 0.289ms | **1.88x** |
| 691 | 0.776ms | 0.441ms | **1.76x** |
| 768 | 0.907ms | 0.436ms | **2.08x** |
| 997 | 1.104ms | 0.594ms | **1.86x** |
| 1152 | 1.290ms | 0.610ms | **2.12x** |
| 1571 | 1.459ms | 1.018ms | 1.43x |
| 1721 | 1.576ms | 1.073ms | 1.47x |
| 2304 | 1.972ms | 1.614ms | 1.22x |
| **2851** | 2.380ms | 2.430ms | 0.98x |
| **3072** | 2.373ms | 2.796ms | 0.85x |
| **4608** | 3.277ms | 5.211ms | **0.63x** ✗ |

There's a clear inflection point around **total_seq_len = 2500**:
- Below: block-diag mask + single SDPA call wins (no Python loop overhead)
- Above: reference's per-sequence loop wins (avoids O(N²) mask memory)

## Solution: Adaptive Baseline

```python
@torch.no_grad()
def run(hidden_states, cu_seqlens, ...):
    total_seq_len = hidden_states.shape[0]
    
    # ... shared preprocessing (QKV, RoPE) ...
    
    if total_seq_len < 2500:
        # Short: use block-diag mask + single SDPA call
        mask = build_block_diag_mask(cu_seqlens, total_seq_len)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling)
    else:
        # Long: per-sequence SDPA loop (matches reference but uses fast SDPA)
        attn_outputs = []
        for i in range(num_seqs - 1):
            s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
            out = F.scaled_dot_product_attention(q[:,:,s:e], k[:,:,s:e], v[:,:,s:e], scale=scaling)
            attn_outputs.append(out.transpose(1, 2))
        attn_output = torch.cat(attn_outputs, dim=1)
    
    return F.linear(attn_output, proj_weight, proj_bias)
```

**Result**: average 1.39x speedup (vs 1.11x for fixed-strategy block-diag, vs 0.91x for fixed-strategy nested tensor).

## When This Pattern Applies

Adaptive baselines help when:

1. **Fused-vs-per-iter tradeoff exists**: SegmentGEMM vs per-expert loop, block-diag mask vs per-sequence SDPA, etc.
2. **The setup cost is significant**: Sorting tokens, building masks, allocating workspace
3. **Workloads span a wide range**: 100x size variation is common in real benchmarks

## Other Patterns Where Adaptive Could Help (Untested)

| Task family | Fast for small | Fast for large |
|---|---|---|
| Sparse MoE | Python for-loop (per-expert matmul) | SegmentGEMM (batched grouped GEMM) |
| Variable-length attention | Block-diag mask SDPA | FlashAttention varlen |
| Cross-attention | torch SDPA + cuDNN | FlashAttention with KV cache |
| Many-expert MoE | Direct expert selection | Token-shuffled grouped GEMM |

## Practical Algorithm

For each task:

1. Benchmark several candidate implementations across all workloads
2. Build a table of `speedup[implementation][workload]`
3. Find the **decision boundary** (single threshold or 2D plane) that maximizes average speedup
4. Encode the heuristic as `if size < threshold: ... else: ...`

## Pitfalls

- **JIT compilation cost**: torch.compile-based approaches (FlexAttention) often lose because each new shape recompiles. Avoid for dynamic workloads.
- **Decision overhead**: The `if` check itself is cheap, but `tensor.item()` is a CPU-GPU sync. Use cached shape information from `tensor.shape`.
- **Workload bias**: The benchmark workloads may not match production. Document the threshold and consider re-tuning per deployment.

## Conclusion

Replacing "use library X" with "use X for small, Y for large" can double the realized speedup. The cost is slightly more code but much higher average performance. This is how real-world inference engines (vLLM, TensorRT-LLM) achieve their numbers — they dispatch to different kernel implementations based on shape heuristics, not a single best-of-all-worlds kernel.