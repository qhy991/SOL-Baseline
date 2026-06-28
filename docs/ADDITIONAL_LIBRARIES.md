# Additional SOTA Libraries Research

本文档调研更多 SOTA GPU kernel 库，分析它们能否覆盖当前 156 个未匹配的 task。

## 核心结论

**部分未匹配的 task 确实可以通过引入更多 SOTA 库来覆盖。** 主要发现如下：

| 库 | 类别 | 可覆盖 task 估计 |
|---|---|---|
| **flash-linear-attention (fla)** | Linear attention / Mamba2 / RWKV | ~10 |
| **NATTEN** | Vision 2D/3D attention | ~5 |
| **TransformerEngine** (full) | FP8 attention/MoE/MLP | ~30 (Quant tasks) |
| **cuDNN Frontend** | Fused Norm+Activation, Grouped GEMM+SwiGLU | ~5 |
| **vllm-flash-attn (Triton backend)** | FP8 attention, paged attention, RoPE | ~5 |
| **MegaBlocks** | Dropless MoE | ~10 |
| **pytorch_scatter / FBGEMM** | scatter / index_add / embedding | ~5 |
| **xformers** | Sparse/block-sparse attention, fused SwiGLU | ~3 |

**理论新增覆盖：~70 个 task**。从 16/235 (6.8%) 提升到 ~86/235 (~37%)。

---

## 一、Linear Attention / SSM 类库

### 1.1 flash-linear-attention (fla) — 高度推荐

**[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)**

提供了大量 linear attention / SSM 变体的高性能 Triton kernel：

| Task | 当前状态 | fla 库 |
|---|---|---|
| L2_060 chunk_gated_delta_rule_linear_attention | 无法匹配 | **`fla.ops.gated_delta_rule.chunk_gated_delta_rule`** — 直接对应 |
| L1_094 time_decay_exponential_stabilization (RWKV) | 架构特定 | **`fla.ops.rwkv6/7`** — RWKV 完整实现 |
| L1_070 mamba2_intra_chunk_diagonal | API 差异 | **`fla/models/mamba2`** — chunk scan |
| L2_043 mamba_chunk_scan_with_segsum | API 差异 | **`fla.ops.mamba2`** — Mamba2 chunk scan |
| L2_044 mamba_discretization_and_segsum | API 差异 | 同上 |
| L2_058 mamba2_selective_scan | API 差异 | 同上 |
| L1_005 conv_gated_projection (Hyena-like) | 已有 causal-conv1d | fla 也有支持 |
| L1_006 hyena_depthwise_conv1d_split_gate | 无 baseline | fla Hyena 实现 |
| L1_051 hyena_complete_forward_block | 无 baseline | fla Hyena 实现 |
| L1_052 hyena_gating_and_output_projection | 无 baseline | fla Hyena 实现 |

**潜在新增覆盖：~10 个 task**

### 1.2 mamba-ssm 高级 API

之前我们用了 mamba-ssm 的低层 API（`selective_scan_fn`、`causal_conv1d_fn`），但 mamba-ssm 还提供 `Mamba2`、`Mamba` 完整 nn.Module。可以在 baseline 中实例化 `Mamba2` 模块来跑 task，绕过分解式 API 差异。

---

## 二、Vision Attention 类库

### 2.1 NATTEN (Neighborhood Attention)

**[SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN)**

提供 2D/3D 邻域注意力的 fused kernel，专为 vision/video 模型设计：

| Task | 当前状态 | NATTEN |
|---|---|---|
| L1_019 vision_3d_rotary_embedding_with_spatial_merge | 无法匹配 (backward) | NATTEN 3D 注意力 |
| L1_020 vision_patch_merger_spatial_shuffle_mlp | 无 baseline | NATTEN 邻域操作 |
| L1_021 vision_cu_seqlens_variable_length_attention | 融合度过高 | natten.na2d |
| L1_027 video_spatial_attention_with_rope_3d | 融合度过高 | natten.na3d |
| L1_089 vae_attention_block_with_groupnorm | 融合度过高 | natten.na2d |
| L2_017 fused_vision_cu_seqlens_attention_with_2d_rope | backward | NATTEN 2D RoPE attn |
| L2_018 cu_seqlens_variable_length_vision_attention | 融合度过高 | natten.na2d |

**潜在新增覆盖：~5 个 task**

### 2.2 vllm-flash-attn (Triton 后端)

**[vllm-project/flash-attention](https://github.com/vllm-project/flash-attention)**

Triton-based FlashAttention，相比官方 FlashAttention 2.8.3 有以下优势：
- **支持 FP8 attention**（通过 FA v3 接口）
- **rotary embeddings 内置**（不需要 baseline 自己做 RoPE）
- **paged attention**
- **MQA/GQA、varlen**
- **更好的 CUDA 13 + SM100 兼容性**

| Task | 当前状态 | vllm-flash-attn |
|---|---|---|
| L1_018 fused_rope_with_qk_norm_and_kv_cache_update | 融合度过高 | flash_attn_with_kvcache + RoPE |
| L1_067 flash_attention_gqa_ultralong | CUDA 13 不兼容 | Triton 后端可能兼容 |
| L1_071 kv_cache_update_with_rope | 融合度过高 | flash_attn_with_kvcache |
| L1_018, L1_062 等 attention 类 | 融合度过高 | RoPE + attention 一体 |

**潜在新增覆盖：~5 个 task**

---

## 三、Quantization 类库

### 3.1 TransformerEngine (完整安装)

**[NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine)**

之前我们只装了 meta-package，没有 PyTorch extension。完整安装后提供：
- **FP8 attention**（fused QKV + attention + output projection）
- **FP8 LayerNorm/RMSNorm**
- **FP8 GEMM**（自动 scaling）
- **MoE FP8 支持**

| Task | 当前状态 | TransformerEngine |
|---|---|---|
| Quant/001-007 fp8_attention | FP8 数据类型问题 | TE Linear FP8 |
| Quant/008-017 fp8_mlp/moe | FP8 数据类型问题 | TE LayerNormMLP FP8 |
| Quant/018-033 nvfp4_* | NVFP4 数据类型问题 | TE NVFP4 支持 |

**潜在新增覆盖：~30 个 task**（如果 eval driver 能生成正确的 FP8 量化数据）

完整安装命令：
```bash
pip install transformer-engine[pytorch] --no-build-isolation
```

### 3.2 NVIDIA cuDNN Frontend

**[NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend)**

提供大量 fused kernel：
- **Fused RMSNorm + SiLU**
- **GEMM + SwiGLU / sReLU / dsReLU / Amax**
- **Grouped GEMM + GLU / Hadamard / Quant**
- **NSA (Native Sparse Attention)**
- **DSA/CSA**（DeepSeek 注意力）

| Task | 当前状态 | cuDNN Frontend |
|---|---|---|
| L2_004 fused_residual_rms_mlp | 融合度过高 | Fused RMSNorm + SwiGLU |
| L1_074 fused_gated_mlp_silu | Liger 精度问题 | GEMM + SwiGLU |
| L1_048 fused_gate_up_projection_with_swiglu | 已用 Liger | cuDNN 可能更快 |
| L2_008+ MoE | 调度操作 | Grouped GEMM + GLU |
| L1_038 flux_multi_head_rmsnorm_qk | per-head weight | 可能支持 |

**潜在新增覆盖：~5 个 task**

---

## 四、MoE 类库

### 4.1 MegaBlocks

**[databricks/megablocks](https://github.com/databricks/megablocks)**

dropless MoE 库，提供：
- **dMoE 层**（block-sparse MoE）
- **Grouped GEMM**（基于 grouped_gemm 库）

| Task | 当前状态 | MegaBlocks |
|---|---|---|
| L1_044 moe_expert_computation | 多专家 SwiGLU | dMoE.forward |
| L1_076 batched_expert_forward | 调度 | dMoE.forward |
| L2_008/011/029 moe_sparse_routing_and_dispatch | 调度 | dMoE + routing |
| L2_010/012/013 moe_expert_computation | 调度 | dMoE |
| L2_016 moe_expert_mlp_with_load_balancing | 调度 | MegaBlocks load balancing |
| L2_024-026 moe_expert_parallel_execution | 调度 | dMoE parallel |

**潜在新增覆盖：~10 个 task**

### 4.2 tgale96/grouped_gemm

**[tgale96/grouped_gemm](https://github.com/tgale96/grouped_gemm)**

CUTLASS-based 分组 GEMM 库，可以作为 MoE expert linear 的 baseline。

---

## 五、Scatter / Embedding 类库

### 5.1 pytorch_scatter

**[rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)**

提供 scatter / segment_coo / segment_csr 等 sparse 操作的 CUDA kernel：

| Task | 当前状态 | pytorch_scatter |
|---|---|---|
| L1_008 expert_output_weighted_index_add | torch 已优化 | scatter_add 可能更快 |
| L1_009 expert_token_scatter (backward) | 调度+backward | - |
| L1_058 moe_expert_token_radix_sort | 排序 | - |

**潜在新增覆盖：~2 个 task**

### 5.2 FBGEMM_GPU

**[pytorch/FBGEMM](https://github.com/pytorch/FBGEMM)**

提供：
- Table batched embedding
- Jagged tensor 操作
- FP8 row-wise quantization
- 推荐系统 GPU 算子

主要用于推荐系统，与 sol-execbench LLM 类 task 重合度较低。

---

## 六、xformers

**[facebookresearch/xformers](https://github.com/facebookresearch/xformers)**

提供：
- **Memory-efficient attention**（10x faster）
- **Sparse attention** / block-sparse attention
- **Fused softmax / linear / LayerNorm / SwiGLU**
- **Fused dropout(activation(x+bias))**

| Task | 当前状态 | xformers |
|---|---|---|
| L1_074 fused_gated_mlp_silu | Liger 精度问题 | xformers fused SwiGLU |
| L1_021 vision_cu_seqlens_variable_length_attention | 融合度过高 | memory_efficient_attention 支持 varlen |
| L1_046 attention_softmax_with_softcapping | 简单 | fused_softmax |

**潜在新增覆盖：~3 个 task**

---

## 七、PyTorch 内置 SDPA

`torch.nn.functional.scaled_dot_product_attention` 在 PyTorch 2.x 中已支持多个后端：
- **FlashAttention 后端**（自动选择）
- **mem_efficient 后端**
- **cuDNN 后端**

| Task | 当前状态 | PyTorch SDPA |
|---|---|---|
| L1_046 attention_softmax_with_softcapping | 简单 | F.scaled_dot_product_attention 内置 |
| L1_031 repeat_kv_attention_matmul | torch.matmul | F.scaled_dot_product_attention 自动 |
| L1_049 attention_qk_matmul_with_gqa_repeat | torch.matmul | F.scaled_dot_product_attention(enable_gqa=True) |

可以作为对比的另一个"SOTA baseline"，但不算引入新库。

---

## 八、Mojo / Triton kernel 库

### 8.1 unsloth

**[unslothai/unsloth](https://github.com/unslothai/unsloth)**

提供高度优化的训练 kernel（GROK、Llama、Phi 等），有 RoPE、RMSNorm、SwiGLU 的 Triton 实现。

### 8.2 ThunderKittens

**[HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)**

CUDA tile primitives，提供：
- Linear attention 实现
- Mamba2 SSD
- 高效的 attention 变体

---

## 九、推荐的下一步

**优先级排序：**

1. **flash-linear-attention (fla)** — 直接对应 L2_060、Mamba2 系列 task（~10 个）
2. **NATTEN** — vision attention 类（~5 个）
3. **vllm-flash-attn (Triton 后端)** — 解决 FlashAttention CUDA 13 兼容性 + RoPE + KV cache（~5 个）
4. **MegaBlocks** — MoE task（~10 个）
5. **cuDNN Frontend** — fused norm+activation 类（~5 个）
6. **TransformerEngine 完整安装** — Quant FP8 task（~30 个，但需 eval driver 支持 FP8 数据）

**预期总覆盖：从 16 提升到 ~60-86 个 task**。

---

## 十、根本限制（无法靠引入更多库解决）

即使引入所有上述库，以下类别仍然无法匹配：

1. **Backward task**（~30 个）：SOTA 库一般不提供 backward
2. **真正自定义的架构操作**（~5 个）：AltUp、特定 Flux 操作
3. **纯 matmul / 简单激活**（~20 个）：torch 已是 SOTA
4. **MoE 路由/调度逻辑**（部分）：算法层面的调度，非 GEMM
5. **eval driver 数据生成限制**：FP8/NVFP4 task 需要量化数据生成支持

**真实上限：约 80-100 个 task 可以匹配 SOTA baseline（占总数 34-43%）。**

---

## 参考来源

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [NATTEN](https://github.com/SHI-Labs/NATTEN)
- [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
- [NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend)
- [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention)
- [databricks/megablocks](https://github.com/databricks/megablocks)
- [tgale96/grouped_gemm](https://github.com/tgale96/grouped_gemm)
- [facebookresearch/xformers](https://github.com/facebookresearch/xformers)
- [rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)
- [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- [unslothai/unsloth](https://github.com/unslothai/unsloth)