# SOTA Baseline Coverage Analysis

本文档分析 sol-execbench 中 L1/L2/Quant 共 209 个 task 的 SOTA baseline 覆盖情况，记录哪些 task 已匹配、哪些无需匹配、哪些暂时无法匹配。

## 总览

| 类别 | 总数 | 已匹配 | 无需匹配 | 无法匹配 |
|---|---|---|---|---|
| FlashInfer-Bench | 26 | 9 | 0 | 17 |
| L1 | 94 | 7 | 45 | 42 |
| L2 | 82 | 0 | 18 | 64 |
| Quant | 33 | 0 | 0 | 33 |
| **合计** | **235** | **16** | **63** | **156** |

---

## 一、已匹配的 Task（16 个）

### 1.1 FlashInfer-Bench RMSNorm（9 个）

这些 task 是 FlashInfer 库的 benchmark 集合，reference 实现就是 FlashInfer API 的 torch 等价物，直接一对一匹配。

| Task | Library | 核心 API | Speedup |
|---|---|---|---|
| 001 fused_add_rmsnorm_h2048 | FlashInfer | `rmsnorm` | 7.3x |
| 002 fused_add_rmsnorm_h4096 | FlashInfer | `rmsnorm` | 8.5x |
| 003 fused_add_rmsnorm_h7168 | FlashInfer | `rmsnorm` | 8.7x |
| 021 rmsnorm_h128 | FlashInfer | `rmsnorm` | 9.3x |
| 022 rmsnorm_h512 | FlashInfer | `rmsnorm` | 7.0x |
| 023 rmsnorm_h1536 | FlashInfer | `rmsnorm` | 10.1x |
| 024 rmsnorm_h2048 | FlashInfer | `rmsnorm` | 12.1x |
| 025 rmsnorm_h4096 | FlashInfer | `rmsnorm` | 13.9x |
| 026 rmsnorm_h7168 | FlashInfer | `rmsnorm` | 14.3x |

### 1.2 L1 Tasks（7 个）

| Task | Library | 核心 API | Speedup |
|---|---|---|---|
| L1_033 post_norm_residual | FlashInfer | `rmsnorm` | 7.2x |
| L1_069 rms_norm | FlashInfer | `rmsnorm` | 6.7x |
| L1_048 fused_gate_up_projection | Liger | `LigerGELUMulFunction` | 2.2x |
| L1_085 geglu_activation | Liger | `LigerGELUMulFunction` | 0.7x |
| L1_078 group_norm_fusion | Liger | `LigerGroupNormFunction` | 0.3x |
| L1_005 conv_gated_projection | causal-conv1d | `causal_conv1d_fn` | 1.6x |
| L1_029 mamba_conv1d_with_gating | causal-conv1d | `causal_conv1d_fn` | 1.5x |

---

## 二、无需匹配的 Task（63 个）

这些 task 的核心计算已经被 torch 充分优化，没有 SOTA 库能提供更快的实现。

### 2.1 纯矩阵乘法（10 个）

这些 task 的核心就是 `torch.matmul`，torch 底层已调用 cuBLAS，没有更快的外部库。

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_003 lm_head_projection | `hidden @ weight.T` | torch.matmul → cuBLAS |
| L1_010 attention_value_projection | `hidden @ weight.T` + transpose | torch.matmul → cuBLAS |
| L1_030 attention_output_projection_with_residual | `attn @ weight.T` + residual | torch.matmul → cuBLAS |
| L1_031 repeat_kv_attention_matmul | `Q @ K^T` + GQA repeat | torch.matmul → cuBLAS |
| L1_032 attention_weights_matmul_with_value_projection | `weights @ V` + transpose | torch.matmul → cuBLAS |
| L1_049 attention_qk_matmul_with_gqa_repeat | `Q @ K^T` + scaling | torch.matmul → cuBLAS |
| L1_063 attention_output_reshape_and_projection | reshape + `attn @ weight.T` | torch.matmul → cuBLAS |
| L1_077 whisper_decoder_output_projection | `hidden @ weight.T` | torch.matmul → cuBLAS |
| L1_081 joint_attention_context_projection | `cat @ weight.T` | torch.matmul → cuBLAS |
| L1_083 attention_score_value_matmul | `weights @ V` | torch.matmul → cuBLAS |

### 2.2 纯激活函数（3 个）

这些 task 的核心是逐元素激活函数，torch 的 `F.gelu` / `F.silu` 已调用 cuDNN/cuBLAS，没有 SOTA 库提供专门的激活 kernel。

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_025 video_latent_gelu_activation | `GELU(x)` | `F.gelu` → cuDNN |
| L1_046 attention_softmax_with_softcapping | `softmax(x)` with softcapping | `F.softmax` → cuDNN |
| L1_053 gaussian_topk_sparse_activation | gaussian + top-k | 自定义算法，torch 已足够 |

### 2.3 纯 Softmax 操作（1 个）

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_046 attention_softmax_with_softcapping_and_dropout | softmax + softcapping | `F.softmax` → cuDNN |

### 2.4 纯三角函数/频率计算（3 个）

这些 task 计算 RoPE 的频率和 cos/sin 值，是纯数学运算，torch 已充分优化。

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_011 rotary_position_embedding | `inv_freq @ position_ids` → cos/sin | 纯数学运算 |
| L1_012 fused_cos_sin_embedding_generation | cos/sin from frequencies | 纯数学运算 |
| L1_016 rope_inverse_frequency_computation | `1.0 / (theta ^ (dim/hd))` | 纯数学运算 |

### 2.5 纯 Embedding 操作（1 个）

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_026 video_patch_embedding_projection | Conv3D → embedding | `F.conv3d` → cuDNN |

### 2.6 纯卷积操作（2 个）

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_040 conv2d_residual_block | Conv2D + residual | `F.conv2d` → cuDNN |
| L1_041 strided_conv2d_downsampling | Strided Conv2D | `F.conv2d` → cuDNN |

### 2.7 纯 MoE 调度/路由（5 个）

这些是 MoE 的路由、排序、负载均衡等调度操作，不涉及 GEMM 计算。

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_008 expert_output_weighted_index_add | index_add | torch 已优化 |
| L1_042 moe_expert_load_balancing | load balancing | 调度逻辑，非 GEMM |
| L1_058 moe_expert_token_radix_sort | radix sort + prefix sum | 排序操作 |
| L1_059 moe_group_score_aggregation | group score aggregation | 调度逻辑 |
| L1_093 grouped_topk_moe_routing | top-k routing | 调度逻辑 |

### 2.8 注意力掩码准备（1 个）

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_028 hybrid_attention_mask_preparation | causal mask generation | 纯 tensor 创建 |

### 2.9 自定义架构特定操作（4 个）

这些是特定模型架构的自定义操作，没有通用 SOTA 库。

| Task | 核心计算 | 原因 |
|---|---|---|
| L1_007 hyena_fft_size_padding_rfft | FFT padding | 架构特定 |
| L1_052 altup_hidden_state_collapse | AltUp stream collapse | 架构特定 |
| L1_065 fused_statistics_projection_and_split | Conv1D → mean/logvar | 架构特定 |
| L1_094 time_decay_exponential_stabilization | RWKV WKV decay | 架构特定 |

---

## 三、无法匹配的 Task（156 个）

### 3.1 Backward Pass（~30 个）

SOTA 库通常只提供前向 kernel，不提供 backward。以下 L1 task 全部是 backward：

001, 004, 009, 013, 014, 017, 019, 022, 024, 039, 042, 051, 056, 060, 061, 062, 066, 068, 072, 079, 084, 087, 090, 091, 093

L2 backward task 也类似。

### 3.2 融合度过高（~50 个）

这些 task 融合了多个不同的操作（QKV 投影 + RoPE + Attention + Output 投影），而 SOTA 库只提供其中单个操作。

**典型例子：**

| Task | 融合的操作 | 为什么无法匹配 |
|---|---|---|
| L1_015 GQA + RoPE + QK Norm | QKV 投影 + RMSNorm + RoPE + SDPA + Output | FlashAttention 只做 SDPA |
| L1_018 RoPE + QK Norm + KV Cache | RMSNorm + RoPE + KV Cache 写入 | 操作太多 |
| L1_047 Attention + QK Norm + RoPE | QKV 投影 + QK Norm + RoPE + SDPA | FlashAttention 只做 SDPA |
| L1_050 QKV Projection + Bias + Reshape | 3个 linear + reshape | 不是单一操作 |
| L1_067 flash_attention_gqa_ultralong | QKV 投影 + RoPE + SDPA + Output | FlashAttention 只做 SDPA（且 CUDA 13 兼容性问题） |
| L1_075 GQA Self-Attention + RoPE | QKV 投影 + RoPE + SDPA + Output | 同上 |

### 3.3 RoPE 实现差异（~10 个）

Liger 和 FlashInfer 的 RoPE 实现采用不同的频率布局（interleaved vs non-interleaved），与标准 cos/sin 方式不兼容。

| Task | 原因 |
|---|---|
| L1_088 rotary_position_embedding_application | Liger RoPE 与标准 cos/sin 不兼容（已验证，数值差异极大） |
| L1_011 rotary_position_embedding | 频率生成，非 RoPE 应用 |
| L1_023 multimodal_rope_position_computation | 多模态 3D RoPE，非标准实现 |
| L1_034 flux_multi_axis_rope | Flux 多轴 RoPE |
| L1_090 batched_2d_rope | 2D RoPE，非标准实现 |

### 3.4 非标准归一化（5 个）

| Task | 原因 |
|---|---|
| L1_035 flux_ada_layer_norm_zero | AdaLayerNormZero，不是标准 LayerNorm |
| L1_036 flux_output_norm_projection_chain | AdaLayerNormContinuous + projection |
| L1_038 flux_multi_head_rmsnorm_qk | Per-head RMSNorm，FlashInfer 只支持 1D weight |
| L1_080 adaptive_layernorm_continuous | 自适应 LayerNorm + modulation |
| L1_082 qk_norm_scaled_dot_product_attention | QK LayerNorm + SDPA，融合度过高 |

### 3.5 FlashInfer-Bench 需要外部数据（10 个）

这些 task 的 workload 中包含 safetensors 文件引用，需要从 HuggingFace 下载 `flashinfer-ai/flashinfer-trace` 数据集。

| Task | 类型 | 所需库 |
|---|---|---|
| 012-013 GQA paged decode | Attention | FlashInfer `BatchDecodeWithPagedKVCacheWrapper` |
| 014-015 GQA paged prefill | Attention | FlashInfer `BatchPrefillWithPagedKVCacheWrapper` |
| 016-017 GQA ragged prefill | Attention | FlashInfer `BatchPrefillWithRaggedKVCacheWrapper` |
| 018-019 MLA paged decode/prefill | Attention | FlashInfer `BatchMLAPagedAttentionWrapper` |
| 020 MoE FP8 | MoE | FlashInfer `trtllm_fp8_block_scale_moe` |

> 这些 task 的 baseline solution 代码已写好，只需下载数据文件即可运行。

### 3.6 GEMM dtype 不匹配（8 个）

FlashInfer-Bench 的 GEMM task（004-011）使用 float16，但 FlashInfer 的 `mm_bf16` 只支持 bfloat16。dtype 转换会导致精度损失，无法通过 correctness 校验。

### 3.7 Quant FP8/NVFP4（33 个）

Quant 目录下的 task 使用 FP8 / NVFP4 数据类型。DeepGEMM 和 FlashInfer 提供了 FP8 GEMM，但 eval driver 生成的随机 tensor 是标准 dtype，不是 FP8 格式。需要特殊的数据生成逻辑。

### 3.8 Mamba/SSM API 差异（6 个）

mamba-ssm 库提供的是融合式 API（如 `Mamba2` 模块），而 L1/L2 的 Mamba task 是分解式实现，API 不兼容。

| Task | 原因 |
|---|---|
| L1_056 mamba_ssm_dt_projection (backward) | Backward + 分解式实现 |
| L1_070 mamba2_intra_chunk_diagonal | 分解式 chunk 操作 |
| L2_043 mamba_chunk_scan_with_segsum | 分解式 chunk scan |
| L2_044 mamba_discretization_and_segsum | 分解式离散化 |
| L2_058 mamba2_selective_scan | 分解式 Mamba2 块 |

### 3.9 MoE 融合操作（~15 个）

MoE task 涉及路由、调度、专家分配、加权聚合等多步操作，不是纯 GEMM。

| Task | 原因 |
|---|---|
| L1_009 expert_token_scatter (backward) | Backward + 调度 |
| L1_044 moe_expert_computation | 多专家 SwiGLU + 调度 |
| L1_076 batched_expert_forward | 批处理专家 + 调度 |
| L2_008-013 MoE routing/dispatch | 路由 + 调度 + 专家分配 |
| L2_024-026 MoE parallel execution | 并行专家 + 调度 |
| L2_047-049 MoE training/inference | 训练/推理特定调度 |
| L2_061-065 MoE sparse routing | 稀疏路由 + 调度 |
| L2_080-082 MoE complete layer | 完整 MoE 层 |

---

## 四、总结

### 已匹配的 16 个 baseline 覆盖了以下算子类型：

| 算子类型 | Baseline 数量 | 库 |
|---|---|---|
| RMSNorm | 11 | FlashInfer |
| GEGLU | 2 | Liger |
| GroupNorm | 1 | Liger |
| Causal Conv1D | 2 | causal-conv1d |

### 无需匹配的 63 个 task 是因为：

1. **torch 已充分优化**：matmul（cuBLAS）、conv（cuDNN）、gelu/silu（cuDNN）、softmax（cuDNN）
2. **纯数学/调度操作**：频率计算、路由、排序、掩码生成
3. **架构特定操作**：Hyena FFT、AltUp、RWKV

### 无法匹配的 156 个 task 是因为：

1. **Backward pass**（~30）：SOTA 库不提供 backward
2. **融合度过高**（~50）：单个 task 包含多个操作，SOTA 库只做其中之一
3. **API 差异**（~20）：RoPE 布局、Mamba 分解式 vs 融合式
4. **数据依赖**（~10）：需要外部数据集
5. **dtype 限制**（~8）：float16 vs bfloat16
6. **FP8/NVFP4**（~33）：需要特殊数据类型支持
7. **MoE 调度**（~15）：非 GEMM 计算