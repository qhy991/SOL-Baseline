# SOTA Library Composition: A Methodology for Kernel Baseline Matching

> **核心洞察**：未来算子优化的方向之一，可能是建立一个开源 SOTA 算子库索引，面对新需求时通过组合现有 SOTA 实现来快速产出高性能 baseline。

## 1. 问题的演化

在 sol-execbench 的 baseline 匹配过程中，我们经历了三个阶段的认知演化：

### 阶段 1：1:1 直接匹配（覆盖 16 个 task）

最初的策略是寻找与 task reference 完全对应的 SOTA 库函数：

```
task reference 调用 rmsnorm
→ baseline 调用 FlashInfer rmsnorm
```

这种方法只能覆盖最简单的 task，因为大多数 task 是融合的多操作 kernel。

### 阶段 2：判定"融合度过高无法匹配"（误判 ~50 个 task）

发现大量 task 是融合操作（QKV 投影 + RoPE + Attention + Output 投影），SOTA 库通常只提供其中单个操作。简单结论是"这些 task 无法匹配 SOTA baseline"。

### 阶段 3：组合 SOTA 库（再覆盖 8+ 个，并持续扩展中）

关键洞察：**虽然单个 SOTA 库无法覆盖一个融合 task，但通过组合多个 SOTA 库就能做到。**

## 2. 组合方法论

### 核心思路

将 task reference 的计算图拆解为：
1. **SOTA 可加速的部分**：RMSNorm、Attention SDPA、激活函数、Causal Conv 等 — 调用 SOTA 库
2. **torch 已优化的部分**：Linear projection、Reshape、Element-wise 操作 — 保持 torch 实现
3. **特殊计算**：如 task 特有的 RoPE 变体、3D position encoding — 保持 reference 实现

### 典型组合示例

#### 例 1：完整 GQA Attention 块 (L1_092, L1_015)

```
task reference: QKV proj → QK RMSNorm → RoPE → GQA Repeat → Causal SDPA → Output proj
                ↓           ↓             ↓       ↓             ↓            ↓
baseline:       torch.linear → FlashInfer.rmsnorm → torch RoPE → (FA 原生支持GQA) → FA flash_attn_func(causal) → torch.linear
```

代码模式：
```python
# 1. QKV 投影 (torch)
q = F.linear(hidden, q_weight)
k = F.linear(hidden, k_weight)
v = F.linear(hidden, v_weight)

# 2. QK RMSNorm (FlashInfer)
q_flat = q.reshape(-1, head_dim).contiguous()
q = flashinfer.norm.rmsnorm(q_flat, q_norm_w, eps).reshape(...)
# 同样处理 k

# 3. RoPE (torch, 因为 RoPE 实现可能因模型而异)
q = (q * cos) + (rotate_half(q) * sin)
k = (k * cos) + (rotate_half(k) * sin)

# 4. FlashAttention (原生支持 GQA + causal)
attn_output = flash_attn_func(q, k, v, causal=True, softmax_scale=scale)

# 5. Output projection (torch)
output = F.linear(attn_output, o_weight)
```

#### 例 2：完整 Decoder Layer (L2_053, L2_020, L2_062)

```
task reference: PreNorm → SelfAttn → Residual → PreNorm → SwiGLU MLP → Residual
                ↓           ↓        ↓          ↓          ↓               ↓
baseline:       FlashInfer → FA SDPA → torch  → FlashInfer → torch+silu → torch
```

可以堆叠多个 SOTA kernel 实现完整的 transformer block。

#### 例 3：Encoder-Decoder Cross Attention (L2_062)

```
task: Self-attn (causal) + Cross-attn (non-causal) + MLP
baseline: 用 flash_attn_func(causal=True) + flash_attn_func(causal=False) + torch MLP
```

#### 例 4：Cu_seqlens 变长 Attention (L2_018)

```
task: ragged sequence attention with cu_seqlens
baseline: 直接用 flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
```

## 3. 已验证的组合模式

### Pattern A: FlashInfer RMSNorm + FlashAttention
- L1_015 gqa_rope_qk_norm ✓ 16/16
- L1_092 gqa_attention_with_qk_norm ✓ 16/16
- L2_020 decoder_layer_pre_post_norm_residual ✓ 16/16
- L2_053 text_decoder_layer_with_self_attention_and_mlp ✓ 16/16
- L2_062 decoder_complete_layer (self+cross+MLP) ✓ 16/16

### Pattern B: FlashInfer RMSNorm only (without attention)
- L1_033 post_norm_residual ✓ 16/16
- L1_069 rms_norm ✓ 16/16
- L1_073 encoder_norm_kv_projection ✓ 16/16
- L2_004 fused_residual_rms_mlp ✓ 16/16

### Pattern C: FlashAttention (without QK norm)
- L2_007 multimodal_rotary_embedding_attention (3D RoPE) ✓ 16/16
- L2_018 cu_seqlens_vision_attention (varlen) ⚠ 11/16

### Pattern D: causal-conv1d in Mamba-like blocks
- L1_005 conv_gated_projection_with_causal_conv ✓ 16/16
- L1_029 mamba_conv1d_with_gating ✓ 16/16

### Pattern E: Liger fused activation
- L1_048 fused_gate_up_projection_with_swiglu (GEGLU) ✓ 16/16
- L1_085 geglu_activation ✓ 16/16
- L1_078 group_norm_fusion ✓ 16/16

## 4. 组合策略的关键原则

### 4.1 拆解粒度

正确的拆解粒度是把 task 拆解为**最大可被 SOTA 库覆盖的子图**，而不是更细。例如：

✗ 错误：把整个 attention block 拆成 5 个 SOTA 调用（Q proj + K proj + V proj + SDPA + O proj）
- 实际上 Q/K/V projection 是 torch.matmul，没有 SOTA 加速

✓ 正确：保留 linear 投影为 torch，只把 RMSNorm 和 SDPA 换成 SOTA
- linear 已经是 cuBLAS 最优
- RMSNorm 和 SDPA 是真正的瓶颈

### 4.2 保留 reference 的特殊逻辑

某些计算在不同模型中有变体（如 RoPE 有标准/复数式/3D/YaRN 多种），SOTA 库通常只支持某一种。**保留 reference 的实现，不试图用 SOTA 库替代**。

例如 L2_007 的 3D multimodal RoPE：
```python
# task 特有的 RoPE 实现 - 保留 reference 代码
cos_combined = torch.cat([m[i % 3] for i, m in enumerate(cos_splits)], dim=-1)
# 然后只把 SDPA 换成 FlashAttention
attn_output = flash_attn_func(q, k, v, causal=True)
```

### 4.3 dtype/shape 适配层

SOTA 库通常对输入有严格要求，wrapper 需要做适配：

```python
# FlashInfer rmsnorm 要求 2D 输入
x_flat = hidden_states.reshape(-1, hidden_size).contiguous()
normed = flashinfer.norm.rmsnorm(x_flat, weight, eps)
normed = normed.reshape(batch, seq_len, hidden_size)

# FlashAttention 要求 (batch, seq_len, num_heads, head_dim) 布局
# 而 RoPE 在 (batch, num_heads, seq_len, head_dim) 布局更自然
q = q.transpose(1, 2).contiguous()  # 切换布局

# FlashAttention 要求 bfloat16/float16，float32 task 不能用
```

### 4.4 容差感知

不同 task 容差不同。组合 SOTA 库时累积的数值误差需控制在容差范围内：

- bfloat16 task with atol=0.005：通常能通过
- float32 task with atol=0.0001：bfloat16 SOTA 库无法通过
- 短序列 (<1024) 比长序列更容易超出容差（FA softmax 数值差异）

## 5. 现实意义：算子优化的新范式

### 5.1 传统范式：从零写 kernel

```
新模型出现 → 工程师从零写 CUDA/Triton kernel → 数月开发周期
```

### 5.2 组合范式：索引 + 组合

```
新模型出现 → 索引 SOTA 库找到匹配组件 → wrapper 组合 → 即可获得高性能 baseline
                ↑
        这是关键基础设施
```

### 5.3 所需的基础设施

要让"SOTA 库组合"成为可行的算子优化范式，需要：

1. **SOTA 算子库索引**：建立类似 PyPI 的索引，标注每个库支持的：
   - 算子类型（attention, norm, activation, conv, ...）
   - dtype 支持（bf16, fp16, fp8, ...）
   - shape 约束
   - GPU 架构（SM80, SM90, SM100）
   - 性能基准

2. **组合模式库**：收集常见组合模式（如本文档的 Pattern A-E）

3. **自动 wrapper 生成**：给定 task reference 和 SOTA 库索引，自动生成 wrapper 代码

4. **性能 + 正确性验证管道**：自动测试 wrapper 是否通过 correctness + 是否真有加速

### 5.4 sol-execbench 验证了这个范式

通过组合 FlashInfer + FlashAttention + Liger + causal-conv1d 四个 SOTA 库，我们在 sol-execbench 上覆盖了：
- 简单算子（RMSNorm, GEGLU）：直接 1:1 匹配
- 中等复杂度（fused RMSNorm + MLP）：2-3 个库组合
- 完整 attention block：3-4 个库组合
- 完整 decoder layer（含 cross-attention）：4+ 个库组合

这证明了**复杂的融合 kernel 不一定需要从零写**，可以通过组合现有 SOTA 实现快速产出。

## 6. 未来探索方向

1. **更多 SOTA 库的引入**：FlashMLA、TransformerEngine FP8、SGL Kernel、cuDNN frontend 等
2. **跨库 fusion 优化**：当多个 SOTA 库依次调用时，能否进一步 fuse 减少 memory roundtrip？
3. **自动组合搜索**：给定 task spec，自动搜索最优 SOTA 库组合
4. **运行时调度**：根据输入 shape 动态选择最优 SOTA 库（如小 seq 用 torch，大 seq 用 FA）

## 7. 结论

**"SOTA 库组合"是一种被低估的算子优化方法论**：

- ✓ 大幅扩展了可匹配的 task 数量（从认为不可能 → 实际可行）
- ✓ 复用业界经过验证的高性能实现，避免重复造轮子
- ✓ 加速从模型出现到高性能部署的时间
- ✓ 为新算子开发提供了 baseline 参照系

但也需要：
- ⚠ 维护 SOTA 库索引和兼容性矩阵
- ⚠ 容差和数值精度的精细把控
- ⚠ 接受组合带来的 wrapper overhead（通常很小）