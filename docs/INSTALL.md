# Baseline Library Installation Guide

This document describes how to install the SOTA GPU kernel libraries used as baselines for sol-execbench evaluation.

## Requirements

- NVIDIA GPU with CUDA support (SM80+)
- Python 3.10+
- PyTorch 2.x with CUDA
- sol-execbench installed (`uv sync --all-groups`)

## Quick Install (All Libraries)

```bash
# Clone sol-execbench and install baseline libs
cd sol-execbench
./scripts/install_baseline_libs.sh

# Or with GPU test verification
./scripts/install_baseline_libs.sh --test
```

## Library Details

### 1. FlashInfer

**Version:** 0.6.12 (pre-installed with sol-execbench)
**Used for:** RMSNorm, fused_add_rmsnorm
**Baselines:** 11

FlashInfer is a high-performance GPU kernel library for LLM inference, providing optimized implementations of normalization, attention, and activation functions.

```bash
# FlashInfer is typically pre-installed with sol-execbench
# Verify installation:
python -c "import flashinfer.norm; print('FlashInfer OK')"
```

**Key APIs used:**
- `flashinfer.norm.rmsnorm(x, weight, eps)` — RMS normalization
- `flashinfer.norm.fused_add_rmsnorm(input, residual, weight, eps)` — Fused residual + RMSNorm

**Supported dtypes:** bfloat16, float16
**Supported GPUs:** SM80+ (A100, H100, B200, RTX 4090)

### 2. Liger Kernel

**Version:** 0.8.0
**Install:** `uv pip install liger-kernel`
**Used for:** GEGLU, SwiGLU, GroupNorm
**Baselines:** 3

Liger Kernel is a Triton-based kernel library providing fused MLP and normalization operations for LLM training.

```bash
uv pip install liger-kernel

# Verify:
python -c "from liger_kernel.ops.geglu import LigerGELUMulFunction; print('Liger OK')"
```

**Key APIs used:**
- `LigerGELUMulFunction.apply(a, b)` — GEGLU: GELU(a) * b
- `LigerSiLUMulFunction.apply(a, b)` — SwiGLU: SiLU(a) * b
- `LigerGroupNormFunction.apply(x, w, b, C, G, eps)` — GroupNorm

**Known issues:**
- Liger 0.8.0 `LigerSiLUMulFunction` has PyTorch 2.9 compatibility issues (`torch.distributed.tensor`). Use `swiglu_forward` directly as a workaround.
- Liger 0.8.0 `LigerRMSNorm` has the same PyTorch 2.9 compatibility issue. Use FlashInfer RMSNorm instead.

**Supported dtypes:** bfloat16, float16
**Supported GPUs:** SM80+

### 3. causal-conv1d

**Version:** 1.6.2.post1
**Install:** Prebuilt wheel from GitHub releases
**Used for:** Causal depthwise convolution
**Baselines:** 2

causal-conv1d provides optimized CUDA kernels for causal 1D convolution, primarily used in Mamba/SSM architectures.

```bash
# Download prebuilt wheel
# Wheel naming: causal_conv1d-{version}+{cuda}torch{torch_ver}cxx11abi{TRUE|FALSE}-{cp}-{cp}-linux_x86_64.whl
# Example for CUDA 13.0, PyTorch 2.9, CXX11 ABI=True, Python 3.12:
wget "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Install
pip install causal_conv1d-*.whl --no-deps

# Verify:
python -c "from causal_conv1d import causal_conv1d_fn; print('causal-conv1d OK')"
```

**Key APIs used:**
- `causal_conv1d_fn(x, weight, bias, activation)` — Causal 1D convolution

**Wheel naming convention:**
```
causal_conv1d-{version}+{cuda}torch{torch_major}.{torch_minor}cxx11abi{TRUE|FALSE}-cp{python_ver}-cp{python_ver}-linux_x86_64.whl
```

To find the right wheel for your environment:
```bash
python -c "
import torch, sys
tv = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
pv = f'cp{sys.version_info.major}{sys.version_info.minor}'
abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
cu = 'cu' + torch.version.cuda.replace('.','')[:4]
print(f'causal_conv1d-1.6.2.post1+{cu}torch{tv}cxx11abi{abi}-{pv}-{pv}-linux_x86_64.whl')
"
```

**Supported dtypes:** bfloat16, float16
**Supported GPUs:** SM80+

## Environment-Specific Notes

### SM100 (NVIDIA B200)
- **FlashInfer RMSNorm:** ✓ Fully supported
- **FlashInfer activation (silu_and_mul, gelu_and_mul):** ⚠ bfloat16 kernels may need JIT compilation. Ensure `ninja` is installed and on PATH.
- **Liger Kernel:** ⚠ PyTorch 2.9 compatibility issues with some functions. Use direct Triton kernels (`swiglu_forward`, `geglu_forward`) as workaround.
- **causal-conv1d:** ✓ Fully supported

### SM90 (NVIDIA H100)
- All libraries fully supported

### SM89 (NVIDIA RTX 4090)
- All libraries fully supported
- DeepGEMM not supported (requires SM90+)

### SM80 (NVIDIA A100)
- All libraries fully supported
- DeepGEMM not supported (requires SM90+)
- FlashMLA additionally supported (SM80 only)

## Wheel Cache

Downloaded wheels are cached in `/tmp/sol-execbench-wheels/` by default. Re-runs of the install script skip completed downloads.

## Network Resilience

The install script handles unstable network connections:
- GitHub proxy: `ghfast.top` (configurable via `--proxy`)
- Wheel caching with integrity check
- Retry with exponential backoff (5 attempts)
- PyPI mirror: USTC mirror (configurable via `--pypi-mirror`)

## Troubleshooting

### ninja not found
FlashInfer may need `ninja` for JIT compilation on SM100:
```bash
uv pip install ninja
export PATH="$(pwd)/.venv/bin:$PATH"
```

### FlashInfer activation dtype error
FlashInfer 0.6.12 on SM100 may not support bfloat16 for activation functions (silu_and_mul, gelu_and_mul). Use Liger Kernel instead.

### Liger PyTorch 2.9 compatibility
```python
# Instead of LigerSiLUMulFunction.apply(a, b), use:
from liger_kernel.ops.swiglu import swiglu_forward
_, _, result = swiglu_forward(a, b)
```

### causal-conv1d weight shape error
causal-conv1d expects weight shape `(dim, width)`. If the task provides weight as `(dim, 1, width)`, squeeze the middle dimension:
```python
conv_w = conv_weight.reshape(conv_weight.shape[0], conv_weight.shape[2]).contiguous()
```