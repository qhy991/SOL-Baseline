#!/usr/bin/env python3
"""Run a small DeepGEMM smoke test for Contest re-attempt decisions.

This script is deliberately narrow. It answers whether DeepGEMM is importable,
whether its BF16 GEMM kernel can launch on the current GPU, and whether the
same API accepts the FP16 input dtype used by Contest/FIB/005.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROBE = r"""
import json
import torch

result = {"imported": False, "cuda_available": torch.cuda.is_available()}
try:
    import deep_gemm
    result["imported"] = True
    result["version"] = getattr(deep_gemm, "__version__", None)
    result["module_file"] = getattr(deep_gemm, "__file__", None)
except BaseException as exc:
    result["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

if not torch.cuda.is_available():
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

result["device_name"] = torch.cuda.get_device_name(0)
result["device_capability"] = torch.cuda.get_device_capability(0)

torch.manual_seed(0)
rows, cols, inner = 128, 256, 7168
a = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
b = torch.randn(cols, inner, device="cuda", dtype=torch.bfloat16)
d = torch.empty(rows, cols, device="cuda", dtype=torch.bfloat16)
try:
    deep_gemm.bf16_gemm_nt(a, b, d)
    torch.cuda.synchronize()
    ref = a @ b.T
    result["bf16_gemm_nt"] = "ok"
    result["bf16_max_abs_diff"] = float((d.float() - ref.float()).abs().max().item())
except BaseException as exc:
    result["bf16_gemm_nt"] = "failed"
    result["bf16_error"] = f"{type(exc).__name__}: {exc}"

try:
    ah = a.to(torch.float16)
    bh = b.to(torch.float16)
    dh = torch.empty(rows, cols, device="cuda", dtype=torch.float16)
    deep_gemm.bf16_gemm_nt(ah, bh, dh)
    torch.cuda.synchronize()
    result["fp16_inputs_to_bf16_gemm_nt"] = "accepted"
except BaseException as exc:
    result["fp16_inputs_to_bf16_gemm_nt"] = "rejected"
    result["fp16_rejection"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

print(json.dumps(result, sort_keys=True))
"""


def run_probe(python: str) -> dict[str, object]:
    proc = subprocess.run(
        [python, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {
            "probe_failed": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    result = json.loads(lines[-1])
    result["returncode"] = proc.returncode
    if proc.stderr.strip():
        result["stderr"] = proc.stderr.strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to probe")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    result = run_probe(args.python)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0 if result.get("imported") and result.get("bf16_gemm_nt") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
