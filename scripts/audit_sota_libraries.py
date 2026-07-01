#!/usr/bin/env python3
"""Audit community SOTA library availability for Contest baselines.

The script intentionally separates two questions:

1. Can the target Python environment import the library today?
2. If yes, which Contest negative/deferred tasks should be re-attempted?

Run it with the same interpreter used for sol-execbench, for example:

  python scripts/audit_sota_libraries.py \
      --python /home/qinhaiyan/sol-execbench/.venv/bin/python \
      --extra-path deep_gemm=/home/qinhaiyan/DeepGEMM \
      --output docs/SOTA_COVERAGE_AUDIT.md
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    key: str
    display: str
    modules: tuple[str, ...]
    source_env: str | None
    category: str
    priority: str
    contest_targets: tuple[str, ...]
    expected_value: str
    blocking_notes: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="deep_gemm",
        display="DeepGEMM",
        modules=("deep_gemm",),
        source_env="DEEPGEMM_SRC",
        category="FP8/BF16 GEMM and MoE",
        priority="P0",
        contest_targets=(
            "FIB/005 fp16 GEMM router shape",
            "FIB/020 FP8 routed MoE",
            "L2/012 padded MoE grouped GEMM",
            "L2/048 fp32 MoE dispatch if precision can be preserved",
            "Quant/011 FP8 router projection",
        ),
        expected_value="Best direct missing candidate for GEMM/MoE negatives.",
        blocking_notes=(
            "Requires compiled deep_gemm._C extension. On SM100, FP8 scale layout may need "
            "packed UE8M0 conversion, so importability is necessary but not sufficient."
        ),
    ),
    Candidate(
        key="transformer_engine",
        display="TransformerEngine",
        modules=("transformer_engine", "transformer_engine.pytorch"),
        source_env=None,
        category="FP8 layers",
        priority="P0",
        contest_targets=(
            "Quant FP8 projection tasks with non-FlashInfer recipes",
            "L2/019 fp32 decoder if an FP8 path is acceptable under tolerance",
            "L2/048 fp32 MoE dispatch if precision can be preserved",
        ),
        expected_value="Main NVIDIA library for FP8 Linear/LayerNorm/MLP experiments.",
        blocking_notes="Full PyTorch extension is required; the meta package alone is not enough.",
    ),
    Candidate(
        key="flash_attn3",
        display="FlashAttention 3 / Hopper-Blackwell attention path",
        modules=("flash_attn_interface", "flash_attn"),
        source_env=None,
        category="Attention",
        priority="P0",
        contest_targets=("L1/067 fp32 ultralong GQA attention",),
        expected_value="Only credible near-term path for re-testing fp32 ultralong attention.",
        blocking_notes="FA2 imports are not enough; fp32 support must be verified by an actual kernel call.",
    ),
    Candidate(
        key="sgl_kernel",
        display="sgl-kernel / SGLang custom ops",
        modules=("sgl_kernel", "sglang.srt._custom_ops"),
        source_env="SGLANG_SRC",
        category="Serving custom ops",
        priority="P1",
        contest_targets=(
            "L1/018 fused RoPE + QK norm + KV update",
            "L1/023 multimodal RoPE position work",
            "FIB/020 alternative FP8 MoE paths",
        ),
        expected_value="Useful for fused serving ops that FlashInfer 0.6.12 cannot cover on SM100.",
        blocking_notes="APIs are less stable than FlashInfer; wrappers need per-op compatibility checks.",
    ),
    Candidate(
        key="vllm",
        display="vLLM custom ops / vllm-flash-attn",
        modules=("vllm", "vllm._C"),
        source_env=None,
        category="Serving attention and MoE",
        priority="P1",
        contest_targets=(
            "L1/067 fp32 or long-context attention re-test",
            "L1/071 KV cache update + RoPE",
            "FIB attention wrappers as alternate implementation",
        ),
        expected_value="Independent serving stack for paged attention and custom CUDA ops.",
        blocking_notes="Large dependency surface; importability often depends on exact torch/CUDA build.",
    ),
    Candidate(
        key="megablocks",
        display="MegaBlocks / grouped_gemm",
        modules=("megablocks", "grouped_gemm"),
        source_env=None,
        category="MoE",
        priority="P1",
        contest_targets=(
            "L1/044 MoE expert compute",
            "L1/076 dense batched expert forward",
            "L2/008/L2/010/L2/012/L2/013/L2/029 MoE variants",
        ),
        expected_value="Potential grouped-GEMM alternative to per-expert torch.mm loops.",
        blocking_notes="Most useful when token routing layout matches dropless/block-sparse assumptions.",
    ),
    Candidate(
        key="fla",
        display="flash-linear-attention",
        modules=("fla",),
        source_env=None,
        category="Linear attention / SSM",
        priority="P2",
        contest_targets=("Not part of the 60-task Contest set; relevant to wider sol-execbench coverage.",),
        expected_value="Important for non-Contest Mamba2/RWKV/linear-attention tasks.",
        blocking_notes="Do not count as Contest coverage until a matching Contest task exists.",
    ),
    Candidate(
        key="natten",
        display="NATTEN",
        modules=("natten",),
        source_env=None,
        category="Vision attention",
        priority="P2",
        contest_targets=("Potential L1/L2 vision attention variants outside current shipped Contest gaps.",),
        expected_value="Worth tracking for wider vision attention coverage.",
        blocking_notes="Neighborhood attention is not a drop-in replacement for arbitrary cu_seqlens attention.",
    ),
    Candidate(
        key="xformers",
        display="xFormers",
        modules=("xformers",),
        source_env=None,
        category="Attention and fused activations",
        priority="P2",
        contest_targets=("L1/046 softcap/softmax sanity check", "L1/021 varlen attention sanity check"),
        expected_value="Useful as an independent comparison for attention/activation negatives.",
        blocking_notes="Often behind PyTorch SDPA/FlashAttention on modern NVIDIA inference shapes.",
    ),
    Candidate(
        key="aiter",
        display="AITER",
        modules=("aiter",),
        source_env=None,
        category="ROCm-oriented kernels",
        priority="P3",
        contest_targets=("None for B200 CUDA environment unless a CUDA backend is present.",),
        expected_value="Track explicitly so CUDA/B200 reviewers do not assume it was forgotten.",
        blocking_notes="Generally ROCm-focused; low expected value for NVIDIA B200 Contest runs.",
    ),
)


PROBE = r"""
import importlib
import importlib.util
import json
import os
import sys

modules = json.loads(os.environ["SOTA_PROBE_MODULES"])
result = {"python": sys.executable, "sys_path_head": sys.path[:5], "modules": []}
for module in modules:
    entry = {"module": module, "found": False, "imported": False, "origin": None, "error": None}
    try:
        spec = importlib.util.find_spec(module)
        if spec is not None:
            entry["found"] = True
            entry["origin"] = spec.origin
            imported = importlib.import_module(module)
            entry["imported"] = True
            entry["version"] = getattr(imported, "__version__", None)
    except BaseException as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    result["modules"].append(entry)
print(json.dumps(result, sort_keys=True))
"""


def parse_extra_paths(values: Iterable[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--extra-path must be KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        paths[key.strip()] = path.strip()
    return paths


def run_probe(python: str, modules: tuple[str, ...], extra_path: str | None) -> dict[str, object]:
    env = os.environ.copy()
    env["SOTA_PROBE_MODULES"] = json.dumps(list(modules))
    if extra_path:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = extra_path if not existing else extra_path + os.pathsep + existing
    proc = subprocess.run(
        [python, "-c", PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    if proc.returncode != 0:
        return {
            "python": python,
            "modules": [],
            "probe_error": proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}",
        }
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {"python": python, "modules": [], "probe_error": proc.stdout.strip()}
    return json.loads(lines[-1])


def candidate_status(probe: dict[str, object]) -> tuple[str, str, str]:
    modules = probe.get("modules") or []
    if not isinstance(modules, list) or not modules:
        return "blocked", "probe failed", str(probe.get("probe_error", "no module data"))

    imported = [m for m in modules if isinstance(m, dict) and m.get("imported")]
    found = [m for m in modules if isinstance(m, dict) and m.get("found")]
    errors = [m for m in modules if isinstance(m, dict) and m.get("error")]

    if imported:
        origins = "; ".join(
            f"{m.get('module')} ({m.get('version') or 'version unknown'}) @ {m.get('origin')}"
            for m in imported
        )
        return "available", "import succeeded", origins
    if errors:
        detail = "; ".join(f"{m.get('module')}: {m.get('error')}" for m in errors)
        if found:
            return "blocked", "found but import failed", detail
        return "missing", "not installed", detail
    return "missing", "not installed", "no candidate module found on sys.path"


def markdown_report(rows: list[dict[str, object]], python: str, extra_paths: dict[str, str]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Community SOTA Coverage Audit",
        "",
        f"Generated: {now}",
        f"Target Python: `{python}`",
        "",
        "This audit checks whether candidate community SOTA libraries are actually importable in the target benchmark environment. It does not claim a library is faster; `available` means the next step is to write and benchmark a task-specific wrapper.",
        "",
    ]
    if extra_paths:
        lines.extend([
            "## Extra Source Paths",
            "",
            "| Key | Path |",
            "|---|---|",
        ])
        for key, path in sorted(extra_paths.items()):
            lines.append(f"| `{key}` | `{path}` |")
        lines.append("")

    lines.extend([
        "## Result Matrix",
        "",
        "| Priority | Library | Status | Probe Result | Contest re-attempt targets |",
        "|---|---|---|---|---|",
    ])
    for row in rows:
        targets = "<br>".join(str(t) for t in row["contest_targets"])
        lines.append(
            f"| {row['priority']} | {row['display']} | **{row['status']}** | {row['summary']} | {targets} |"
        )

    lines.extend([
        "",
        "## Details",
        "",
    ])
    for row in rows:
        lines.extend([
            f"### {row['display']}",
            "",
            f"- Category: {row['category']}",
            f"- Status: {row['status']} ({row['summary']})",
            f"- Probe detail: `{row['detail']}`",
            f"- Expected value: {row['expected_value']}",
            f"- Blocking notes: {row['blocking_notes']}",
            "",
        ])

    available = [r for r in rows if r["status"] == "available"]
    blocked = [r for r in rows if r["status"] == "blocked"]
    missing = [r for r in rows if r["status"] == "missing"]
    lines.extend([
        "## Reviewer Conclusion",
        "",
        f"- Available candidates: {len(available)}",
        f"- Blocked candidates: {len(blocked)}",
        f"- Missing candidates: {len(missing)}",
        "- A claim that all community SOTA implementations have been exhausted is valid only when every P0/P1 candidate is either benchmarked or has a recorded task-specific blocker.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to probe")
    parser.add_argument("--extra-path", action="append", default=[], help="Add KEY=PATH to PYTHONPATH for that candidate")
    parser.add_argument("--output", default="-", help="Write Markdown report to file, or '-' for stdout")
    parser.add_argument("--json-output", help="Optional JSON report path")
    args = parser.parse_args()

    extra_paths = parse_extra_paths(args.extra_path)
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        source_path = None
        if candidate.key in extra_paths:
            source_path = extra_paths[candidate.key]
        elif candidate.source_env and os.environ.get(candidate.source_env):
            source_path = os.environ[candidate.source_env]

        probe = run_probe(args.python, candidate.modules, source_path)
        status, summary, detail = candidate_status(probe)
        rows.append({
            "key": candidate.key,
            "display": candidate.display,
            "modules": candidate.modules,
            "source_path": source_path,
            "category": candidate.category,
            "priority": candidate.priority,
            "contest_targets": candidate.contest_targets,
            "expected_value": candidate.expected_value,
            "blocking_notes": candidate.blocking_notes,
            "status": status,
            "summary": summary,
            "detail": detail,
            "probe": probe,
        })

    report = markdown_report(rows, args.python, extra_paths)
    if args.output == "-":
        print(report)
    else:
        Path(args.output).write_text(report)

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(rows, indent=2, sort_keys=True))

    p0_blocked = [r for r in rows if r["priority"] == "P0" and r["status"] != "available"]
    if p0_blocked:
        blocked = ", ".join(str(r["display"]) for r in p0_blocked)
        print(f"P0 candidates are not fully available: {blocked}", file=sys.stderr)
    print(f"Wrote {args.output if args.output != '-' else 'stdout'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
