#!/usr/bin/env python3
"""Discover all Contest baselines, run them, and emit a Markdown summary table.

Usage:
  cd sol-execbench
  uv run python ../sol-baseline/scripts/aggregate_contest.py [--output FILE]
"""
import argparse
import json
import os
import statistics as stats
import subprocess
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent.parent / "baselines"
BENCHMARK_ROOT = Path("data/benchmark/Contest")
CONFIG_FULL = Path(__file__).resolve().parent / "bench_config.json"
CONFIG_FAST = Path(__file__).resolve().parent / "bench_config_fast.json"

# Tasks whose reference is a Python loop (per-expert MoE, per-batch attention)
# that exceeds the 100-iter benchmark timeout. Run correctness-only.
SLOW_REFERENCE = {
    "044_moe_expert_computation",
    "008_moe_sparse_routing_and_dispatch",
    "010_moe_expert_computation_with_weighted_accumulation",
    "013_expert_weighted_aggregation_with_shared_expert",
    "029_moe_sparse_routing_and_dispatch",
    "065_sparse_expert_dispatch_and_combine",
    "081_moe_sparse_expert_dispatch",
    "082_moe_layer_complete_forward_with_residual",
    "009_decoder_layer_with_residual_connections",
    # FIB attention wrappers - reference is Python loop
    "013_gqa_paged_decode_h32_kv8_d128_ps1",
    "017_gqa_ragged_prefill_causal_h32_kv8_d128",
    "018_mla_paged_decode_h16_ckv512_kpe64_ps1",
    "019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
}

ENV = {**os.environ, "FLASHINFER_TRACE_DIR": str(Path("data/flashinfer-trace").resolve())}


def find_task_dir(def_name):
    for cat_dir in BENCHMARK_ROOT.iterdir():
        if not cat_dir.is_dir():
            continue
        for d in cat_dir.iterdir():
            if d.is_dir() and d.name.endswith(def_name):
                return cat_dir.name, d
    return None, None


def run_one(task_dir, sol_path, slow):
    cfg = CONFIG_FAST if slow else CONFIG_FULL
    cmd = [
        "uv", "run", "--no-sync", "sol-execbench",
        str(task_dir),
        "--solution", str(sol_path),
        "--config", str(cfg),
        "--timeout", "600",
        "--json",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=1500)
    except subprocess.TimeoutExpired:
        return None, None, 0
    lines = [L for L in r.stdout.splitlines() if L.strip().startswith("{")]
    passed = 0
    total = 0
    speedups = []
    for L in lines:
        try:
            d = json.loads(L)
        except json.JSONDecodeError:
            continue
        total += 1
        ev = d.get("evaluation") or {}
        if ev.get("status") == "PASSED":
            passed += 1
        p = ev.get("performance") or {}
        ref = p.get("reference_latency_ms", 0) or 0
        sol = p.get("latency_ms", 0) or 0
        if ref > 0 and sol > 0:
            speedups.append(ref / sol)
    return passed, total, speedups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="-", help="Write Markdown to file (default stdout)")
    args = ap.parse_args()

    solutions = sorted(BASELINE_ROOT.glob("*/Contest/*/solution.json"))
    rows = []
    for sol_path in solutions:
        with open(sol_path) as f:
            sol = json.load(f)
        def_name = sol["definition"]
        library = sol["author"]
        cat, task_dir = find_task_dir(def_name)
        if task_dir is None:
            print(f"  SKIP {def_name}: not found in benchmark", file=sys.stderr)
            continue
        slow = def_name in SLOW_REFERENCE
        print(f"  Running {cat}/{def_name} ({library}) {'[fast]' if slow else ''}", file=sys.stderr)
        passed, total, speedups = run_one(task_dir, sol_path, slow)
        if total is None:
            rows.append((cat, def_name, library, "TIMEOUT", "n/a", "n/a", "n/a"))
        else:
            stat = f"{passed}/{total}"
            if speedups:
                mean = f"{stats.mean(speedups):.2f}x"
                rng = f"{min(speedups):.2f}–{max(speedups):.2f}x"
            else:
                mean = "n/a"
                rng = "n/a"
            rows.append((cat, def_name, library, stat, mean, rng,
                         "fast" if slow else "full"))

    out_lines = [
        "# Contest baselines — aggregate results",
        "",
        f"Total: **{len(rows)} baselines**, environment: NVIDIA B200 (SM100), CUDA 13, torch 2.9, FlashInfer 0.6.12, flash_attn 2.8.3.",
        "",
        "| Category | Task | Library | Pass | Mean speedup | Range | Bench |",
        "|---|---|---|---|---|---|---|",
    ]
    rows.sort(key=lambda r: (r[0], r[1]))
    for cat, name, lib, stat, mean, rng, kind in rows:
        out_lines.append(f"| {cat} | {name} | {lib} | {stat} | {mean} | {rng} | {kind} |")

    nums = []
    for _, _, _, _, mean, _, _ in rows:
        if mean.endswith("x"):
            try:
                nums.append(float(mean[:-1]))
            except ValueError:
                pass
    if nums:
        gm = 1.0
        for x in nums:
            gm *= x
        gm = gm ** (1 / len(nums))
        out_lines.extend(["", f"**Geometric mean speedup over reference:** {gm:.2f}x ({len(nums)} timed)"])

    text = "\n".join(out_lines) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
        print(f"  wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
