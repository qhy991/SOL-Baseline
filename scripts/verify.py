#!/usr/bin/env python3
"""
Verify all baseline solutions pass correctness checks.

Usage:
    cd sol-execbench
    python ../SOL-Baseline/scripts/verify.py
"""

import json
import subprocess
import sys
from pathlib import Path

BASELINE_DIR = Path("baselines") if Path("baselines").exists() else Path(__file__).parent.parent
BENCHMARK_DIR = Path("data/benchmark")


def find_problem_dir(definition_name):
    """Find the problem directory for a given definition."""
    for cat in ["FlashInfer-Bench", "L1", "L2", "Quant", "Contest"]:
        cat_dir = BENCHMARK_DIR / cat
        if not cat_dir.exists():
            continue
        for d in cat_dir.iterdir():
            if d.is_dir() and d.name.endswith(definition_name):
                return d
    return None


def main():
    baseline_dir = BASELINE_DIR / "baselines"
    if not baseline_dir.exists():
        print("ERROR: baselines directory not found. Run from sol-execbench root.")
        sys.exit(1)

    # Match both depth-2 ({lib}/{task}/solution.json) and depth-3
    # ({lib}/Contest/{task}/solution.json) layouts.
    solutions = sorted(set(baseline_dir.glob("*/*/solution.json")) |
                       set(baseline_dir.glob("*/*/*/solution.json")))
    print(f"Found {len(solutions)} baseline solutions\n")

    passed = 0
    failed = 0

    for sol_path in sorted(solutions):
        with open(sol_path) as f:
            solution = json.load(f)

        def_name = solution["definition"]
        library = solution["author"]
        problem_dir = find_problem_dir(def_name)

        if not problem_dir:
            print(f"  SKIP {def_name}: problem directory not found")
            continue

        result = subprocess.run(
            [
                "uv", "run", "sol-execbench",
                str(problem_dir),
                "--solution", str(sol_path),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env={**__import__("os").environ, "PATH": f"{Path('.venv/bin').absolute()}:{__import__('os').environ.get('PATH', '')}"},
        )

        if result.returncode != 0:
            print(f"  ✗ {def_name} ({library}): CLI ERROR")
            failed += 1
            continue

        statuses = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                statuses.append(d["evaluation"]["status"])
            except (json.JSONDecodeError, KeyError):
                continue

        pass_count = statuses.count("PASSED")
        total = len(statuses)

        if total > 0 and pass_count == total:
            print(f"  ✓ {def_name} ({library}): {pass_count}/{total} PASSED")
            passed += 1
        elif total > 0:
            bad = [s for s in statuses if s != "PASSED"]
            print(f"  ✗ {def_name} ({library}): {pass_count}/{total} PASSED, failures: {set(bad)}")
            failed += 1
        else:
            print(f"  ✗ {def_name} ({library}): no output")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Total: {passed} passed, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()