#!/usr/bin/env python3
"""
Benchmark SOTA baseline vs torch reference by running both in the same subprocess.

Strategy: modify the eval driver to run both reference and solution and
measure both latencies. We do this by creating a wrapper solution that
runs both.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

BENCHMARK = Path("data/benchmark")
BASELINE = Path("data/baseline")
WARMUP = 50
ITERS = 200


def resolve_axes(axes, axes_def):
    """Resolve all axes: constants, expressions, and variables."""
    resolved = dict(axes)  # Start with variable values from workload
    # First pass: resolve constant axes
    for name, spec in axes_def.items():
        if spec["type"] == "const":
            resolved[name] = spec["value"]
    # Second pass: resolve expression axes
    for name, spec in axes_def.items():
        if spec["type"] == "expr":
            resolved[name] = eval(spec["expression"], {"__builtins__": {}}, resolved)
    return resolved


def generate_inputs(definition, axes, device):
    """Generate random inputs based on definition and resolved axes."""
    inputs = {}
    for name, spec in definition["inputs"].items():
        shape = spec.get("shape")
        if shape is None:
            continue

        resolved_shape = []
        for dim in shape:
            # dim can be: axis name (str), integer literal as string ('1', '2'...),
            # or expression that needs evaluation
            if dim in axes:
                resolved_shape.append(axes[dim])
            elif isinstance(dim, str) and dim.lstrip('-').isdigit():
                resolved_shape.append(int(dim))
            elif isinstance(dim, int):
                resolved_shape.append(dim)
            else:
                # Try to evaluate as expression
                try:
                    resolved_shape.append(eval(dim, {"__builtins__": {}}, axes))
                except:
                    raise ValueError(f"Cannot resolve dimension: {dim} in axes {axes}")

        dtype_str = spec.get("dtype", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        if dtype == torch.int64:
            inputs[name] = torch.randint(0, 1000, resolved_shape, device=device)
        elif dtype == torch.bool:
            inputs[name] = torch.rand(resolved_shape, device=device) > 0.5
        else:
            inputs[name] = torch.randn(resolved_shape, dtype=dtype, device=device)

    return inputs


def load_module(content, name):
    """Load a module from string content."""
    import importlib.util
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(content, mod.__dict__)
    return mod


def benchmark(fn, inputs, warmup=WARMUP, iters=ITERS):
    """Benchmark a function."""
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(**inputs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1000


def main():
    device = torch.device("cuda")
    all_results = []

    for sol_dir in sorted(BASELINE.glob("*/*")):
        sol_path = sol_dir / "solution.json"
        if not sol_path.exists():
            continue

        with open(sol_path) as f:
            solution = json.load(f)

        def_name = solution["definition"]
        library = solution["author"]

        # Find problem directory
        problem_dir = None
        for cat in ["L1", "L2", "Quant"]:
            for d in (BENCHMARK / cat).iterdir():
                if d.is_dir() and d.name.endswith(def_name):
                    problem_dir = d
                    break
            if problem_dir:
                break
        if not problem_dir:
            continue

        # Load definition
        with open(problem_dir / "definition.json") as f:
            definition = json.load(f)

        # Load workloads
        with open(problem_dir / "workload.jsonl") as f:
            workloads = [json.loads(line) for line in f if line.strip()]

        # Load reference
        with open(problem_dir / "reference.py") as f:
            ref_code = f.read()
        ref_mod = load_module(ref_code, "reference")

        # Load baseline
        baseline_code = solution["sources"][0]["content"]
        baseline_mod = load_module(baseline_code, "baseline")

        task_results = []
        for wl in workloads:
            axes = resolve_axes(wl["axes"], definition["axes"])

            try:
                inputs = generate_inputs(definition, axes, device)

                # Handle custom_inputs_entrypoint
                if definition.get("custom_inputs_entrypoint"):
                    custom_fn = getattr(ref_mod, definition["custom_inputs_entrypoint"])
                    custom_inputs = custom_fn(axes, device)
                    inputs.update(custom_inputs)

                # Add scalar inputs
                for name, spec in definition["inputs"].items():
                    if spec["shape"] is None:
                        val = wl["inputs"][name]["value"]
                        if isinstance(val, float):
                            inputs[name] = val
                        elif isinstance(val, int):
                            inputs[name] = val

                # Benchmark both
                sota_lat = benchmark(baseline_mod.run, inputs)
                ref_lat = benchmark(ref_mod.run, inputs)

                task_results.append({
                    "axes": axes,
                    "sota_ms": sota_lat,
                    "ref_ms": ref_lat,
                    "speedup": ref_lat / sota_lat if sota_lat > 0 else float("inf"),
                })

            except Exception as e:
                print(f"  ERROR {def_name}: {e}")
                continue

        if task_results:
            avg_sota = sum(r["sota_ms"] for r in task_results) / len(task_results)
            avg_ref = sum(r["ref_ms"] for r in task_results) / len(task_results)
            avg_speedup = avg_ref / avg_sota if avg_sota > 0 else float("inf")

            all_results.append({
                "task": def_name,
                "library": library,
                "avg_sota_ms": avg_sota,
                "avg_ref_ms": avg_ref,
                "avg_speedup": avg_speedup,
                "workloads": len(task_results),
                "details": task_results,
            })

            print(f"  {def_name} ({library}): "
                  f"SOTA={avg_sota:.4f}ms, Ref={avg_ref:.4f}ms, "
                  f"Speedup={avg_speedup:.1f}x")

    # Print final table
    print(f"\n{'='*90}")
    print(f"{'Task':<40} {'Library':<14} {'SOTA(ms)':<10} {'Ref(ms)':<10} {'Speedup':<8}")
    print(f"{'-'*90}")
    for r in sorted(all_results, key=lambda x: x["avg_speedup"], reverse=True):
        print(f"{r['task']:<40} {r['library']:<14} {r['avg_sota_ms']:<10.4f} "
              f"{r['avg_ref_ms']:<10.4f} {r['avg_speedup']:<8.1f}x")
    print(f"{'='*90}")

    # Print per-workload details for each task
    print(f"\n{'='*90}")
    print("  Per-workload details")
    print(f"{'='*90}")
    for r in all_results:
        print(f"\n--- {r['task']} ({r['library']}) ---")
        for i, d in enumerate(r["details"]):
            if i >= 3:  # Show first 3 workloads
                print(f"  ... ({len(r['details'])-3} more workloads)")
                break
            print(f"  {d['axes']}: SOTA={d['sota_ms']:.4f}ms, "
                  f"Ref={d['ref_ms']:.4f}ms, Speedup={d['speedup']:.1f}x")


if __name__ == "__main__":
    main()