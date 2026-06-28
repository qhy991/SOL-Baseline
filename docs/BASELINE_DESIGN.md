# Baseline Design Notes

## How Baselines Work

Each baseline is a `solution.json` file that wraps a SOTA library function. The file contains:

1. **Metadata**: name, author, description
2. **Spec**: language (pytorch), entry point (baseline.py::run), calling convention
3. **Source**: the inline Python code with a `run()` function matching the task's reference signature

The sol-execbench eval driver:
1. Loads the baseline `solution.json`
2. Extracts the source code and writes it to a staging directory
3. Executes the `run()` function with task-generated inputs
4. Compares output against the reference implementation
5. Records correctness and performance metrics

## Matching Strategy

### Direct Match (FlashInfer-Bench)
FlashInfer-Bench tasks are explicitly designed to benchmark FlashInfer APIs. The reference implementation is a direct torch equivalent of the FlashInfer function. The baseline simply calls the FlashInfer function with the same inputs.

### Wrapper Match (L1 Tasks)
L1 tasks are decomposed kernel patterns from real models. The baseline needs to:
1. Accept the same arguments as the reference
2. Perform any necessary preprocessing (dtype conversion, reshape, splitting)
3. Call the SOTA library function
4. Perform any necessary postprocessing
5. Return results matching the reference output signature

### Why Some Tasks Cannot Be Matched

1. **Backward passes**: SOTA libraries generally don't provide backward kernels
2. **Fused multi-operation patterns**: Tasks that fuse QKV projection + RoPE + Attention cannot be matched to FlashAttention (which expects pre-projected Q, K, V)
3. **API convention mismatch**: Liger RoPE uses a different rotation convention than the standard cos/sin-based RoPE
4. **Dtype limitations**: FlashInfer GEMM only supports bfloat16, but GEMM tasks use float16
5. **External data dependencies**: Some tasks require safetensors files from external datasets

## File Format

Each `solution.json` follows the sol-execbench Solution format:

```json
{
  "name": "baseline_unique_name",
  "definition": "task_definition_name",
  "author": "library_name",
  "description": "Description of what this baseline does",
  "spec": {
    "languages": ["pytorch"],
    "target_hardware": ["LOCAL"],
    "entry_point": "baseline.py::run",
    "destination_passing_style": false
  },
  "sources": [
    {
      "path": "baseline.py",
      "content": "import torch\n...\ndef run(...):\n    ..."
    }
  ]
}
```

## Adding New Baselines

1. Understand the task's reference implementation
2. Identify the core computation that matches a SOTA library function
3. Write a wrapper `run()` function with the same signature
4. Test with `uv run sol-execbench <problem_dir> --solution <solution.json> --json`
5. Verify all workloads pass correctness checks
6. Benchmark with `scripts/benchmark.py`