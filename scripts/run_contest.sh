#!/usr/bin/env bash
set -u
cd /home/qinhaiyan/sol-execbench
TASKS=(
  "Contest/L1/069_rms_norm|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_L1_069_rms_norm/solution.json|full"
  "Contest/L1/048_fused_gate_up_projection_with_swiglu|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_L1_048_swiglu/solution.json|full"
  "Contest/L1/044_moe_expert_computation|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L1_044_moe_sorted/solution.json|fast"
  "Contest/Quant/002_fp8_attention_qkv_projection|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_002_fp8_qkv/solution.json|full"
  "Contest/Quant/003_fp8_mlp_gate_up_projection|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_003_fp8_mlp/solution.json|full"
  "Contest/Quant/004_fp8_moe_expert_linear|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_004_fp8_moe_expert/solution.json|full"
  "Contest/Quant/005_fp8_moe_router_projection|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_005_fp8_router/solution.json|full"
  "Contest/Quant/011_fp8_moe_gate_routing|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_011_fp8_routing/solution.json|full"
  "Contest/Quant/012_fp8_shared_expert_mlp|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_012_fp8_shared_expert/solution.json|full"
  "Contest/Quant/013_fp8_mla_kv_compression_projection|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_013_fp8_mla_kv/solution.json|full"
  "Contest/Quant/015_fp8_mla_attention_output_projection|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_015_fp8_mla_oproj/solution.json|full"
  "Contest/Quant/016_fp8_multi_latent_attention_qkv_projection|/home/qinhaiyan/sol-baseline/baselines/flashinfer/Contest/Contest_Quant_016_fp8_mla_qkv/solution.json|full"
  "Contest/L2/008_moe_sparse_routing_and_dispatch|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_008_moe_routing/solution.json|fast"
  "Contest/L2/010_moe_expert_computation_with_weighted_accumulation|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_010_moe/solution.json|fast"
  "Contest/L2/013_expert_weighted_aggregation_with_shared_expert|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_013_moe_shared/solution.json|fast"
  "Contest/L2/029_moe_sparse_routing_and_dispatch|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_029_moe_ernie/solution.json|fast"
  "Contest/L2/065_sparse_expert_dispatch_and_combine|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_065_gptoss_moe/solution.json|fast"
  "Contest/L2/081_moe_sparse_expert_dispatch|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_081_moe_routing/solution.json|fast"
  "Contest/L2/082_moe_layer_complete_forward_with_residual|/home/qinhaiyan/sol-baseline/baselines/torch/Contest/Contest_L2_082_moe_complete/solution.json|fast"
)
printf 'task,library,passed,total,mean_x,min_x,max_x\n'
for entry in "${TASKS[@]}"; do
  IFS="|" read -r t sol kind <<< "$entry"
  cfg=/home/qinhaiyan/sol-baseline/scripts/bench_config.json
  [ "$kind" = "fast" ] && cfg=/home/qinhaiyan/sol-baseline/scripts/bench_config_fast.json
  lib=$(basename $(dirname $(dirname "$sol")))
  T="$t" L="$lib" timeout 600 uv run --no-sync sol-execbench "data/benchmark/$t" --solution "$sol" --config "$cfg" --json 2>/dev/null | T="$t" L="$lib" python3 -c "
import sys, json, statistics as s, os
task=os.environ['T']; lib=os.environ['L']
lines=[l for l in sys.stdin if l.strip().startswith('{')]
pas=0; sp=[]
for L in lines:
    try: d=json.loads(L)
    except: continue
    ev=d.get('evaluation') or {}
    if ev.get('status')=='PASSED': pas+=1
    p=ev.get('performance') or {}
    r=p.get('reference_latency_ms',0) or 0
    l=p.get('latency_ms',0) or 0
    if r>0 and l>0: sp.append(r/l)
if sp:
    print(f'{task},{lib},{pas},{len(lines)},{s.mean(sp):.2f}x,{min(sp):.2f}x,{max(sp):.2f}x')
else:
    print(f'{task},{lib},{pas},{len(lines)},N/A,N/A,N/A')
"
done
