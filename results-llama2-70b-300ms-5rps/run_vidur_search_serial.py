#!/usr/bin/env python3
"""Run Vidur config search for Llama-2-70b on H100 (serial, no timeout).

Aligned with experiments/llama2_70b_config.yaml concept_map.
Runs TP=4 and TP=8 configs only; TP=2 confirmed infeasible (timeout at 5 rps).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

VIDUR_ROOT = Path(__file__).resolve().parent.parent / "estimators" / "vidur"
VIDUR_PYTHON = str(VIDUR_ROOT / ".venv" / "bin" / "python")
TRACE_FILE = str(Path(__file__).resolve().parent / "vidur_trace_shared_prefix_codegen.csv")
OUTPUT_BASE = "/tmp/vidur_sweep"

# Aligned with experiments/llama2_70b_config.yaml
CONFIGS = {
    "model": "meta-llama/Llama-2-70b-hf",
    "device": "h100",
    "max_tokens": 4096,       # concept_map.max_model_len
    "num_requests": 200,      # deterministic sim; steady-state reached by ~100 reqs
    "qps": 5.0,              # concept_map.tools.blis.search.fixed_rate
    "seed": 42,              # concept_map.seed
    "slo_ttft_ms": 300,      # concept_map.slo_ttft_ms
    "gpu_budget": 8,         # concept_map.max_gpus
}

# concept_map.tp.value = [2, 4, 8]
# TP=2 all timeout (Vidur cannot simulate 5 rps at TP=2 within reasonable time)
TP_SIZES = [4, 8]

# concept_map.max_num_seqs.value = [32, 64, 128, 256, 512]
BATCH_SIZES = [32, 64, 128, 256, 512]


def run_vidur(tp, batch_size, replicas):
    """Run a single Vidur simulation and return metrics."""
    run_id = f"tp{tp}_rep{replicas}_bs{batch_size}"
    output_dir = f"{OUTPUT_BASE}/{run_id}"

    cmd = [
        VIDUR_PYTHON, "-m", "vidur.main",
        "--replica_config_device", CONFIGS["device"],
        "--replica_config_model_name", CONFIGS["model"],
        "--cluster_config_num_replicas", str(replicas),
        "--replica_config_tensor_parallel_size", str(tp),
        "--replica_config_num_pipeline_stages", "1",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(CONFIGS["num_requests"]),
        "--length_generator_config_type", "trace",
        "--trace_request_length_generator_config_max_tokens", str(CONFIGS["max_tokens"]),
        "--trace_request_length_generator_config_trace_file", TRACE_FILE,
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(CONFIGS["qps"]),
        "--replica_scheduler_config_type", "vllm",
        "--vllm_scheduler_config_batch_size_cap", str(batch_size),
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size", str(CONFIGS["max_tokens"]),
        "--random_forrest_execution_time_predictor_config_prediction_max_batch_size", str(batch_size),
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request", str(CONFIGS["max_tokens"]),
        "--metrics_config_output_dir", output_dir,
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    sys.stdout.write(f"  {run_id}... ")
    sys.stdout.flush()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(VIDUR_ROOT), env=env
        )
        if result.returncode != 0:
            print(f"FAILED (rc={result.returncode})")
            return None
    except subprocess.TimeoutExpired:
        print("TIMEOUT (>10min)")
        return None

    # Find output directory (timestamped)
    try:
        run_dirs = sorted(Path(output_dir).iterdir())
        if not run_dirs:
            print("NO OUTPUT")
            return None
        latest = run_dirs[-1]
    except FileNotFoundError:
        print("NO OUTPUT DIR")
        return None

    # Read metrics
    try:
        metrics_file = latest / "request_metrics.csv"
        df = pd.read_csv(metrics_file)

        ttft_mean = df["prefill_e2e_time"].mean()
        ttft_p99 = df["prefill_e2e_time"].quantile(0.99)
        sched_delay_mean = df["request_scheduling_delay"].mean()
        e2e_mean = df["request_e2e_time"].mean()
        completed = len(df)
        tput = CONFIGS["qps"]  # effective throughput = arrival rate if all complete

        slo_status = "PASS" if ttft_mean * 1000 <= CONFIGS["slo_ttft_ms"] else "FAIL"
        print(f"TTFT={ttft_mean*1000:.0f}ms, sched={sched_delay_mean*1000:.0f}ms [{slo_status}]")

        return {
            "run_id": run_id,
            "tp": tp,
            "replicas": replicas,
            "gpus": tp * replicas,
            "scheduler": "vllm",
            "batch_size": batch_size,
            "ttft_mean_ms": round(ttft_mean * 1000, 1),
            "ttft_p99_ms": round(ttft_p99 * 1000, 1),
            "scheduling_delay_mean_ms": round(sched_delay_mean * 1000, 1),
            "e2e_mean_ms": round(e2e_mean * 1000, 1),
            "throughput_qps": tput,
            "completed": completed,
            "num_requests": CONFIGS["num_requests"],
            "slo_met": bool(ttft_mean * 1000 <= CONFIGS["slo_ttft_ms"]),
        }
    except Exception as e:
        print(f"PARSE ERROR: {e}")
        return None


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    gpu_budget = CONFIGS["gpu_budget"]

    # Build job list
    jobs = []
    for tp in TP_SIZES:
        max_replicas = gpu_budget // tp
        for replicas in range(1, max_replicas + 1):
            for batch_size in BATCH_SIZES:
                jobs.append((tp, batch_size, replicas))

    print(f"Vidur config search: {CONFIGS['model']} on {CONFIGS['device']}")
    print(f"Workload: ISL=566, OSL=247, constant {CONFIGS['qps']} rps, {CONFIGS['num_requests']} requests")
    print(f"Search space: TP={TP_SIZES}, batch={BATCH_SIZES}, vllm scheduler")
    print(f"Configs to evaluate: {len(jobs)} (TP=2 excluded: all timeout at 5 rps)")
    print(f"Note: TP=2 x {gpu_budget//2} replicas configs all timeout (>10min)")
    print(f"      because Vidur's pre-chunked-prefill model cannot sustain 5 rps at TP=2.")
    print()

    results = []
    for i, (tp, batch_size, replicas) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}]", end="")
        r = run_vidur(tp, batch_size, replicas)
        if r:
            results.append(r)

    # Also record TP=2 as infeasible
    for replicas in range(1, gpu_budget // 2 + 1):
        for batch_size in BATCH_SIZES:
            results.append({
                "run_id": f"tp2_rep{replicas}_bs{batch_size}",
                "tp": 2,
                "replicas": replicas,
                "gpus": 2 * replicas,
                "scheduler": "vllm",
                "batch_size": batch_size,
                "ttft_mean_ms": None,
                "ttft_p99_ms": None,
                "scheduling_delay_mean_ms": None,
                "e2e_mean_ms": None,
                "throughput_qps": 0,
                "completed": 0,
                "num_requests": CONFIGS["num_requests"],
                "slo_met": False,
                "note": "timeout: Vidur cannot simulate 5 rps at TP=2 (pre-chunked-prefill scheduling)",
            })

    # Save results
    out_file = str(Path(__file__).resolve().parent / "raw" / "vidur.jsonl")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    completed_runs = [r for r in results if r.get("ttft_mean_ms") is not None]
    print(f"\nDone. {len(completed_runs)} configs completed, {len(results) - len(completed_runs)} infeasible (timeout).")
    print(f"Results: {out_file}")

    # Print summary
    if completed_runs:
        df = pd.DataFrame(completed_runs)
        print("\n--- All completed configs (sorted by TTFT) ---")
        df_sorted = df.sort_values("ttft_mean_ms")
        for _, row in df_sorted.iterrows():
            slo = "PASS" if row["slo_met"] else "FAIL"
            print(f"  TP={row['tp']}x{row['replicas']} ({row['gpus']} GPUs) "
                  f"batch={row['batch_size']:>3}: "
                  f"TTFT={row['ttft_mean_ms']:>8.0f}ms [{slo}]")


if __name__ == "__main__":
    main()
