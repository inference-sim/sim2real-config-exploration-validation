#!/usr/bin/env python3
"""Run Vidur config search for Llama-2-70b on H100.

Sweeps the same config space as other tools: TP={2,4,8}, batch sizes,
scheduler types. Collects predicted TTFT and throughput for each config.
"""

import json
import os
import subprocess
import sys
import itertools
from pathlib import Path

VIDUR_ROOT = Path(__file__).resolve().parent.parent / "estimators" / "vidur"
VIDUR_PYTHON = str(VIDUR_ROOT / ".venv" / "bin" / "python")
TRACE_FILE = str(Path(__file__).resolve().parent / "vidur_trace_shared_prefix_codegen.csv")
OUTPUT_BASE = "/tmp/vidur_sweep"

# Aligned with experiments/llama2_70b_config.yaml concept_map
CONFIGS = {
    "model": "meta-llama/Llama-2-70b-hf",
    "device": "h100",
    "max_tokens": 4096,       # concept_map.max_model_len
    "num_requests": 200,      # sufficient for deterministic simulator steady-state
    "qps": 5.0,              # concept_map.tools.blis.search.fixed_rate
    "seed": 42,              # concept_map.seed
    "slo_ttft_ms": 300,      # concept_map.slo_ttft_ms
    "gpu_budget": 8,         # concept_map.max_gpus
}

# Search space from concept_map
# concept_map.tp.value = [2, 4, 8]
TP_SIZES = [2, 4, 8]

# Vidur vllm scheduler only (pre-chunked-prefill scheduling semantics)
SCHEDULERS = [
    {"type": "vllm", "batch_cap_key": "vllm_scheduler_config_batch_size_cap"},
]

# concept_map.max_num_seqs.value = [32, 64, 128, 256, 512]
BATCH_SIZES = [32, 64, 128, 256, 512]

TIMEOUT_S = 120  # seconds per simulation (200 reqs, TP=4 ~1.5min)


def run_vidur(tp, scheduler, batch_size, replicas):
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
        "--replica_scheduler_config_type", scheduler["type"],
        f"--{scheduler['batch_cap_key']}", str(batch_size),
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size", str(CONFIGS["max_tokens"]),
        "--random_forrest_execution_time_predictor_config_prediction_max_batch_size", str(batch_size),
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request", str(CONFIGS["max_tokens"]),
        "--metrics_config_output_dir", output_dir,
    ]


    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    print(f"  Running {run_id}...", end=" ", flush=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
            cwd=str(VIDUR_ROOT), env=env
        )
        if result.returncode != 0:
            print(f"FAILED (rc={result.returncode})")
            stderr_last = result.stderr.strip().split("\n")[-3:] if result.stderr else []
            for line in stderr_last:
                print(f"    {line}")
            return None
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
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
    import pandas as pd
    try:
        metrics_file = latest / "request_metrics.csv"
        df = pd.read_csv(metrics_file)

        ttft_mean = df["prefill_e2e_time"].mean()
        ttft_p99 = df["prefill_e2e_time"].quantile(0.99)
        sched_delay_mean = df["request_scheduling_delay"].mean()
        e2e_mean = df["request_e2e_time"].mean()
        completed = len(df)

        # Throughput: requests / total_time
        total_time = df["request_e2e_time"].max() + df["request_inter_arrival_delay"].iloc[0]
        # Better: use actual simulation time from arrivals
        tput = completed / (completed / CONFIGS["qps"])  # effective = arrival rate if all complete

        print(f"TTFT={ttft_mean*1000:.1f}ms, sched_delay={sched_delay_mean*1000:.1f}ms")

        return {
            "run_id": run_id,
            "tp": tp,
            "replicas": replicas,
            "gpus": tp * replicas,
            "scheduler": scheduler["type"],
            "chunk_size": scheduler.get("chunk_size"),
            "batch_size": batch_size,
            "ttft_mean_ms": ttft_mean * 1000,
            "ttft_p99_ms": ttft_p99 * 1000,
            "scheduling_delay_mean_ms": sched_delay_mean * 1000,
            "e2e_mean_ms": e2e_mean * 1000,
            "throughput_qps": tput,
            "completed": completed,
            "num_requests": CONFIGS["num_requests"],
        }
    except Exception as e:
        print(f"PARSE ERROR: {e}")
        return None


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    gpu_budget = CONFIGS["gpu_budget"]

    print(f"Vidur config search: Llama-2-70b on H100, {CONFIGS['qps']} rps")
    print(f"Search space: TP={TP_SIZES}, schedulers={[s['type'] for s in SCHEDULERS]}, batch={BATCH_SIZES}")
    print()

    # Build job list
    jobs = []
    for tp in TP_SIZES:
        max_replicas = gpu_budget // tp
        for replicas in range(1, max_replicas + 1):
            for scheduler in SCHEDULERS:
                for batch_size in BATCH_SIZES:
                    jobs.append((tp, scheduler, batch_size, replicas))

    total = len(jobs)
    print(f"Total configs to evaluate: {total}")
    print()

    # Run with parallelism (4 workers to avoid CPU thrashing)
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_vidur, *job): job for job in jobs}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            r = future.result()
            if r:
                results.append(r)
            if done_count % 10 == 0:
                print(f"  Progress: {done_count}/{total} done, {len(results)} successful")

    print(f"\nAll done: {done_count}/{total} attempted, {len(results)} successful")

    # Save results
    out_file = str(Path(__file__).resolve().parent / "raw" / "vidur.jsonl")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(results)}/{total} configs completed.")
    print(f"Results: {out_file}")

    # Print summary
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n--- Top configs by TTFT (lower is better) ---")
        top = df.nsmallest(10, "ttft_mean_ms")
        for _, row in top.iterrows():
            slo = "PASS" if row["ttft_mean_ms"] <= 300 else "FAIL"
            print(f"  TP={row['tp']}x{row['replicas']} ({row['gpus']} GPUs) "
                  f"{row['scheduler']} bs={row['batch_size']}: "
                  f"TTFT={row['ttft_mean_ms']:.0f}ms [{slo}]")


if __name__ == "__main__":
    main()
