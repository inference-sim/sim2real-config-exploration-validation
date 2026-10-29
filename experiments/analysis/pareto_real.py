"""Pareto front analysis for real trace data from config-explore-70b-codegen runs.

Reads TraceV3 CSV files, computes per-config metrics (throughput, latency percentiles),
and outputs a comparison table plus Pareto front identification.
"""

import csv
import statistics
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "raw" / "config-explore-70b-codegen"
GPU_COST_PER_HOUR = 3.20
SLO_TTFT_MS = 300

CONFIGS = {
    "blis2": {
        "label": "BLIS #2",
        "estimator": "BLIS",
        "tp": 4, "replicas": 1, "max_num_seqs": 256,
        "max_batched_tokens": 8192, "chunked_prefill": True,
        "prefix_caching": False,
    },
    "blis3": {
        "label": "BLIS #3",
        "estimator": "BLIS",
        "tp": 8, "replicas": 1, "max_num_seqs": 256,
        "max_batched_tokens": 8192, "chunked_prefill": True,
        "prefix_caching": False,
    },
    "aiconfigurator2": {
        "label": "AIConfigurator #2",
        "estimator": "AIConfigurator",
        "tp": 4, "replicas": 1, "max_num_seqs": 128,
        "max_batched_tokens": None, "chunked_prefill": True,
        "prefix_caching": True,
    },
    "aiconfigurator3": {
        "label": "AIConfigurator #3",
        "estimator": "AIConfigurator",
        "tp": 4, "replicas": 2, "max_num_seqs": 80,
        "max_batched_tokens": 566, "chunked_prefill": False,
        "prefix_caching": False,
    },
    "llmoptimizer3": {
        "label": "LLM-Optimizer #3",
        "estimator": "LLM-Optimizer",
        "tp": 4, "replicas": 1, "max_num_seqs": 8,
        "max_batched_tokens": None, "chunked_prefill": False,
        "prefix_caching": False,
    },
}


def percentile(data: list[float], p: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def load_trace(name: str) -> list[dict]:
    path = RESULTS_DIR / f"{name}-trace_data.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "ok":
                rows.append(row)
    return rows


def compute_metrics(rows: list[dict], config: dict) -> dict:
    num_gpus = config["tp"] * config["replicas"]
    cost_per_hour = num_gpus * GPU_COST_PER_HOUR

    ttft_ms_list = []
    e2e_ms_list = []
    tpot_ms_list = []
    total_output_tokens = 0
    send_times = []
    last_chunk_times = []

    for row in rows:
        send_t = int(row["send_time_us"])
        first_chunk_t = int(row["first_chunk_time_us"])
        last_chunk_t = int(row["last_chunk_time_us"])
        output_tokens = int(row["output_tokens"])

        ttft_us = first_chunk_t - send_t
        e2e_us = last_chunk_t - send_t
        ttft_ms_list.append(ttft_us / 1000.0)
        e2e_ms_list.append(e2e_us / 1000.0)

        if output_tokens > 1:
            tpot_us = (last_chunk_t - first_chunk_t) / (output_tokens - 1)
            tpot_ms_list.append(tpot_us / 1000.0)

        total_output_tokens += output_tokens
        send_times.append(send_t)
        last_chunk_times.append(last_chunk_t)

    duration_s = (max(last_chunk_times) - min(send_times)) / 1e6
    throughput_tok_s = total_output_tokens / duration_s
    throughput_req_s = len(rows) / duration_s

    ttft_p50 = percentile(ttft_ms_list, 0.50)
    ttft_p99 = percentile(ttft_ms_list, 0.99)
    e2e_p50 = percentile(e2e_ms_list, 0.50)
    e2e_p99 = percentile(e2e_ms_list, 0.99)
    tpot_p50 = percentile(tpot_ms_list, 0.50) if tpot_ms_list else 0
    tpot_p99 = percentile(tpot_ms_list, 0.99) if tpot_ms_list else 0

    meets_slo = ttft_p99 <= SLO_TTFT_MS

    return {
        "label": config["label"],
        "estimator": config["estimator"],
        "num_gpus": num_gpus,
        "cost_per_hour": cost_per_hour,
        "throughput_tok_s": throughput_tok_s,
        "throughput_req_s": throughput_req_s,
        "ttft_p50_ms": ttft_p50,
        "ttft_p99_ms": ttft_p99,
        "e2e_p50_ms": e2e_p50,
        "e2e_p99_ms": e2e_p99,
        "tpot_p50_ms": tpot_p50,
        "tpot_p99_ms": tpot_p99,
        "meets_slo": meets_slo,
        "num_requests": len(rows),
        "duration_s": duration_s,
        "config": config,
    }


def compute_pareto_front(metrics: list[dict]) -> list[dict]:
    """Pareto front: minimize cost, maximize throughput (among SLO-meeting configs)."""
    viable = [m for m in metrics if m["meets_slo"]]
    if not viable:
        print("  (No configs meet SLO; showing all)")
        viable = metrics

    viable.sort(key=lambda m: (m["cost_per_hour"], -m["throughput_tok_s"]))
    front = []
    max_tp = float("-inf")
    for m in viable:
        if m["throughput_tok_s"] > max_tp:
            front.append(m)
            max_tp = m["throughput_tok_s"]
    return front


def main():
    all_metrics = []
    for name, config in CONFIGS.items():
        trace_path = RESULTS_DIR / f"{name}-trace_data.csv"
        if not trace_path.exists():
            print(f"Skipping {name}: no trace file")
            continue
        rows = load_trace(name)
        m = compute_metrics(rows, config)
        all_metrics.append(m)
        print(f"\n{'='*60}")
        print(f" {m['label']} ({m['estimator']})")
        print(f"{'='*60}")
        print(f"  GPUs: {m['num_gpus']}  Cost: ${m['cost_per_hour']:.2f}/hr")
        print(f"  Throughput: {m['throughput_tok_s']:.1f} tok/s  ({m['throughput_req_s']:.2f} req/s)")
        print(f"  TTFT:  p50={m['ttft_p50_ms']:.1f}ms  p99={m['ttft_p99_ms']:.1f}ms")
        print(f"  E2E:   p50={m['e2e_p50_ms']:.1f}ms  p99={m['e2e_p99_ms']:.1f}ms")
        print(f"  TPOT:  p50={m['tpot_p50_ms']:.2f}ms  p99={m['tpot_p99_ms']:.2f}ms")
        print(f"  Meets SLO (TTFT p99 < {SLO_TTFT_MS}ms): {m['meets_slo']}")
        print(f"  Requests: {m['num_requests']}  Duration: {m['duration_s']:.1f}s")

    front = compute_pareto_front(all_metrics)
    print(f"\n{'='*60}")
    print(f" PARETO FRONT ({len(front)} configs)")
    print(f"{'='*60}")
    for m in front:
        slo_str = "MEETS SLO" if m["meets_slo"] else "VIOLATES SLO"
        print(f"  {m['label']}: ${m['cost_per_hour']:.2f}/hr, "
              f"{m['throughput_tok_s']:.1f} tok/s, "
              f"TTFT p99={m['ttft_p99_ms']:.0f}ms [{slo_str}]")

    print(f"\n{'='*60}")
    print(f" COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Tput(tok/s)':<12} {'TTFT p99':<10} {'E2E p99':<10} {'TPOT p50':<10} {'SLO':<6} {'Cost':<8}")
    print("-" * 76)
    for m in all_metrics:
        slo = "OK" if m["meets_slo"] else "FAIL"
        print(f"{m['label']:<20} {m['throughput_tok_s']:<12.1f} {m['ttft_p99_ms']:<10.1f} "
              f"{m['e2e_p99_ms']:<10.1f} {m['tpot_p50_ms']:<10.2f} {slo:<6} ${m['cost_per_hour']:.2f}/hr")

    summary = [{k: v for k, v in m.items() if k != "config"} for m in all_metrics]
    summary_path = RESULTS_DIR / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
