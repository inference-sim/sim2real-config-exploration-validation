#!/usr/bin/env python3
"""Analyze sim2real results: compare predicted vs actual TTFT and throughput."""

import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
SELECTION_FILE = Path(__file__).parents[3] / "results-70b" / "processed" / "top3_selection.json"
VALIDATION_FILE = Path(__file__).parents[3] / "results-70b" / "validated" / "validation.jsonl"
OUTPUT_DIR = Path(__file__).parent / "figures"

PACKAGE_TO_SELECTION = {
    "blis1": ("blis", 0),
    "aiconfigurator2": ("aiconfigurator", 1),
    "aiconfigurator3": ("aiconfigurator", 2),
    "llmoptimizer3": ("llm-optimizer", 2),
}

FAILED_DEPLOYMENTS = [
    ("1d065cac", "aiconfigurator", "OOM: TP=2 insufficient for 70B on H100"),
    ("2d78fecf", "llm-optimizer", "OOM: TP=2 insufficient for 70B on H100"),
    ("0da30292", "llm-optimizer", "OOM: TP=3 insufficient for 70B on H100"),
]

TOOL_LABELS = {
    "blis": "BLIS",
    "aiconfigurator": "AIConfigurator",
    "llm-optimizer": "llm-optimizer",
}

TOOL_COLORS = {
    "blis": "#1f77b4",
    "aiconfigurator": "#2ca02c",
    "llm-optimizer": "#9467bd",
}


def parse_trace(trace_path: Path) -> dict:
    """Parse trace_data.csv and compute actual metrics."""
    ttfts = []
    tpots = []
    e2e_ms_list = []
    total_output_tokens = 0
    first_send_us = None
    last_finish_us = None

    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok":
                continue
            send_us = int(row["send_time_us"])
            first_chunk_us = int(row["first_chunk_time_us"])
            last_chunk_us = int(row["last_chunk_time_us"])
            output_tokens = int(row["output_tokens"])

            e2e_ms = (last_chunk_us - send_us) / 1000.0
            ttfts.append((first_chunk_us - send_us) / 1000.0)
            tpots.append((last_chunk_us - first_chunk_us) / 1000.0 / max(output_tokens - 1, 1))
            e2e_ms_list.append(e2e_ms)
            total_output_tokens += output_tokens

            if first_send_us is None or send_us < first_send_us:
                first_send_us = send_us
            if last_finish_us is None or last_chunk_us > last_finish_us:
                last_finish_us = last_chunk_us

    if not ttfts:
        return {}

    e2e_s = (last_finish_us - first_send_us) / 1e6
    throughput = total_output_tokens / e2e_s if e2e_s > 0 else 0
    # Per-request token rate: output_tokens / e2e_time
    total_e2e_s = sum(e2e_ms_list) / 1000.0
    token_rate = total_output_tokens / total_e2e_s if total_e2e_s > 0 else 0

    return {
        "ttft_mean_ms": np.mean(ttfts),
        "ttft_p50_ms": np.percentile(ttfts, 50),
        "ttft_p99_ms": np.percentile(ttfts, 99),
        "tpot_mean_ms": np.mean(tpots),
        "throughput_tok_s": throughput,
        "token_rate_tok_s": token_rate,
        "num_requests": len(ttfts),
    }


def load_predictions() -> dict:
    """Load predicted metrics from top3_selection.json."""
    with open(SELECTION_FILE) as f:
        data = json.load(f)

    preds = {}
    for pkg_name, (group, idx) in PACKAGE_TO_SELECTION.items():
        entry = data[group][idx]
        preds[pkg_name] = {
            "tool": group,
            "config_hash": entry["metadata"]["config_hash"],
            "tp": entry["vllm_args"]["tensor_parallel_size"],
            "replicas": entry["vllm_args"]["num_replicas"],
            "pred_ttft_mean_ms": entry["results"]["ttft_mean_ms"],
            "pred_throughput_tok_s": entry["results"]["max_throughput_tok_s"],
            "pred_tpot_mean_ms": entry["results"].get("tpot_mean_ms"),
            "cost_per_hour": entry["results"]["cost_per_hour"],
        }
    return preds


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    results = []

    for pkg_name in PACKAGE_TO_SELECTION:
        trace_path = RESULTS_DIR / pkg_name / "chatbot" / "trace_data.csv"
        if not trace_path.exists():
            print(f"  [skip] {pkg_name}: no trace data")
            continue

        actual = parse_trace(trace_path)
        pred = predictions[pkg_name]

        results.append({
            "package": pkg_name,
            "config_hash": pred["config_hash"],
            "tool": pred["tool"],
            "tp": pred["tp"],
            "replicas": pred["replicas"],
            "pred_ttft": pred["pred_ttft_mean_ms"],
            "actual_ttft": actual["ttft_mean_ms"],
            "pred_throughput": pred["pred_throughput_tok_s"],
            "actual_throughput": actual["throughput_tok_s"],
            "actual_token_rate": actual["token_rate_tok_s"],
            "pred_tpot": pred["pred_tpot_mean_ms"],
            "actual_tpot": actual["tpot_mean_ms"],
            "cost": pred["cost_per_hour"],
            "actual_num_requests": actual["num_requests"],
        })

        print(f"  {pkg_name} ({TOOL_LABELS[pred['tool']]}, TP={pred['tp']}x{pred['replicas']}):")
        print(f"    TTFT:       pred={pred['pred_ttft_mean_ms']:.1f}ms  actual={actual['ttft_mean_ms']:.1f}ms  "
              f"drift={actual['ttft_mean_ms'] - pred['pred_ttft_mean_ms']:+.1f}ms")
        print(f"    Throughput: pred={pred['pred_throughput_tok_s']:.0f}  actual={actual['throughput_tok_s']:.0f} tok/s  "
              f"ratio={actual['throughput_tok_s'] / pred['pred_throughput_tok_s']:.2f}x")
        if pred["pred_tpot_mean_ms"]:
            print(f"    TPOT:       pred={pred['pred_tpot_mean_ms']:.1f}ms  actual={actual['tpot_mean_ms']:.1f}ms")

    if not results:
        print("No results to plot.")
        return

    # Save summary JSON
    summary_path = OUTPUT_DIR / "sim2real_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save validation.jsonl for chart1
    VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_FILE, "w") as f:
        for r in results:
            f.write(json.dumps({
                "config_hash": r["config_hash"],
                "package": r["package"],
                "tool": r["tool"],
                "actual_ttft_mean_ms": r["actual_ttft"],
                "actual_throughput_tok_s": r["actual_throughput"],
                "actual_token_rate_tok_s": r["actual_token_rate"],
                "num_requests": r["actual_num_requests"],
                "status": "ok",
            }) + "\n")
        # Configs that failed to deploy (OOM: TP<4 for 70B on H100)
        for cfg_hash, tool, reason in FAILED_DEPLOYMENTS:
            f.write(json.dumps({
                "config_hash": cfg_hash,
                "tool": tool,
                "status": "deploy_failed",
                "reason": reason,
            }) + "\n")
    print(f"Validation saved to {VALIDATION_FILE}")

    # Filter out degenerate results (TTFT > 10s means system was overloaded)
    plot_results = [r for r in results if r["actual_ttft"] < 10000]
    if len(plot_results) < len(results):
        excluded = [r for r in results if r["actual_ttft"] >= 10000]
        for r in excluded:
            print(f"  [excluded from plot] {r['package']}: actual TTFT={r['actual_ttft']:.0f}ms (overloaded)")

    # Plot: Pareto frontier style (TTFT vs Throughput) with drift arrows
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by tool for Pareto frontiers
    by_tool = {}
    for r in plot_results:
        by_tool.setdefault(r["tool"], []).append(r)

    seen_labels = set()
    for tool, entries in by_tool.items():
        color = TOOL_COLORS[tool]
        label = TOOL_LABELS[tool]

        # Sort predicted points by TTFT for Pareto frontier line
        pred_points = sorted(entries, key=lambda r: r["pred_ttft"])
        pred_ttft = [r["pred_ttft"] for r in pred_points]
        pred_tput = [r["pred_throughput"] for r in pred_points]

        # Predicted Pareto frontier
        ax.plot(pred_ttft, pred_tput, color=color, linewidth=2, alpha=0.7, zorder=3)
        ax.scatter(pred_ttft, pred_tput, facecolors="none", edgecolors=color,
                   s=120, linewidths=2, marker="o", zorder=4, label=f"{label} (predicted)")

        # Actual points
        actual_ttft = [r["actual_ttft"] for r in pred_points]
        actual_tput = [r["actual_throughput"] for r in pred_points]
        ax.scatter(actual_ttft, actual_tput, color=color, s=80, marker="x",
                   linewidths=2, zorder=5, label=f"{label} (actual)")

        # Drift arrows from predicted to actual
        for r in entries:
            ax.annotate(
                "", xy=(r["actual_ttft"], r["actual_throughput"]),
                xytext=(r["pred_ttft"], r["pred_throughput"]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6),
                zorder=2,
            )
            ax.annotate(f"TP{r['tp']}x{r['replicas']}",
                        (r["pred_ttft"], r["pred_throughput"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8, color=color)

    # SLO line
    ax.axvline(300, color="red", linestyle=":", linewidth=2, alpha=0.7, label="SLO: TTFT < 300ms")

    ax.set_xscale("log")
    ax.set_xlabel("Mean TTFT (ms, log scale)", fontsize=13)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=13)
    ax.set_title("Sim2Real: Predicted vs Actual Pareto (Llama 3.1 70B, H100, chatbot)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    png_path = OUTPUT_DIR / "sim2real_pareto.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "sim2real_pareto.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Pareto plot saved to {png_path}")


if __name__ == "__main__":
    main()
