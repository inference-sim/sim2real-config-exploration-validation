#!/usr/bin/env python3
"""Generate pareto chart (TTFT vs Throughput) with predicted and actual results."""

import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parents[1] / "raw"
RESULTS_DIR = Path(__file__).parents[2] / "sim2real" / "workspace" / "runs" / "config-explore-run1" / "results"
OUTPUT_DIR = Path(__file__).parent

TOOL_STYLES = {
    "blis": {"color": "#1f77b4", "marker": "o", "label": "BLIS (inference-sim)"},
    "aiconfigurator": {"color": "#ff7f0e", "marker": "D", "label": "AIConfigurator"},
    "llm-optimizer": {"color": "#2ca02c", "marker": "s", "label": "llm-optimizer"},
}


def load_predictions():
    predictions = {}
    for tool_file in ["blis", "aiconfigurator", "llm-optimizer"]:
        path = RAW_DIR / f"{tool_file}.jsonl"
        if not path.exists():
            continue
        entries = []
        seen_hashes = set()
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                h = d["metadata"]["config_hash"]
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                tp = d["vllm_args"]["tensor_parallel_size"]
                replicas = d["vllm_args"].get("num_replicas", 1)
                entries.append({
                    "tool": tool_file,
                    "hash": h,
                    "tp": tp,
                    "replicas": replicas,
                    "gpus": tp * replicas,
                    "ttft_mean_ms": d["results"]["ttft_mean_ms"],
                    "throughput_tok_s": d["results"]["max_throughput_tok_s"],
                    "tpot_mean_ms": d["results"].get("tpot_mean_ms", 0),
                    "meets_slo": d["results"].get("meets_slo", False),
                })
        predictions[tool_file] = entries
    return predictions


def parse_trace(trace_path):
    ttfts, tpots = [], []
    total_output_tokens = 0
    first_send_us, last_finish_us = None, None
    ok_count, total = 0, 0
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["status"] != "ok":
                continue
            ok_count += 1
            send_us = int(row["send_time_us"])
            first_chunk_us = int(row["first_chunk_time_us"])
            last_chunk_us = int(row["last_chunk_time_us"])
            output_tokens = int(row["output_tokens"])
            ttfts.append((first_chunk_us - send_us) / 1000.0)
            tpots.append((last_chunk_us - first_chunk_us) / 1000.0 / max(output_tokens - 1, 1))
            total_output_tokens += output_tokens
            if first_send_us is None or send_us < first_send_us:
                first_send_us = send_us
            if last_finish_us is None or last_chunk_us > last_finish_us:
                last_finish_us = last_chunk_us
    if not ttfts:
        return None
    e2e_s = (last_finish_us - first_send_us) / 1e6
    throughput = total_output_tokens / e2e_s if e2e_s > 0 else 0
    return {
        "ttft_mean_ms": np.mean(ttfts),
        "throughput_tok_s": throughput,
        "tpot_mean_ms": np.mean(tpots),
        "num_ok": ok_count,
        "num_total": total,
    }


def main():
    predictions = load_predictions()

    # (tool, config_hash, tp, replicas, max_num_seqs)
    actual_configs = {
        "aiconfigurator2": ("aiconfigurator", "89220430", 4, 1, 96),
        "aiconfigurator3": ("aiconfigurator", "16f33277", 4, 2, 96),
        "blis2": ("blis", "00aeb3d7", 4, 1, 512),
        "blis3": ("blis", "c2bd6baf", 8, 1, 512),
        "llmoptimizer3": ("llm-optimizer", "fa0cb0b0", 4, 1, 8),
    }

    actuals = {}
    for pkg, (tool, cfg_hash, tp, replicas, max_num_seqs) in actual_configs.items():
        trace_path = RESULTS_DIR / pkg / "chatbot" / "trace_data.csv"
        if trace_path.exists():
            parsed = parse_trace(trace_path)
            if parsed:
                # Estimate actual max throughput using decode bandwidth:
                # max_throughput = max_num_seqs * replicas / TPOT
                total_slots = max_num_seqs * replicas
                decode_bandwidth = total_slots / (parsed["tpot_mean_ms"] / 1000.0)
                actuals[pkg] = {
                    **parsed, "tool": tool, "hash": cfg_hash,
                    "tp": tp, "replicas": replicas, "gpus": tp * replicas,
                    "max_num_seqs": max_num_seqs,
                    "estimated_max_tok_s": decode_bandwidth,
                }

    failed = [
        {"tool": "aiconfigurator", "hash": "1d065cac", "tp": 2, "replicas": 1, "gpus": 2},
        {"tool": "llm-optimizer", "hash": "2d78fecf", "tp": 2, "replicas": 1, "gpus": 2},
        {"tool": "llm-optimizer", "hash": "0da30292", "tp": 3, "replicas": 1, "gpus": 3},
    ]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(13, 8))

    # Plot all predictions (faded, small)
    for tool, entries in predictions.items():
        style = TOOL_STYLES[tool]
        ttft = [e["ttft_mean_ms"] for e in entries]
        tput = [e["throughput_tok_s"] for e in entries]
        ax.scatter(ttft, tput, c=[style["color"]] * len(entries),
                   marker=style["marker"], s=50, alpha=0.2, zorder=2)

    # Pareto fronts for predictions
    for tool, entries in predictions.items():
        style = TOOL_STYLES[tool]
        slo_entries = [e for e in entries if e["meets_slo"]]
        if not slo_entries:
            continue
        slo_entries.sort(key=lambda e: e["throughput_tok_s"])
        front = []
        min_ttft = float("inf")
        for e in reversed(slo_entries):
            if e["ttft_mean_ms"] < min_ttft:
                front.append(e)
                min_ttft = e["ttft_mean_ms"]
        front.sort(key=lambda e: e["throughput_tok_s"])

        if front:
            ax.plot([e["ttft_mean_ms"] for e in front],
                    [e["throughput_tok_s"] for e in front],
                    color=style["color"], linewidth=2.5, alpha=0.7,
                    linestyle="--", zorder=3)
            ax.scatter([e["ttft_mean_ms"] for e in front],
                       [e["throughput_tok_s"] for e in front],
                       facecolors="none", edgecolors=style["color"],
                       marker=style["marker"], s=150, linewidths=2.5,
                       label=f"{style['label']} predicted", zorder=4)
            for e in front:
                ax.annotate(f"TP{e['tp']}x{e['replicas']}",
                           (e["ttft_mean_ms"], e["throughput_tok_s"]),
                           textcoords="offset points", xytext=(6, -12),
                           fontsize=8, color=style["color"], alpha=0.8)

    # Plot actual results (filled, prominent, black edge)
    # Y axis = estimated max throughput (decode bandwidth = max_seqs*replicas/TPOT)
    actual_tools_seen = set()
    for pkg, a in actuals.items():
        style = TOOL_STYLES[a["tool"]]
        ax.scatter([a["ttft_mean_ms"]], [a["estimated_max_tok_s"]],
                   color=style["color"], marker=style["marker"], s=200,
                   linewidths=2, zorder=6, edgecolors="black")
        if a["tool"] not in actual_tools_seen:
            ax.scatter([], [], color=style["color"], marker=style["marker"], s=200,
                       edgecolors="black", linewidths=2,
                       label=f"{style['label']} actual")
            actual_tools_seen.add(a["tool"])

    # Drift arrows from predicted to actual
    for pkg, a in actuals.items():
        style = TOOL_STYLES[a["tool"]]
        for e in predictions.get(a["tool"], []):
            if e["hash"] == a["hash"]:
                ax.annotate(
                    "", xy=(a["ttft_mean_ms"], a["estimated_max_tok_s"]),
                    xytext=(e["ttft_mean_ms"], e["throughput_tok_s"]),
                    arrowprops=dict(arrowstyle="->", color=style["color"],
                                   lw=2.0, alpha=0.7),
                    zorder=5,
                )
                ax.annotate(f"TP{a['tp']}x{a['replicas']}",
                           (a["ttft_mean_ms"], a["estimated_max_tok_s"]),
                           textcoords="offset points", xytext=(8, 6),
                           fontsize=9, fontweight="bold", color=style["color"])
                break

    # Mark failed deployments
    for fd in failed:
        style = TOOL_STYLES[fd["tool"]]
        for e in predictions.get(fd["tool"], []):
            if e["hash"] == fd["hash"]:
                ax.scatter([e["ttft_mean_ms"]], [e["throughput_tok_s"]],
                           s=250, color="red", marker="X", linewidths=2.5, zorder=7)
                ax.annotate(f"TP{fd['tp']}x{fd['replicas']}\nOOM",
                           (e["ttft_mean_ms"], e["throughput_tok_s"]),
                           textcoords="offset points", xytext=(8, 6),
                           fontsize=8, color="red", fontweight="bold")
                break

    # SLO line
    ax.axvline(300, color="red", linestyle=":", linewidth=2, alpha=0.7, label="SLO (300ms TTFT)")
    ax.scatter([], [], s=250, color="red", marker="X", linewidths=2.5, label="Deploy failed (KV cache OOM)")

    ax.set_xscale("log")
    ax.set_xlabel("Mean TTFT (ms)", fontsize=13)
    ax.set_ylabel("Output Tokens/sec", fontsize=13)
    ax.set_title("Llama 3.1 70B on H100: Output Throughput vs. Mean TTFT\n(upper-left is better)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    png_path = OUTPUT_DIR / "pareto_latency_throughput.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "pareto_latency_throughput.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {png_path}")


if __name__ == "__main__":
    main()
