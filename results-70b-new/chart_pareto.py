"""Scatter plot of 70B estimator results: GPU count vs TTFT with Pareto frontier.

Axes: GPU count (x, integer) vs mean TTFT (y, ms). Lower-left is better.
Pareto-optimal points (minimize GPUs AND minimize TTFT) are highlighted.
Marker size encodes throughput for context.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR = Path(__file__).parent

TOOL_STYLES = {
    "inference-sim": {"color": "#1f77b4", "marker": "o", "label": "BLIS (inference-sim)"},
    "AIConfigurator": {"color": "#2ca02c", "marker": "^", "label": "AIConfigurator"},
    "llm-optimizer": {"color": "#9467bd", "marker": "v", "label": "llm-optimizer"},
}

FILE_TO_TOOL = {
    "blis": "inference-sim",
    "aiconfigurator": "AIConfigurator",
    "llm-optimizer": "llm-optimizer",
}


def load_results():
    """Load feasible results grouped by tool."""
    tool_data = {}
    for jsonl_file in RESULTS_DIR.glob("*.jsonl"):
        tool_name = FILE_TO_TOOL.get(jsonl_file.stem, jsonl_file.stem)
        records = []
        for line in jsonl_file.read_text().strip().split("\n"):
            if not line:
                continue
            r = json.loads(line)
            if r.get("results") and r["metadata"].get("status") == "ok":
                records.append(r)
        if records:
            tool_data[tool_name] = records
    return tool_data


def compute_pareto_front(gpu_counts, ttfts):
    """Pareto front: minimize GPU count AND minimize TTFT.

    Returns indices of non-dominated points sorted by ascending GPU count.
    """
    n = len(gpu_counts)
    if n == 0:
        return []
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if gpu_counts[j] <= gpu_counts[i] and ttfts[j] <= ttfts[i]:
                if gpu_counts[j] < gpu_counts[i] or ttfts[j] < ttfts[i]:
                    dominated[i] = True
                    break
    pareto = [i for i in range(n) if not dominated[i]]
    return sorted(pareto, key=lambda i: gpu_counts[i])


def main():
    tool_data = load_results()
    fig, ax = plt.subplots(figsize=(12, 7))

    slo_ttft = 500.0  # ms

    for tool_name, records in tool_data.items():
        style = TOOL_STYLES.get(tool_name, {"color": "gray", "marker": ".", "label": tool_name})

        gpu_counts = []
        ttfts = []
        throughputs = []
        for r in records:
            tp = r["vllm_args"]["tensor_parallel_size"]
            replicas = r["vllm_args"].get("num_replicas", 1)
            gpu_counts.append(tp * replicas)
            ttfts.append(r["results"]["ttft_mean_ms"])
            throughputs.append(r["results"]["max_throughput_tok_s"])

        gpu_counts = np.array(gpu_counts)
        ttfts = np.array(ttfts)
        throughputs = np.array(throughputs)

        pareto_idx = compute_pareto_front(gpu_counts.tolist(), ttfts.tolist())
        pareto_set = set(pareto_idx)
        non_pareto_idx = [i for i in range(len(records)) if i not in pareto_set]

        # Marker sizes proportional to throughput (provides context)
        if throughputs.max() > throughputs.min():
            sizes = 30 + 120 * (throughputs - throughputs.min()) / (throughputs.max() - throughputs.min())
        else:
            sizes = np.full_like(throughputs, 60)

        # Non-pareto points (faded, small)
        if non_pareto_idx:
            ax.scatter(
                gpu_counts[non_pareto_idx],
                ttfts[non_pareto_idx],
                s=sizes[non_pareto_idx] * 0.4,
                color=style["color"],
                marker=style["marker"],
                alpha=0.2,
                zorder=2,
            )

        # Pareto points (prominent, outlined)
        if pareto_idx:
            p_gpus = gpu_counts[pareto_idx]
            p_ttfts = ttfts[pareto_idx]
            p_sizes = sizes[pareto_idx]

            ax.scatter(
                p_gpus,
                p_ttfts,
                s=p_sizes,
                facecolors="none",
                edgecolors=style["color"],
                marker=style["marker"],
                linewidths=2.5,
                label=f'{style["label"]} (Pareto)',
                zorder=5,
            )

            # Frontier line
            sorted_pareto = sorted(zip(p_gpus, p_ttfts))
            if len(sorted_pareto) > 1:
                ax.plot(
                    [p[0] for p in sorted_pareto],
                    [p[1] for p in sorted_pareto],
                    color=style["color"],
                    linewidth=2,
                    alpha=0.7,
                    linestyle="-",
                    zorder=4,
                )

    # SLO horizontal line
    ax.axhline(
        slo_ttft, color="red", linestyle=":",
        linewidth=2, alpha=0.7, label=f"SLO: TTFT < {slo_ttft:.0f}ms",
    )

    ax.set_xlabel("GPU Count (TP x Replicas)", fontsize=13)
    ax.set_ylabel("Mean TTFT (ms)", fontsize=13)
    ax.set_title(
        "70B Config Exploration: GPU Count vs TTFT\n"
        "(Pareto-optimal points highlighted; marker size = throughput)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 9))

    plt.tight_layout()
    png_path = OUTPUT_DIR / "chart_pareto.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "chart_pareto.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
