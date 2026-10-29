"""Pareto Frontiers: TTFT vs. Throughput by GPU Budget Tier for BLIS configs.

Shows that BLIS's NSGA-II search produces Pareto-optimal configs across
different GPU budgets. Each tier represents a fixed hardware allocation,
and the Pareto frontier shows the best achievable TTFT-throughput tradeoff
within that budget.

Axes:
  - X: System Throughput (tokens/s) - higher is better
  - Y: Mean TTFT (ms) - lower is better
  - Ideal region: lower-right (high throughput, low latency)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR = Path(__file__).parent


def load_blis_results():
    f = RESULTS_DIR / "blis.jsonl"
    records = []
    for line in f.read_text().strip().split("\n"):
        if not line:
            continue
        r = json.loads(line)
        if r.get("results") and r["metadata"].get("status") == "ok":
            records.append(r)
    return records


def compute_pareto_front_min_y_max_x(xs, ys):
    """Pareto front: maximize x (throughput) and minimize y (TTFT).

    Returns indices sorted by ascending x.
    """
    n = len(xs)
    if n == 0:
        return []
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has higher throughput AND lower TTFT
            if xs[j] >= xs[i] and ys[j] <= ys[i]:
                if xs[j] > xs[i] or ys[j] < ys[i]:
                    dominated[i] = True
                    break
    pareto = [i for i in range(n) if not dominated[i]]
    return sorted(pareto, key=lambda i: xs[i])


def main():
    records = load_blis_results()

    # GPU budget tiers
    tier_styles = {
        2: {
            "frontier_color": "#8B0000",
            "scatter_color": "#CD853F",
            "label": "2 GPUs",
            "marker": "o",
        },
        4: {
            "frontier_color": "#2E8B57",
            "scatter_color": "#90EE90",
            "label": "4 GPUs",
            "marker": "s",
        },
        8: {
            "frontier_color": "#1E3A8A",
            "scatter_color": "#6495ED",
            "label": "8 GPUs",
            "marker": "D",
        },
    }

    # Group records by GPU count
    tier_records = {}
    for r in records:
        tp = r["vllm_args"]["tensor_parallel_size"]
        replicas = r["vllm_args"].get("num_replicas", 1)
        gpu_count = tp * replicas
        if gpu_count not in tier_records:
            tier_records[gpu_count] = []
        tier_records[gpu_count].append(r)

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # SLO line
    slo_ttft = 500.0

    for gpu_tier in sorted(tier_styles.keys()):
        if gpu_tier not in tier_records:
            continue

        style = tier_styles[gpu_tier]
        recs = tier_records[gpu_tier]

        # Also include 6-GPU configs in the 4-GPU visual group (TP=2, replicas=3)
        if gpu_tier == 4 and 6 in tier_records:
            recs = recs + tier_records[6]

        throughputs = np.array([r["results"]["max_throughput_tok_s"] for r in recs])
        ttfts = np.array([r["results"]["ttft_mean_ms"] for r in recs])

        # Compute Pareto frontier across ALL configs in this tier
        pareto_idx = compute_pareto_front_min_y_max_x(
            throughputs.tolist(), ttfts.tolist()
        )
        pareto_set = set(pareto_idx)
        non_pareto_idx = [i for i in range(len(recs)) if i not in pareto_set]

        # Non-Pareto points (faded background)
        if non_pareto_idx:
            ax.scatter(
                throughputs[non_pareto_idx], ttfts[non_pareto_idx],
                s=25,
                color=style["scatter_color"],
                alpha=0.4,
                zorder=2,
                edgecolors="none",
            )

        if pareto_idx:
            p_xs = throughputs[pareto_idx]
            p_ys = ttfts[pareto_idx]

            # Frontier points (prominent)
            ax.scatter(
                p_xs, p_ys,
                s=55,
                color=style["frontier_color"],
                marker=style["marker"],
                zorder=5,
                edgecolors=style["frontier_color"],
                linewidths=1.5,
            )

            # Frontier line
            ax.plot(
                p_xs, p_ys,
                color=style["frontier_color"],
                linewidth=2.5,
                alpha=0.85,
                zorder=4,
                label=f"Pareto Frontier: {style['label']}",
            )

            # Best config: highest throughput while meeting SLO
            slo_passing = [(i, p_xs[i], p_ys[i]) for i in range(len(p_xs))
                           if p_ys[i] <= slo_ttft]
            if slo_passing:
                best = max(slo_passing, key=lambda t: t[1])
                ax.scatter(
                    [best[1]], [best[2]],
                    s=350,
                    color=style["frontier_color"],
                    marker="*",
                    zorder=6,
                    label=f"Best Config: {style['label']}",
                    edgecolors="black",
                    linewidths=0.8,
                )

    # SLO horizontal line
    ax.axhline(
        slo_ttft, color="red", linestyle=":",
        linewidth=2, alpha=0.7, zorder=3,
        label=f"SLO: TTFT < {slo_ttft:.0f}ms",
    )

    ax.set_xlabel("System Throughput (tokens/s)", fontsize=13)
    ax.set_ylabel("Mean TTFT (ms)", fontsize=13)
    ax.set_title(
        "Pareto Frontiers: TTFT vs. Throughput by GPU Budget",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax.set_yscale("log")
    ax.set_ylim(bottom=30, top=50000)
    ax.set_xlim(left=500)

    plt.tight_layout()
    png_path = OUTPUT_DIR / "chart_pareto_frontier.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "chart_pareto_frontier.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
