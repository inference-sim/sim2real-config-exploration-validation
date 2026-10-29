"""Pareto Frontier: Throughput vs. TTFT P99 from BLIS NSGA-II search.

Shows per-GPU-tier Pareto frontiers: the best achievable throughput
at each TTFT P99 level. Ideal region is upper-left (high throughput,
low latency).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR = Path(__file__).parent

AVG_TOKENS_PER_REQUEST = 813  # ISL=566 + OSL=247


def load_search_results():
    f = RESULTS_DIR / "blis_search_output_llama2.json"
    data = json.loads(f.read_text())
    return data["all_configs"]


def compute_pareto_front(xs, ys):
    """Pareto front: minimize x (TTFT) and maximize y (throughput)."""
    n = len(xs)
    if n == 0:
        return []
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i]:
                if xs[j] < xs[i] or ys[j] > ys[i]:
                    dominated[i] = True
                    break
    pareto = [i for i in range(n) if not dominated[i]]
    return sorted(pareto, key=lambda i: xs[i])


def main():
    all_configs = load_search_results()

    tier_config = {
        "2 GPUs": {"range": (2, 2), "color": "#D62728", "marker": "o"},
        "4 GPUs": {"range": (4, 4), "color": "#2CA02C", "marker": "s"},
        "6-8 GPUs": {"range": (6, 8), "color": "#1F77B4", "marker": "D"},
    }

    # Parse and deduplicate configs per tier
    tier_data = {}
    for tier_name, cfg in tier_config.items():
        lo, hi = cfg["range"]
        seen = set()
        points = []
        for entry in all_configs:
            if not entry["feasible"]:
                continue
            obj = entry["objectives"]
            gpu_count = obj["gpu_count"]
            if not (lo <= gpu_count <= hi):
                continue
            throughput = obj["throughput"] * AVG_TOKENS_PER_REQUEST
            ttft = obj["ttft_p99"]
            key = (round(throughput, 1), round(ttft, 1))
            if key in seen:
                continue
            seen.add(key)
            points.append((ttft, throughput))
        tier_data[tier_name] = points

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for tier_name, cfg in tier_config.items():
        points = tier_data[tier_name]
        if not points:
            continue

        xs = np.array([p[0] for p in points])  # TTFT P99 (x-axis)
        ys = np.array([p[1] for p in points])  # Throughput (y-axis)

        # Pareto: minimize TTFT, maximize throughput
        front_idx = compute_pareto_front(xs.tolist(), ys.tolist())
        front_set = set(front_idx)
        non_front_idx = [i for i in range(len(points)) if i not in front_set]

        # Non-frontier points (faded)
        if non_front_idx:
            ax.scatter(
                xs[non_front_idx], ys[non_front_idx],
                s=30, color=cfg["color"], alpha=0.2,
                marker=cfg["marker"], edgecolors="none", zorder=2,
            )

        # Frontier points with connecting line
        if front_idx:
            f_x = xs[front_idx]
            f_y = ys[front_idx]

            # Filled area under the frontier
            ax.fill_between(
                f_x, 0, f_y,
                color=cfg["color"], alpha=0.06, zorder=1,
            )

            ax.plot(
                f_x, f_y,
                color=cfg["color"], linewidth=2.5, alpha=0.85,
                zorder=4,
            )
            ax.scatter(
                f_x, f_y,
                s=70, color=cfg["color"], marker=cfg["marker"],
                zorder=5, edgecolors="black", linewidths=0.7,
                label=f"{tier_name} Pareto ({len(front_idx)} pts)",
            )

    ax.set_xlabel("TTFT P99 (ms)", fontsize=11)
    ax.set_ylabel("System Throughput (tokens/s)", fontsize=11)
    ax.set_title(
        "BLIS Config Search: Throughput vs. TTFT P99 by GPU Budget\n"
        "(Llama-2-70B, 1100 configs evaluated)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(left=30, right=100000)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    png_path = OUTPUT_DIR / "chart_pareto_frontier.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "chart_pareto_frontier.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
