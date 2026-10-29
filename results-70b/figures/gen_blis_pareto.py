#!/usr/bin/env python3
"""Generate publication-quality BLIS NSGA-II Pareto frontier chart."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
SEARCH_RESULTS = Path("/tmp/claude-501/blis_nsga2_70b_2obj.json")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 24,
    "axes.labelsize": 28,
    "axes.titlesize": 30,
    "axes.titleweight": 600,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.8,
    "grid.linewidth": 0.8,
})


GPU_COLORS = {
    2: "#e67300",
    3: "#fdbb84",
    4: "#e34a33",
    5: "#d73027",
    6: "#a50026",
    7: "#800026",
    8: "#4a0014",
}


def gpu_color(gpu_count):
    return GPU_COLORS.get(int(gpu_count), "#999999")


def main():
    with open(SEARCH_RESULTS) as f:
        data = json.load(f)

    configs = data["configs"]
    # Filter: TP>=2 required (70B doesn't fit on 1 GPU), and at least 2 GPUs total
    configs = [c for c in configs if c["config"]["tp"] >= 2 and int(c["objectives"]["gpu_count"]) >= 2]

    SLO = 120
    pareto = [c for c in configs if c["is_pareto"] and c["objectives"]["ttft_p99_ms"] <= SLO]
    non_pareto = [c for c in configs if c not in pareto]

    fig, ax = plt.subplots(figsize=(16, 10))

    # 1. All configs (with jitter to show density)
    all_plotted = [c for c in configs if c["objectives"]["ttft_p99_ms"] <= 200]
    rng = np.random.default_rng(42)
    for c in all_plotted:
        ttft = c["objectives"]["ttft_p99_ms"] * (1 + rng.uniform(-0.05, 0.05))
        tput = c["objectives"]["throughput_rps"] * (1 + rng.uniform(-0.03, 0.03))
        color = gpu_color(c["objectives"]["gpu_count"])
        alpha = 0.4 if c["objectives"]["ttft_p99_ms"] <= SLO else 0.25
        ax.scatter(ttft, tput, c=color, marker="o", s=45, alpha=alpha, zorder=1, rasterized=True)

    # 2. Best config per GPU tier
    slo_meeting = [c for c in configs if c["objectives"]["ttft_p99_ms"] <= SLO]
    all_within_view = [c for c in configs if c["objectives"]["ttft_p99_ms"] <= 200]
    tier_best = {}
    for c in all_within_view:
        gpus = int(c["objectives"]["gpu_count"])
        tput = c["objectives"]["throughput_rps"]
        if gpus not in tier_best or tput > tier_best[gpus]["objectives"]["throughput_rps"]:
            tier_best[gpus] = c
    # Star markers on best config per tier (only SLO-meeting tiers)
    for gpus, c in sorted(tier_best.items()):
        if c["objectives"]["ttft_p99_ms"] > SLO:
            continue
        color = gpu_color(gpus)
        ax.scatter(
            c["objectives"]["ttft_p99_ms"], c["objectives"]["throughput_rps"],
            c=color, s=600, marker="*",
            edgecolors="black", linewidths=1.2, zorder=8, alpha=1.0,
        )

    # 4. Region annotations with specific config args (placed in empty right area)
    bbox_style = dict(boxstyle="round,pad=0.4", alpha=0.15, linewidth=0)
    ax.annotate(
        "High budget (8 GPU)\nTP8 x 1 replica\nnum-gpu-blocks-override=4-5k\nmax-num-seqs=256",
        xy=(145, 11.0), fontsize=17, color="#4a0014",
        fontweight=500, va="top", ha="left",
        bbox=dict(**bbox_style, facecolor="#4a0014"),
    )
    ax.annotate(
        "Mid budget (4-6 GPU)\nTP4 x 1 replica\nnum-gpu-blocks-override=10k, max-num-seqs=512\n\n"
        "TP2 x 3 replicas\nnum-gpu-blocks-override=7.5k, max-num-seqs=512",
        xy=(145, 8.8), fontsize=17, color="#a50026",
        fontweight=500, va="top", ha="left",
        bbox=dict(**bbox_style, facecolor="#a50026"),
    )
    ax.annotate(
        "Low budget (2 GPU, fails SLO)\nTP2 x 1 replica\nnum-gpu-blocks-override=10k\nmax-num-seqs=512",
        xy=(145, 5.8), fontsize=17, color="#e34a33",
        fontweight=500, va="top", ha="left",
        bbox=dict(**bbox_style, facecolor="#e34a33"),
    )

    # Arrows from boxes to best points
    arrow_style = dict(arrowstyle="->", color="black", lw=2.0, alpha=0.6)
    # High budget box (xy=145,11) -> 8 GPU best
    if 8 in tier_best:
        c = tier_best[8]
        ax.annotate("", xy=(c["objectives"]["ttft_p99_ms"], c["objectives"]["throughput_rps"]),
                    xytext=(145, 10.8), arrowprops=arrow_style, zorder=7)
    # Mid budget box -> 4 GPU best
    if 4 in tier_best:
        c = tier_best[4]
        ax.annotate("", xy=(c["objectives"]["ttft_p99_ms"], c["objectives"]["throughput_rps"]),
                    xytext=(145, 7.8), arrowprops=arrow_style, zorder=7)
    # Mid budget box -> 6 GPU best
    if 6 in tier_best:
        c = tier_best[6]
        ax.annotate("", xy=(c["objectives"]["ttft_p99_ms"], c["objectives"]["throughput_rps"]),
                    xytext=(145, 7.4), arrowprops=arrow_style, zorder=7)

    # 5. SLO line
    ax.axvline(SLO, color="#d62728", linestyle="--", linewidth=2.0, alpha=0.7, zorder=4)
    ax.set_xlim(left=30, right=260)
    ylim = ax.get_ylim()
    ax.text(
        122, ylim[1] * 0.42,
        "120 ms\nSLO", color="#d62728", fontsize=18, fontweight=600,
        va="top", ha="left",
    )

    # 6. Axes and styling
    ax.set_xlabel("TTFT P99 (ms)", fontweight=500)
    ax.set_ylabel("Max Sustainable Rate (req/s)", fontweight=500)
    ax.set_title(
        "BLIS Config Search: Pareto Frontier for Llama-3.1-70B on H100",
        pad=12,
    )
    ax.grid(True, alpha=0.2, which="major", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 7. Legend (only GPU counts present in plotted data)
    present_gpus = sorted(set(int(c["objectives"]["gpu_count"]) for c in all_plotted))
    handles = [
        mpatches.Patch(color=GPU_COLORS[g], label=f"{g} GPUs")
        for g in present_gpus
    ]
    legend = ax.legend(
        handles=handles, title="GPU count",
        loc="upper right", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", fancybox=True,
    )
    legend.get_frame().set_linewidth(1.0)

    # Caption
    total_slo = len([c for c in configs if c["objectives"]["ttft_p99_ms"] <= SLO])
    fig.text(
        0.5, 0.01,
        f"{len(configs)} configs evaluated, "
        f"{total_slo} meet SLO, {len(pareto)} on Pareto front. "
        f"Offered rate = 30 req/s.",
        ha="center", fontsize=14, color="#666666", style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    png_path = OUTPUT_DIR / "blis_pareto_frontier.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "blis_pareto_frontier.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {png_path}")


if __name__ == "__main__":
    main()
