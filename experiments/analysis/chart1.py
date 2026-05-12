import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

TOOL_STYLES = {
    "inference-sim": {"color": "#1f77b4", "marker": "o", "label": "inference-sim (BLIS)"},
    "LLMServingSim": {"color": "#ff7f0e", "marker": "s", "label": "LLMServingSim"},
    "AIConfigurator": {"color": "#2ca02c", "marker": "^", "label": "AIConfigurator"},
    "Vidur": {"color": "#d62728", "marker": "D", "label": "Vidur"},
    "llm-optimizer": {"color": "#9467bd", "marker": "v", "label": "llm-optimizer"},
}

# Aliases for JSONL filenames that differ from tool names
TOOL_FILE_ALIASES = {
    "blis": "inference-sim",
    "llmservingsim": "LLMServingSim",
    "aiconfigurator": "AIConfigurator",
    "vidur": "Vidur",
    "llm-optimizer": "llm-optimizer",
}


def load_tool_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load SLO-meeting results from per-tool JSONL files."""
    tool_results = {}
    for jsonl_file in results_dir.glob("*.jsonl"):
        stem = jsonl_file.stem
        tool_name = TOOL_FILE_ALIASES.get(stem, stem)
        results = []
        for line in jsonl_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("results") and data["metadata"].get("status") == "ok":
                    results.append(data)
            except json.JSONDecodeError:
                continue
        if results:
            tool_results[tool_name] = results
    return tool_results


def compute_pareto_indices(costs: list[float], throughputs: list[float]) -> list[int]:
    """Return indices of Pareto-optimal points (min cost, max throughput)."""
    n = len(costs)
    if n == 0:
        return []

    sorted_indices = sorted(range(n), key=lambda i: (costs[i], -throughputs[i]))
    pareto = []
    max_tput = float("-inf")
    for idx in sorted_indices:
        if throughputs[idx] > max_tput:
            pareto.append(idx)
            max_tput = throughputs[idx]
    return pareto


def load_validation_data(validation_file: Path) -> dict[str, dict]:
    """Load validation results keyed by config_hash."""
    if not validation_file.exists():
        return {}
    lookup = {}
    for line in validation_file.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            h = data.get("config_hash")
            if h:
                lookup[h] = data
        except json.JSONDecodeError:
            continue
    return lookup


def plot_chart1(
    results_dir: Path,
    output_path: Path,
    validation_file: Optional[Path] = None,
    slo_ttft_ms: float = 300.0,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Generate Chart 1: config exploration scatter with Pareto fronts.

    Args:
        results_dir: Directory containing per-tool JSONL files.
        output_path: Base path for output (will create .pdf and .png).
        validation_file: Optional JSONL with validated measurements.
        slo_ttft_ms: SLO threshold for TTFT.
        figsize: Figure size in inches.

    Returns:
        Path to the generated PNG file.
    """
    fig, ax = plt.subplots(figsize=figsize)

    tool_results = load_tool_results(results_dir)
    val_lookup = load_validation_data(validation_file) if validation_file else {}

    for tool_name, results in tool_results.items():
        style = TOOL_STYLES.get(tool_name, {"color": "gray", "marker": ".", "label": tool_name})

        slo_meeting = [
            r for r in results
            if r["results"].get("meets_slo", False)
        ]
        if not slo_meeting:
            continue

        ttft = [r["results"]["ttft_mean_ms"] for r in slo_meeting]
        throughput = [r["results"]["max_throughput_tok_s"] for r in slo_meeting]
        costs = [r["results"]["cost_per_hour"] for r in slo_meeting]

        pareto_idx = compute_pareto_indices(ttft, throughput)
        non_pareto_idx = [i for i in range(len(slo_meeting)) if i not in pareto_idx]

        # Marker sizes proportional to cost (min 30, max 200)
        cost_arr = np.array(costs)
        if cost_arr.max() > cost_arr.min():
            sizes = 30 + 170 * (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min())
        else:
            sizes = np.full_like(cost_arr, 80)

        # Non-Pareto points (small, filled, faded)
        if non_pareto_idx:
            ax.scatter(
                [ttft[i] for i in non_pareto_idx],
                [throughput[i] for i in non_pareto_idx],
                s=[sizes[i] * 0.4 for i in non_pareto_idx],
                color=style["color"], marker=style["marker"],
                alpha=0.15, zorder=2,
            )

        # Pareto points (hollow, prominent)
        if pareto_idx:
            pareto_ttft = [ttft[i] for i in pareto_idx]
            pareto_tput = [throughput[i] for i in pareto_idx]
            pareto_sizes = [sizes[i] for i in pareto_idx]

            ax.scatter(
                pareto_ttft, pareto_tput,
                s=pareto_sizes,
                facecolors="none", edgecolors=style["color"],
                marker=style["marker"], linewidths=2,
                label=style["label"], zorder=4,
            )

            # Pareto frontier line (sorted by cost ascending = left to right in cost space)
            sorted_pareto = sorted(zip(pareto_ttft, pareto_tput))
            ax.plot(
                [p[0] for p in sorted_pareto],
                [p[1] for p in sorted_pareto],
                color=style["color"], linewidth=1.5, alpha=0.6,
                linestyle="--", zorder=3,
            )

        # Drift arrows for validated configs
        for r in slo_meeting:
            cfg_hash = r.get("metadata", {}).get("config_hash")
            if cfg_hash and cfg_hash in val_lookup:
                val = val_lookup[cfg_hash]
                pred_ttft = r["results"]["ttft_mean_ms"]
                pred_tput = r["results"]["max_throughput_tok_s"]
                actual_ttft = val["actual_ttft_mean_ms"]
                actual_tput = val["actual_throughput_tok_s"]

                ax.annotate(
                    "", xy=(actual_ttft, actual_tput),
                    xytext=(pred_ttft, pred_tput),
                    arrowprops=dict(
                        arrowstyle="->", color=style["color"],
                        lw=2.0, alpha=0.8,
                    ),
                    zorder=5,
                )

                # Red X if actual violates SLO
                if actual_ttft > slo_ttft_ms:
                    ax.scatter(
                        [actual_ttft], [actual_tput],
                        s=120, color="red", marker="x",
                        linewidths=3, zorder=6,
                    )

    # SLO vertical line
    ax.axvline(
        slo_ttft_ms, color="red", linestyle=":",
        linewidth=2, alpha=0.7, label=f"SLO: TTFT < {slo_ttft_ms:.0f}ms",
    )

    ax.set_xlabel("Mean TTFT (ms)", fontsize=13)
    ax.set_ylabel("Max Throughput (tokens/sec)", fontsize=13)
    ax.set_title("Config Exploration: Predicted Performance by Tool", fontsize=14)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    plt.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Chart 1 saved to %s and %s", pdf_path, png_path)
    return png_path
