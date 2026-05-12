"""Config Exploration Experiment - Main Orchestration

Usage:
    python -m experiments.run_all [--config CONFIG] [--results-dir DIR] setup [--seed 42]
    python -m experiments.run_all [--config CONFIG] [--results-dir DIR] prune --tool TOOL
    python -m experiments.run_all [--config CONFIG] [--results-dir DIR] sweep --tool TOOL
    python -m experiments.run_all [--config CONFIG] [--results-dir DIR] analyze

The --config flag points to a YAML file defining the experiment (model, hardware,
workload, SLO, tools). Defaults to experiments/default_config.yaml. Copy it,
edit it, and pass your copy to customize the experiment.
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from experiments.schema.output import WorkloadInfo

logger = logging.getLogger(__name__)

ALL_TOOLS = ("blis", "llmservingsim", "aiconfigurator", "vidur", "llm-optimizer")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def workload_from_config(cfg: dict) -> WorkloadInfo:
    model = cfg.get("model", {})
    wl = cfg.get("workload", {})
    slo = cfg.get("slo", {})
    return WorkloadInfo(
        model=model.get("name", "meta-llama/Llama-3.1-8B"),
        hardware=model.get("hardware", "H100_SXM_80GB"),
        preset=wl.get("preset", "chatbot"),
        num_requests=wl.get("num_requests", 10000),
        isl_mean=wl.get("isl_mean", 512),
        isl_max=wl.get("isl_max", 2048),
        osl_mean=wl.get("osl_mean", 256),
        osl_max=wl.get("osl_max", 1024),
        arrival_pattern=wl.get("arrival_pattern", "poisson"),
        slo_ttft_mean_ms=slo.get("ttft_mean_ms", 300),
        seed=wl.get("seed", 42),
    )


def tools_from_config(cfg: dict) -> tuple[str, ...]:
    tools = cfg.get("tools", list(ALL_TOOLS))
    return tuple(t for t in tools if t in ALL_TOOLS)


def cmd_setup(args, cfg: dict):
    """Generate canonical workloads for all tools."""
    from experiments.workloads.generate import generate_all_workloads

    wl = cfg.get("workload", {})
    workloads_dir = Path(args.results_dir) / "workloads"
    logger.info("Generating workloads in %s", workloads_dir)

    paths = generate_all_workloads(
        workloads_dir,
        preset=wl.get("preset", "chatbot"),
        num_requests=wl.get("num_requests", 10000),
        rate=wl.get("rate", 10.0),
    )
    for tool, path in paths.items():
        logger.info("  %s: %s", tool, path)
    logger.info("Setup complete.")


def cmd_prune(args, cfg: dict):
    """Run topology pruning pre-pass for a tool."""
    from experiments.config.pruning import PruneConfig, save_prune_results, PruneResult

    model_cfg = cfg.get("model", {})
    slo_cfg = cfg.get("slo", {})
    max_gpus = args.max_gpus if args.max_gpus is not None else model_cfg.get("max_gpus", 8)
    slo_ttft = args.slo_ttft_ms if args.slo_ttft_ms is not None else slo_cfg.get("ttft_mean_ms", 300.0)

    pruned_dir = Path(args.results_dir) / "pruned"
    pruned_dir.mkdir(parents=True, exist_ok=True)

    if args.tool == "blis":
        from experiments.config.topology import enumerate_topologies_no_pp
        topologies = enumerate_topologies_no_pp(max_gpus=max_gpus)
        logger.info("Pruning %d inference-sim topologies (SLO TTFT: %.0fms)", len(topologies), slo_ttft)
        logger.warning(
            "Topology pruning requires blis binary. "
            "Saving all %d topologies as passed (no pruning).",
            len(topologies),
        )
        results = [
            PruneResult(
                topology={"tp": t.tp, "replicas": t.replicas},
                ttft_ms=None, status="skipped", passes=True,
            )
            for t in topologies
        ]

    elif args.tool in ("llmservingsim", "aiconfigurator", "vidur"):
        from experiments.config.topology import enumerate_topologies
        topologies = enumerate_topologies(max_gpus=max_gpus)
        logger.info("Pruning %d %s topologies", len(topologies), args.tool)
        logger.warning("Pruning not implemented for %s; passing all topologies.", args.tool)
        results = [
            PruneResult(
                topology={"tp": t.tp, "pp": t.pp, "replicas": t.replicas},
                ttft_ms=None, status="skipped", passes=True,
            )
            for t in topologies
        ]

    elif args.tool == "llm-optimizer":
        from experiments.config.topology import enumerate_topologies_dp
        topologies = enumerate_topologies_dp(max_gpus=max_gpus)
        logger.info("Pruning %d llm-optimizer topologies", len(topologies))
        logger.warning("Pruning not implemented for llm-optimizer; passing all topologies.")
        results = [
            PruneResult(
                topology={"tp": t.tp, "pp": t.pp, "dp": t.dp},
                ttft_ms=None, status="skipped", passes=True,
            )
            for t in topologies
        ]
    else:
        logger.error("Unknown tool: %s", args.tool)
        return

    save_prune_results(results, pruned_dir / f"{args.tool}_topologies.json")
    passing = [r for r in results if r.passes]
    logger.info("Pruning: %d/%d topologies passed", len(passing), len(results))


def cmd_sweep(args, cfg: dict):
    """Run full config sweep for a tool."""
    model_cfg = cfg.get("model", {})
    workload = workload_from_config(cfg)
    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"{args.tool}.jsonl"

    logger.info("Sweeping %s -> %s", args.tool, output)

    if args.tool == "blis":
        from experiments.config.blis_configs import generate_blis_configs
        from experiments.runners.run_blis import BLISRunner

        max_gpus = model_cfg.get("max_gpus", 8)
        configs = generate_blis_configs(max_gpus=max_gpus)
        logger.info("Generated %d configs for inference-sim", len(configs))

        runner = BLISRunner(
            workload=workload,
            output_path=output,
        )
        runner.run_batch(
            configs,
            hash_fn=lambda c: json.dumps(c, sort_keys=True).__hash__().__abs__().__str__()[:8],
        )

    elif args.tool == "aiconfigurator":
        from experiments.runners.run_aiconfigurator import AIConfiguratorRunner

        max_gpus = model_cfg.get("max_gpus", 8)
        gpu_cost = model_cfg.get("gpu_cost_per_hour", 3.20)
        runner = AIConfiguratorRunner(
            workload=workload,
            output_path=output,
            gpu_cost_per_hour=gpu_cost,
            max_gpus=max_gpus,
        )
        n = runner.run_full_sweep()
        logger.info("AIConfigurator: %d total results written", n)

    elif args.tool == "llm-optimizer":
        from experiments.runners.run_llm_optimizer import LLMOptimizerRunner

        max_gpus = model_cfg.get("max_gpus", 8)
        gpu_cost = model_cfg.get("gpu_cost_per_hour", 3.20)
        runner = LLMOptimizerRunner(
            workload=workload,
            output_path=output,
            gpu_cost_per_hour=gpu_cost,
            max_gpus=max_gpus,
        )
        n = runner.run_full_sweep()
        logger.info("llm-optimizer: %d total results written", n)

    else:
        logger.warning("Runner for %s not yet implemented. See experiments/runners/", args.tool)


def cmd_analyze(args, cfg: dict):
    """Compute Pareto fronts, select top-3, generate Chart 1."""
    analysis_cfg = cfg.get("analysis", {})
    slo_cfg = cfg.get("slo", {})
    tools = tools_from_config(cfg)
    top_k = analysis_cfg.get("top_k", 3)
    min_throughput = analysis_cfg.get("min_throughput_tok_s", 200)
    slo_ttft = slo_cfg.get("ttft_mean_ms", 300.0)

    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    processed_dir = results_dir / "processed"
    figures_dir = results_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from experiments.analysis.pareto import compute_pareto_front
    from experiments.analysis.select import select_top_k

    all_top = {}
    for tool in tools:
        results_file = raw_dir / f"{tool}.jsonl"
        if not results_file.exists():
            logger.warning("No results for %s, skipping", tool)
            continue

        results = []
        for line in results_file.read_text().strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    if data.get("results") and data["metadata"].get("status") == "ok":
                        results.append(data)
                except json.JSONDecodeError:
                    continue

        front = compute_pareto_front(results)
        selected = select_top_k(front, k=top_k, min_throughput=min_throughput)
        logger.info(
            "%s: %d results, %d Pareto, %d top-%d",
            tool, len(results), len(front), len(selected), top_k,
        )
        all_top[tool] = selected

    selection_path = processed_dir / f"top{top_k}_selection.json"
    selection_path.write_text(json.dumps(all_top, indent=2))
    logger.info("Top-%d selection saved to %s", top_k, selection_path)

    from experiments.analysis.chart1 import plot_chart1

    validation_file = results_dir / "validated" / "validation.jsonl"
    chart_path = figures_dir / "chart1"
    try:
        png = plot_chart1(
            raw_dir, chart_path,
            validation_file=validation_file if validation_file.exists() else None,
            slo_ttft_ms=slo_ttft,
        )
        logger.info("Chart 1 saved to %s", png)
    except Exception as e:
        logger.error("Failed to generate Chart 1: %s", e)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Config Exploration Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to experiment YAML config (default: experiments/default_config.yaml)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Results directory (default: results/; use /data/config-exploration/results on k8s)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup
    subparsers.add_parser("setup", help="Generate canonical workloads")

    # prune
    prune_parser = subparsers.add_parser("prune", help="Topology pruning pre-pass")
    prune_parser.add_argument("--tool", required=True, choices=ALL_TOOLS)
    prune_parser.add_argument("--max-gpus", type=int, default=None,
                              help="Override max GPU budget from config")
    prune_parser.add_argument("--slo-ttft-ms", type=float, default=None,
                              help="Override SLO TTFT from config")

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="Full config sweep")
    sweep_parser.add_argument("--tool", required=True, choices=ALL_TOOLS)

    # analyze
    subparsers.add_parser("analyze", help="Compute Pareto fronts and Chart 1")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "setup":
        cmd_setup(args, cfg)
    elif args.command == "prune":
        cmd_prune(args, cfg)
    elif args.command == "sweep":
        cmd_sweep(args, cfg)
    elif args.command == "analyze":
        cmd_analyze(args, cfg)


if __name__ == "__main__":
    main()
