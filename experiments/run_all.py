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
import multiprocessing
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


def concept_value(cfg: dict, concept: str, default=None):
    """Get a concept's canonical value from concept_map."""
    cmap = cfg.get("concept_map", {})
    entry = cmap.get(concept, {})
    if isinstance(entry, dict):
        return entry.get("value", default)
    return default


def workload_from_config(cfg: dict, config_dir: Path | None = None) -> WorkloadInfo:
    wl = cfg.get("workload", {})
    isl = wl.get("input_seq_len", {})
    osl = wl.get("output_seq_len", {})

    workload_file = None
    if wl.get("file"):
        wf = Path(wl["file"])
        if wf.exists():
            workload_file = str(wf.resolve())
        elif not wf.is_absolute() and config_dir:
            wf = config_dir / wl["file"]
            if wf.exists():
                workload_file = str(wf.resolve())

    preset = wl.get("preset", "distribution" if workload_file else "chatbot")

    return WorkloadInfo(
        model=concept_value(cfg, "model", "meta-llama/Llama-3.1-8B"),
        hardware=concept_value(cfg, "hardware", "H100_SXM_80GB"),
        preset=preset,
        num_requests=wl.get("num_requests", 10000),
        isl_mean=isl.get("mean", wl.get("isl_mean", 512)),
        isl_max=isl.get("max", wl.get("isl_max", 2048)),
        osl_mean=osl.get("mean", wl.get("osl_mean", 256)),
        osl_max=osl.get("max", wl.get("osl_max", 1024)),
        arrival_pattern=wl.get("arrival", wl.get("arrival_pattern", "poisson")),
        slo_ttft_mean_ms=concept_value(cfg, "slo_ttft_ms", 300),
        seed=concept_value(cfg, "seed", 42),
        workload_file=workload_file,
    )


def tools_from_config(cfg: dict) -> tuple[str, ...]:
    tools = cfg.get("tools", list(ALL_TOOLS))
    if isinstance(tools, dict):
        return tuple(t for t in tools if t in ALL_TOOLS)
    return tuple(t for t in tools if t in ALL_TOOLS)


def resources_from_config(cfg: dict) -> dict:
    """Extract resource constraints from concept_map."""
    return {
        "max_gpus": concept_value(cfg, "max_gpus", 8),
        "gpu_cost_per_hour": concept_value(cfg, "gpu_cost_per_hour", 3.20),
    }


def tool_config(cfg: dict, tool_name: str) -> dict:
    """Get a tool's config from tools.<name>."""
    tools = cfg.get("tools", {})
    if isinstance(tools, dict) and tool_name in tools:
        return tools[tool_name]
    return cfg.get(tool_name, {})


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

    res = resources_from_config(cfg)
    slo_cfg = cfg.get("slo", {})
    max_gpus = args.max_gpus if args.max_gpus is not None else res["max_gpus"]
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


def _blis_worker(
    worker_index: int, total_workers: int,
    configs: list[dict], workload: WorkloadInfo,
    shard_path: Path,
) -> int:
    """Run a shard of BLIS configs in a worker process. Returns count of results written."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [worker-{worker_index}] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from experiments.runners.run_blis import BLISRunner

    shard = configs[worker_index::total_workers]
    runner = BLISRunner(workload=workload, output_path=shard_path)

    completed = runner.load_completed_hashes()
    total = len(shard)
    written = 0

    for i, config in enumerate(shard):
        config_hash = json.dumps(config, sort_keys=True).__hash__().__abs__().__str__()[:8]
        if config_hash in completed:
            continue

        try:
            result = runner.evaluate_config(config)
            result.metadata.config_hash = config_hash
            runner.append_result(result)
            completed.add(config_hash)
            written += 1
        except Exception as e:
            logging.getLogger(__name__).error("Config %s failed: %s", config_hash, e)

        if (i + 1) % 10 == 0:
            logging.getLogger(__name__).info(
                "%d/%d evaluated (%d written)", i + 1, total, written,
            )

    logging.getLogger(__name__).info("Worker %d done: %d/%d written", worker_index, written, total)
    return written


def cmd_sweep(args, cfg: dict):
    """Run full config sweep for a tool."""
    import time as _time
    sweep_start = _time.monotonic()

    res = resources_from_config(cfg)
    config_dir = Path(args.config).parent if hasattr(args, "config") else None
    workload = workload_from_config(cfg, config_dir=config_dir)
    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"{args.tool}.jsonl"

    logger.info("Sweeping %s -> %s", args.tool, output)

    if args.tool == "blis":
        from experiments.config.blis_configs import generate_pruned_blis_configs
        from experiments.runners.run_blis import BLISRunner

        max_gpus = res["max_gpus"]
        blis_cfg = tool_config(cfg, "blis")
        search_cfg = blis_cfg.get("search", blis_cfg)
        strategy = search_cfg.get("strategy", "adaptive-hierarchical")

        runner = BLISRunner(
            workload=workload,
            output_path=output,
            workload_spec=workload.workload_file,
            search_budget=search_cfg.get("budget", 20),
            phase1_k=search_cfg.get("phase1_k", 1),
            phase1_profiles=search_cfg.get("phase1_profiles", "bracket"),
            tp_candidates=search_cfg.get("tp_candidates", "lean"),
            skip_phase2=search_cfg.get("skip_phase2", False),
            fitness_weights=search_cfg.get(
                "fitness_weights", "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3",
            ),
            search_script=search_cfg.get("search_script"),
            defaults_filepath=search_cfg.get("defaults_filepath"),
            model_config_folder=search_cfg.get("model_config_folder"),
        )
        if "fixed_rate" in search_cfg:
            runner._search_rate = search_cfg["fixed_rate"]

        if strategy == "adaptive-hierarchical":
            logger.info(
                "Running adaptive-hierarchical search (budget=%d, tp=%s)",
                runner.search_budget, runner.tp_candidates,
            )
            results = runner.run_search(search_seed=workload.seed)
            logger.info("Search complete: %d configs evaluated", len(results))
        elif strategy == "tpe":
            logger.info(
                "Running TPE search (budget=%d)",
                runner.search_budget,
            )
            results = runner.run_tpe_search(search_seed=workload.seed)
            logger.info("TPE search complete: %d configs evaluated", len(results))
        elif strategy == "nsga2":
            nsga2_cfg = search_cfg.get("nsga2", {})
            pop_size = nsga2_cfg.get("pop_size", 40)
            n_gens = nsga2_cfg.get("n_gens", 4)
            workers = nsga2_cfg.get("workers", 8)
            objectives = nsga2_cfg.get("objectives", ["throughput", "ttft_mean", "gpu_count"])
            constraints = nsga2_cfg.get("constraints", [])
            params = nsga2_cfg.get("params", None)
            scorer_space = nsga2_cfg.get("scorer_space", None)

            # Derive params and scorer_space from concept_map
            if params is None:
                cmap = cfg.get("concept_map", {})
                # Map concept_map names to BLIS CLI arg names
                blis_param_mapping = {
                    "tp": "tp",
                    "deployment_replicas": "num_instances",
                    "max_num_seqs": "max_num_running_reqs",
                    "max_batched_tokens": "max_num_scheduled_tokens",
                    "chunked_prefill_threshold": "long_prefill_token_threshold",
                    "block_size": "block_size_in_tokens",
                    "kv_blocks": "total_kv_blocks",
                    "scheduler": "scheduler",
                    "routing_policy": "routing_policy",
                    "admission_policy": "admission_policy",
                    "preemption_policy": "preemption_policy",
                }
                params = {}
                for concept, blis_name in blis_param_mapping.items():
                    entry = cmap.get(concept, {})
                    if isinstance(entry, dict):
                        val = entry.get("value")
                        if isinstance(val, list):
                            params[blis_name] = val

                # Extract scorer_space from concept_map if not in nsga2 config
                if scorer_space is None:
                    scorer_entry = cmap.get("routing_scorer_space", {})
                    if isinstance(scorer_entry, dict):
                        val = scorer_entry.get("value")
                        if isinstance(val, dict):
                            scorer_space = val

            max_model_len = concept_value(cfg, "max_model_len")

            logger.info(
                "Running native NSGA-II search (pop=%d, gens=%d, workers=%d, objectives=%s)",
                pop_size, n_gens, workers, objectives,
            )
            results = runner.run_native_search(
                search_seed=workload.seed,
                pop_size=pop_size,
                n_gens=n_gens,
                workers=workers,
                objectives=objectives,
                constraints=constraints,
                params=params,
                scorer_space=scorer_space,
                fixed_rate=search_cfg.get("fixed_rate"),
                max_model_len=max_model_len,
            )
            pareto_count = sum(1 for r in results if r.metadata.pareto_optimal)
            logger.info(
                "NSGA-II complete: %d evaluated, %d Pareto-optimal",
                len(results), pareto_count,
            )
        elif strategy == "nsga2-legacy":
            nsga2_cfg = search_cfg.get("nsga2", {})
            pop_size = nsga2_cfg.get("pop_size", 40)
            n_gens = nsga2_cfg.get("n_gens", 4)
            workers = nsga2_cfg.get("workers", 8)
            if "fixed_rate" in search_cfg:
                runner._search_rate = search_cfg["fixed_rate"]

            from experiments.runners.run_blis import NSGA2_PARAM_SPACE
            param_space = None
            cmap = cfg.get("concept_map", {})
            tp_values = cmap.get("tp", {})
            if isinstance(tp_values, dict):
                tp_list = tp_values.get("value")
            else:
                tp_list = tp_values
            if isinstance(tp_list, list) and tp_list != NSGA2_PARAM_SPACE['tp']:
                param_space = dict(NSGA2_PARAM_SPACE)
                param_space['tp'] = tp_list

            logger.info(
                "Running legacy NSGA-II search (pop=%d, gens=%d, workers=%d)",
                pop_size, n_gens, workers,
            )
            results = runner.run_nsga2_search(
                search_seed=workload.seed,
                pop_size=pop_size,
                n_gens=n_gens,
                workers=workers,
                param_space=param_space,
            )
            pareto_count = sum(1 for r in results if r.metadata.pareto_optimal)
            logger.info(
                "NSGA-II legacy complete: %d evaluated, %d Pareto-optimal",
                len(results), pareto_count,
            )
        else:
            configs = generate_pruned_blis_configs(max_gpus=max_gpus)
            logger.info("Generated %d configs for brute-force sweep", len(configs))

            workers = getattr(args, "workers", 1)
            if workers > 1:
                shard_dir = raw_dir / "blis_shards"
                shard_dir.mkdir(parents=True, exist_ok=True)
                shard_paths = [shard_dir / f"shard-{i:03d}.jsonl" for i in range(workers)]

                logger.info(
                    "Parallel sweep: %d workers, %d configs/worker (approx)",
                    workers, len(configs) // workers,
                )

                processes = []
                for i in range(workers):
                    p = multiprocessing.Process(
                        target=_blis_worker,
                        args=(i, workers, configs, workload, shard_paths[i]),
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                total_written = 0
                with open(output, "w") as out:
                    for sp in shard_paths:
                        if sp.exists():
                            text = sp.read_text()
                            out.write(text)
                            total_written += text.count("\n")

                logger.info("Aggregated %d total results to %s", total_written, output)
            else:
                runner.run_batch(
                    configs,
                    hash_fn=lambda c: json.dumps(c, sort_keys=True).__hash__().__abs__().__str__()[:8],
                )

    elif args.tool == "aiconfigurator":
        from experiments.runners.run_aiconfigurator import AIConfiguratorRunner

        aic_cfg = tool_config(cfg, "aiconfigurator")
        search_cfg = aic_cfg.get("search", aic_cfg)
        max_gpus = search_cfg.get("max_gpus", res["max_gpus"])
        gpu_cost = res["gpu_cost_per_hour"]
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

        lmo_cfg = tool_config(cfg, "llm-optimizer")
        search_cfg = lmo_cfg.get("search", lmo_cfg)
        max_gpus = search_cfg.get("max_gpus", res["max_gpus"])
        gpu_cost = res["gpu_cost_per_hour"]
        runner = LLMOptimizerRunner(
            workload=workload,
            output_path=output,
            gpu_cost_per_hour=gpu_cost,
            max_gpus=max_gpus,
        )
        n = runner.run_full_sweep()
        logger.info("llm-optimizer: %d total results written", n)

    elif args.tool == "vidur":
        from experiments.runners.run_vidur import VidurRunner

        vidur_cfg = tool_config(cfg, "vidur")
        search_cfg = vidur_cfg.get("search", vidur_cfg)
        max_gpus = search_cfg.get("max_gpus", res["max_gpus"])
        gpu_cost = res["gpu_cost_per_hour"]
        runner = VidurRunner(
            workload=workload,
            output_path=output,
            gpu_cost_per_hour=gpu_cost,
            max_gpus=max_gpus,
            num_requests=search_cfg.get("num_requests", 500),
            start_qps=search_cfg.get("start_qps", 10.0),
            max_bisect_iters=search_cfg.get("max_bisect_iters", 8),
            lean=search_cfg.get("lean", True),
            slo_multiplier=search_cfg.get("slo_multiplier", 1.0),
        )
        n = runner.run_full_sweep()
        logger.info("Vidur: %d total results written", n)

    elif args.tool == "llmservingsim":
        from experiments.runners.run_llmservingsim import LLMServingSimRunner

        lss_cfg = tool_config(cfg, "llmservingsim")
        search_cfg = lss_cfg.get("search", lss_cfg)
        max_gpus = search_cfg.get("max_gpus", res["max_gpus"])
        gpu_cost = res["gpu_cost_per_hour"]
        runner = LLMServingSimRunner(
            workload=workload,
            output_path=output,
            gpu_cost_per_hour=gpu_cost,
            max_gpus=max_gpus,
            num_requests=search_cfg.get("num_requests", 500),
            start_rate=search_cfg.get("start_rate", 10.0),
            max_bisect_iters=search_cfg.get("max_bisect_iters", 8),
            lean=search_cfg.get("lean", True),
        )
        n = runner.run_full_sweep()
        logger.info("LLMServingSim: %d total results written", n)

    else:
        logger.warning("Runner for %s not yet implemented. See experiments/runners/", args.tool)

    elapsed = _time.monotonic() - sweep_start
    logger.info("Sweep %s completed in %.1fs (%.1f min)", args.tool, elapsed, elapsed / 60)


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
    sweep_parser.add_argument("--workers", type=int, default=1,
                              help="Parallel workers (default: 1; try 6 on 8-core Mac)")

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
