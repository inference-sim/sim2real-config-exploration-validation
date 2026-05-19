"""Shard entrypoint for parallel BLIS sweep on K8s.

Each pod computes its slice of the config space and evaluates it.
Reads from environment:
    JOB_COMPLETION_INDEX  - pod index (0-based, set by K8s Indexed Job)
    TOTAL_COMPLETIONS     - total number of pods
    BLIS_BINARY           - path to blis binary
    DEFAULTS_FILEPATH     - path to defaults.yaml
    RESULTS_DIR           - directory for per-shard output files
    EXPERIMENT_CONFIG     - (optional) path to experiment YAML config

Usage (local smoke test):
    TOTAL_COMPLETIONS=4 JOB_COMPLETION_INDEX=0 python3 -m experiments.runners.blis_shard
"""

import json
import logging
import os
import sys
from pathlib import Path

from experiments.config.blis_configs import generate_blis_configs
from experiments.run_all import load_config, workload_from_config, DEFAULT_CONFIG_PATH
from experiments.runners.run_blis import BLISRunner
from experiments.schema.output import compute_config_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    index = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    total = int(os.environ.get("TOTAL_COMPLETIONS", "512"))
    blis_binary = os.environ.get("BLIS_BINARY", "/data/config-exploration/bin/blis")
    defaults = os.environ.get("DEFAULTS_FILEPATH", "/data/config-exploration/defaults.yaml")
    hardware_config = os.environ.get("HARDWARE_CONFIG", "/data/config-exploration/hardware_config.json")
    results_dir = Path(os.environ.get("RESULTS_DIR", "/data/config-exploration/results/raw/blis"))
    config_path = os.environ.get("EXPERIMENT_CONFIG", "")

    if config_path:
        cfg = load_config(Path(config_path))
    else:
        cfg = load_config(DEFAULT_CONFIG_PATH)

    workload = workload_from_config(cfg)
    max_gpus = cfg.get("model", {}).get("max_gpus", 8)

    configs = generate_blis_configs(max_gpus=max_gpus)
    shard = configs[index::total]

    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / f"shard-{index:04d}.jsonl"

    runner = BLISRunner(
        workload=workload,
        output_path=output,
        blis_binary=blis_binary,
        defaults_filepath=defaults,
        hardware_config=hardware_config if os.path.exists(hardware_config) else None,
    )

    completed = runner.load_completed_hashes()
    logger.info(
        "Shard %d/%d: %d configs total, %d already done, %d to evaluate",
        index, total, len(shard), len(completed), len(shard) - len(completed),
    )

    newly_done = 0
    for i, config in enumerate(shard):
        config_hash = json.dumps(config, sort_keys=True).__hash__().__abs__().__str__()[:8]
        if config_hash in completed:
            continue

        result = runner.evaluate_config(config)
        result.metadata.config_hash = config_hash
        runner.append_result(result)
        completed.add(config_hash)
        newly_done += 1

        if newly_done % 10 == 0:
            logger.info(
                "  Shard %d: %d/%d evaluated (%d skipped)",
                index, newly_done, len(shard) - (len(completed) - newly_done), len(completed) - newly_done,
            )

    logger.info("Shard %d complete: %d new results written", index, newly_done)


if __name__ == "__main__":
    main()
