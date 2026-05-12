import logging
from pathlib import Path

import yaml

from experiments.runners.base import BaseRunner
from experiments.schema.output import ConfigResult, WorkloadInfo

logger = logging.getLogger(__name__)


def build_vidur_config_yaml(config: dict, model: str) -> str:
    vidur_config = {
        "cluster_config": {
            "num_replicas": config["replicas"],
            "replica_config": {
                "model_name": model,
                "tensor_parallel_size": config["tp"],
                "pipeline_parallel_size": config["pp"],
                "device": "h100",
                "network_device": "a100_pairwise_nvlink",
            },
            "replica_scheduler_config": {
                "type": config["scheduler_type"],
                "batch_size_cap": config["max_num_seqs"],
                "block_size": config["block_size"],
            },
        },
    }

    sched = vidur_config["cluster_config"]["replica_scheduler_config"]
    if config["scheduler_type"] == "vllm" and config.get("max_tokens_in_batch"):
        sched["max_tokens_in_batch"] = config["max_tokens_in_batch"]
    if config["scheduler_type"] == "sarathi" and config.get("chunk_size"):
        sched["chunk_size"] = config["chunk_size"]

    if config["replicas"] > 1 and config.get("routing"):
        vidur_config["cluster_config"]["global_scheduler_config"] = {
            "type": config["routing"],
        }

    return yaml.dump(vidur_config, default_flow_style=False)


class VidurRunner(BaseRunner):
    tool_name = "Vidur"
    timeout_seconds = 600

    def __init__(self, workload: WorkloadInfo, output_path: Path):
        super().__init__(workload, output_path)

    def evaluate_config(self, config: dict) -> ConfigResult:
        raise NotImplementedError(
            "Vidur runner requires Vidur installation. "
            "See estimators/VIDUR.md for setup."
        )
