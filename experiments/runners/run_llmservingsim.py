import logging
from pathlib import Path

from experiments.runners.base import BaseRunner
from experiments.schema.output import ConfigResult, WorkloadInfo

logger = logging.getLogger(__name__)


def build_llmservingsim_args(config: dict, dataset_path: str) -> list[str]:
    args = [
        "--dataset", dataset_path,
        "--num-instances", str(config["replicas"]),
        "--tp-size", str(config["tp"]),
        "--pp-size", str(config["pp"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["max_batched_tokens"]),
        "--block-size", str(config["block_size"]),
        "--network-backend", "analytical",
    ]
    if config["enable_chunked_prefill"]:
        args.append("--enable-chunked-prefill")
        if config["chunked_prefill_threshold"] is not None:
            args.extend([
                "--long-prefill-token-threshold",
                str(config["chunked_prefill_threshold"]),
            ])
    if config["prefix_caching"]:
        args.append("--enable-prefix-caching")
    if config.get("routing_policy"):
        args.extend(["--request-routing-policy", config["routing_policy"]])
    return args


class LLMServingSimRunner(BaseRunner):
    tool_name = "LLMServingSim"
    timeout_seconds = 300

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        dataset_path: str = "workloads/canonical_servegen_m-mid.jsonl",
    ):
        super().__init__(workload, output_path)
        self.dataset_path = dataset_path

    def evaluate_config(self, config: dict) -> ConfigResult:
        raise NotImplementedError(
            "LLMServingSim runner requires LLMServingSim installation. "
            "See estimators/LLMSERVINGSIM.md for setup."
        )
