import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from experiments.schema.output import ConfigResult, Metadata, VllmArgs, WorkloadInfo

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    tool_name: str
    timeout_seconds: int

    def __init__(self, workload: WorkloadInfo, output_path: Path):
        self.workload = workload
        self.output_path = output_path

    @abstractmethod
    def evaluate_config(self, config: dict) -> ConfigResult:
        ...

    def append_result(self, result: ConfigResult) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(result.model_dump_json() + "\n")

    def load_completed_hashes(self) -> set[str]:
        if not self.output_path.exists():
            return set()
        hashes = set()
        for line in self.output_path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                h = data.get("metadata", {}).get("config_hash")
                if h:
                    hashes.add(h)
            except json.JSONDecodeError:
                continue
        return hashes

    def run_batch(
        self,
        configs: list[dict],
        hash_fn: Callable[[dict], str],
    ) -> None:
        completed = self.load_completed_hashes()
        total = len(configs)
        skipped = 0

        for i, config in enumerate(configs):
            config_hash = hash_fn(config)
            if config_hash in completed:
                skipped += 1
                continue

            logger.info("[%d/%d] Evaluating config %s", i + 1, total, config_hash)
            try:
                result = self.evaluate_config(config)
                result.metadata.config_hash = config_hash
                self.append_result(result)
                completed.add(config_hash)
            except Exception as e:
                logger.error("Config %s failed: %s", config_hash, e)
                fail_result = ConfigResult(
                    tool=self.tool_name,
                    workload=self.workload,
                    vllm_args=self._config_to_vllm_args(config),
                    results=None,
                    metadata=Metadata(status="crashed", config_hash=config_hash),
                )
                self.append_result(fail_result)
                completed.add(config_hash)

        if skipped:
            logger.info("Skipped %d already-completed configs", skipped)

    def _config_to_vllm_args(self, config: dict) -> VllmArgs:
        return VllmArgs(
            tensor_parallel_size=config.get("tp", 1),
            pipeline_parallel_size=config.get("pp", 1),
            num_instances=config.get("replicas", 1),
            data_parallel_size=config.get("dp", 1),
            max_num_seqs=config.get("max_num_seqs", 128),
            max_num_batched_tokens=config.get("max_batched_tokens", 4096),
            enable_chunked_prefill=config.get("enable_chunked_prefill", False),
            block_size=config.get("block_size", 16),
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_prefix_caching=config.get("prefix_caching", False),
            enforce_eager=False,
            swap_space=4,
        )
