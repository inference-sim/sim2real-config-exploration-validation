import logging
from pathlib import Path

try:
    from aiconfigurator.cli import cli_default
except ImportError:
    cli_default = None  # type: ignore[assignment]

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

logger = logging.getLogger(__name__)


class AIConfiguratorRunner(BaseRunner):
    tool_name = "AIConfigurator"
    timeout_seconds = 120

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        gpu_cost_per_hour: float = 3.20,
        max_gpus: int = 8,
    ):
        super().__init__(workload, output_path)
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.max_gpus = max_gpus

    def evaluate_config(self, config: dict) -> ConfigResult:
        raise NotImplementedError("Use run_full_sweep() instead.")

    def run_full_sweep(self) -> int:
        """Call cli default for each total_gpus in {1..max_gpus}.

        Each call lets AIConfigurator enumerate all valid (TP, PP, replicas)
        triples for that GPU budget, run its internal batch size sweep, apply
        the TTFT SLO constraint, and return a Pareto frontier. Together, the
        8 calls cover all 25 valid topology triples satisfying
        TP * PP * replicas <= max_gpus.
        """
        if cli_default is None:
            logger.error("aiconfigurator not installed; skipping sweep")
            return 0

        total_written = 0

        for total_gpus in range(1, self.max_gpus + 1):
            logger.info("Running aiconfigurator cli default with total_gpus=%d", total_gpus)
            try:
                result = cli_default(
                    model_path=self.workload.model,
                    total_gpus=total_gpus,
                    system="h100_sxm",
                    backend="vllm",
                    isl=self.workload.isl_mean,
                    osl=self.workload.osl_mean,
                    ttft=float(self.workload.slo_ttft_mean_ms),
                )
            except Exception as e:
                logger.warning(
                    "cli default failed for total_gpus=%d: %s", total_gpus, e,
                )
                continue

            df = result.best_configs.get("agg")
            if df is None or df.empty:
                logger.info("No agg results for total_gpus=%d", total_gpus)
                continue

            for _, row in df.iterrows():
                tp = int(row["tp"])
                pp = int(row["pp"])
                # replicas not in DataFrame; derive from GPU budget
                replicas = total_gpus // int(row["num_total_gpus"])
                batch_size = int(row["bs"])
                ttft = float(row["ttft"])
                tpot = float(row["tpot"]) if row["tpot"] else None

                # tokens/s and request_rate are per-replica; scale to cluster
                cluster_throughput = float(row["tokens/s"]) * replicas
                cluster_qps = float(row["request_rate"]) * replicas

                num_gpus = tp * pp * replicas
                cost = num_gpus * self.gpu_cost_per_hour

                vllm_args = VllmArgs(
                    tensor_parallel_size=tp,
                    pipeline_parallel_size=pp,
                    num_replicas=replicas,
                    data_parallel_size=1,
                    max_num_seqs=batch_size,
                    max_num_batched_tokens=batch_size * (self.workload.isl_mean + self.workload.osl_mean),
                    enable_chunked_prefill=False,
                    block_size=16,
                )

                routing = RoutingConfig(strategy="round-robin") if replicas > 1 else None
                tool_cfg = ToolConfig(scheduler="agg")
                config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

                cr = ConfigResult(
                    tool=self.tool_name,
                    workload=self.workload,
                    vllm_args=vllm_args,
                    routing_config=routing,
                    tool_config=tool_cfg,
                    results=Results(
                        max_throughput_tok_s=cluster_throughput,
                        max_throughput_qps=cluster_qps,
                        ttft_mean_ms=ttft,
                        ttft_p50_ms=None,
                        ttft_p99_ms=None,
                        tpot_mean_ms=tpot,
                        meets_slo=ttft <= self.workload.slo_ttft_mean_ms,
                        cost_per_hour=cost,
                        cost_per_1k_tokens=(cost / cluster_throughput * 1000 / 3600)
                        if cluster_throughput > 0 else None,
                    ),
                    metadata=Metadata(
                        status="ok",
                        config_hash=config_hash,
                    ),
                )
                self.append_result(cr)
                total_written += 1

        logger.info(
            "AIConfigurator: %d results across total_gpus=1..%d",
            total_written, self.max_gpus,
        )
        return total_written
