import logging
import re
import subprocess
from pathlib import Path

from experiments.config.topology import Topology, enumerate_topologies
from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

logger = logging.getLogger(__name__)

MAX_NUM_SEQS = (32, 64, 128, 256, 512)


def run_aiconfigurator_estimate(
    model: str,
    tp: int,
    pp: int,
    batch_size: int,
    isl: int = 512,
    osl: int = 256,
    timeout: int = 30,
) -> dict | None:
    """Run aiconfigurator cli estimate in agg mode for a single config point.

    Returns parsed metrics dict or None on failure.
    """
    cmd = [
        "aiconfigurator", "cli", "estimate",
        "--model-path", model,
        "--system", "h100_sxm",
        "--estimate-mode", "agg",
        "--backend", "vllm",
        "--isl", str(isl),
        "--osl", str(osl),
        "--tp-size", str(tp),
        "--pp-size", str(pp),
        "--batch-size", str(batch_size),
        "--no-color",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "aiconfigurator estimate timed out (TP=%d PP=%d BS=%d)", tp, pp, batch_size,
        )
        return None

    if result.returncode != 0:
        logger.debug(
            "aiconfigurator failed (TP=%d PP=%d BS=%d): %s",
            tp, pp, batch_size, result.stderr[:200],
        )
        return None

    return parse_estimate_output(result.stdout, tp, pp, batch_size)


def parse_estimate_output(output: str, tp: int, pp: int, batch_size: int) -> dict | None:
    """Parse aiconfigurator estimate text output into structured metrics."""
    metrics = {"tp": tp, "pp": pp, "batch_size": batch_size}

    ttft_match = re.search(r"TTFT:\s*([\d.]+)\s*ms", output)
    if ttft_match:
        metrics["ttft_ms"] = float(ttft_match.group(1))

    tpot_match = re.search(r"TPOT:\s*([\d.]+)\s*ms", output)
    if tpot_match:
        metrics["tpot_ms"] = float(tpot_match.group(1))

    tput_match = re.search(r"tokens/s:\s*([\d,.]+)", output)
    if tput_match:
        metrics["tokens_per_sec"] = float(tput_match.group(1).replace(",", ""))

    seq_match = re.search(r"seq/s:\s*([\d,.]+)", output)
    if seq_match:
        metrics["qps"] = float(seq_match.group(1).replace(",", ""))

    if "tokens_per_sec" not in metrics:
        return None

    return metrics


class AIConfiguratorRunner(BaseRunner):
    tool_name = "AIConfigurator"
    timeout_seconds = 30

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
        """Run aiconfigurator estimate for all topologies x batch sizes.

        Enumerates all valid (TP, PP, replicas) triples and sweeps
        max_num_seqs values. Each call to aiconfigurator cli estimate
        returns a single-point prediction in agg mode.

        Per the spec, AIConfigurator assumes linear throughput scaling
        with replicas (no routing model), so single-replica throughput
        is multiplied by replica count.
        """
        topologies = enumerate_topologies(self.max_gpus)
        total_written = 0

        for topo in topologies:
            for batch_size in MAX_NUM_SEQS:
                metrics = run_aiconfigurator_estimate(
                    model=self.workload.model,
                    tp=topo.tp,
                    pp=topo.pp,
                    batch_size=batch_size,
                    isl=self.workload.isl_mean,
                    osl=self.workload.osl_mean,
                    timeout=self.timeout_seconds,
                )

                if not metrics:
                    continue

                single_throughput = metrics["tokens_per_sec"]
                single_qps = metrics.get("qps", 0)
                ttft = metrics.get("ttft_ms", 0)
                tpot = metrics.get("tpot_ms", 0)

                cluster_throughput = single_throughput * topo.replicas
                cluster_qps = single_qps * topo.replicas
                num_gpus = topo.tp * topo.pp * topo.replicas

                vllm_args = VllmArgs(
                    tensor_parallel_size=topo.tp,
                    pipeline_parallel_size=topo.pp,
                    num_instances=topo.replicas,
                    data_parallel_size=1,
                    max_num_seqs=batch_size,
                    max_num_batched_tokens=batch_size * (self.workload.isl_mean + self.workload.osl_mean),
                    enable_chunked_prefill=False,
                    block_size=16,
                )

                routing = None
                if topo.replicas > 1:
                    routing = RoutingConfig(strategy="round-robin")

                tool_cfg = ToolConfig(scheduler="agg")
                config_hash = compute_config_hash(vllm_args, routing, tool_cfg)
                cost = num_gpus * self.gpu_cost_per_hour
                meets_slo = ttft <= self.workload.slo_ttft_mean_ms

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
                        meets_slo=meets_slo,
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
            "AIConfigurator: %d results across %d topologies x %d batch sizes",
            total_written, len(topologies), len(MAX_NUM_SEQS),
        )
        return total_written
