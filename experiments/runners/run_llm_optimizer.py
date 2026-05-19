import json
import logging
import re
import subprocess
from pathlib import Path

from experiments.config.llm_optimizer_configs import build_grid_search_args
from experiments.config.topology import TopologyDP, enumerate_topologies_dp
from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

logger = logging.getLogger(__name__)


def run_llm_optimizer_estimate(
    model: str,
    num_gpus: int,
    input_len: int = 512,
    output_len: int = 256,
    constraints: str = "ttft:mean<300ms",
    timeout: int = 60,
) -> dict | None:
    """Run llm-optimizer estimate for a given GPU count.

    The estimate subcommand is analytical (no GPU needed) and determines
    the optimal TP/PP/batch configuration internally for the given GPU
    budget. It does not accept per-config flags like --tp or --max-batch-size.
    """
    cmd = [
        "llm-optimizer", "estimate",
        "--model", model,
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--gpu", "H100",
        "--num-gpus", str(num_gpus),
        "--framework", "vllm",
        "--constraints", constraints,
        "--target", "throughput",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning(
                "llm-optimizer estimate failed for %d GPUs: %s",
                num_gpus, result.stderr[:200],
            )
            return None
        return parse_estimate_output(result.stdout, num_gpus)
    except subprocess.TimeoutExpired:
        logger.warning("llm-optimizer estimate timed out for %d GPUs", num_gpus)
        return None
    except FileNotFoundError:
        logger.error("llm-optimizer binary not found")
        return None


def parse_estimate_output(output: str, num_gpus: int) -> dict | None:
    """Parse llm-optimizer estimate text output into structured metrics."""
    metrics = {"num_gpus": num_gpus}

    tput_match = re.search(r"Best Throughput.*?Output:\s*([\d.]+)\s*tokens/s", output, re.DOTALL)
    if tput_match:
        metrics["max_throughput_tok_s"] = float(tput_match.group(1))

    req_match = re.search(r"Requests:\s*([\d.]+)\s*req/s", output)
    if req_match:
        metrics["max_throughput_qps"] = float(req_match.group(1))

    constrained = re.search(
        r"Performance under Constraints.*?Concurrency:\s*(\d+).*?TTFT:\s*([\d.]+)\s*ms.*?ITL:\s*([\d.]+)\s*ms.*?Output throughput:\s*([\d.]+)\s*tokens/s",
        output, re.DOTALL,
    )
    if constrained:
        metrics["concurrency"] = int(constrained.group(1))
        metrics["ttft_mean_ms"] = float(constrained.group(2))
        metrics["tpot_mean_ms"] = float(constrained.group(3))
        metrics["constrained_throughput_tok_s"] = float(constrained.group(4))

    optimal_match = re.search(r"Empirical Optimal Concurrency:\s*(\d+)", output)
    if optimal_match:
        metrics["optimal_concurrency"] = int(optimal_match.group(1))

    # Parse TP/PP if reported
    tp_match = re.search(r"TP[:\s=]+(\d+)", output)
    if tp_match:
        metrics["tp"] = int(tp_match.group(1))
    pp_match = re.search(r"PP[:\s=]+(\d+)", output)
    if pp_match:
        metrics["pp"] = int(pp_match.group(1))

    if "max_throughput_tok_s" not in metrics and "constrained_throughput_tok_s" not in metrics:
        return None

    return metrics


def build_llm_optimizer_cmd(
    model: str,
    output_json: str = "results/raw/llm_optimizer.json",
    constraints: str = "ttft:mean<300ms",
) -> list[str]:
    """Build the llm-optimizer native grid search command (requires GPU)."""
    grid = build_grid_search_args()

    cmd = [
        "llm-optimizer",
        "--framework", "vllm",
        "--model", model,
        "--output-json", output_json,
        "--constraints", constraints,
        "--continue",
    ]

    for sa in grid["server_args"]:
        cmd.extend(["--server-args", sa])
    for ca in grid["client_args"]:
        cmd.extend(["--client-args", ca])

    return cmd


class LLMOptimizerRunner(BaseRunner):
    tool_name = "llm-optimizer"
    timeout_seconds = 60

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
        """Run llm-optimizer estimate for each unique GPU count.

        The estimate subcommand is analytical and determines the optimal
        config internally per GPU budget. It sweeps TP/PP/batch_size
        internally, so we call it once per distinct GPU count derived
        from the topology enumeration.

        For the full per-config sweep (11,200 configs), use the native
        grid search via build_llm_optimizer_cmd() which requires GPU hardware.
        """
        topologies = enumerate_topologies_dp(self.max_gpus)
        constraints = f"ttft:mean<{self.workload.slo_ttft_mean_ms}ms"

        # Deduplicate by GPU count since estimate only accepts --num-gpus
        seen_gpus = set()
        gpu_counts = []
        for topo in topologies:
            g = topo.total_gpus
            if g not in seen_gpus:
                seen_gpus.add(g)
                gpu_counts.append(g)
        gpu_counts.sort()

        total_written = 0

        for num_gpus in gpu_counts:
            logger.info("Running llm-optimizer estimate with %d GPUs", num_gpus)
            metrics = run_llm_optimizer_estimate(
                model=self.workload.model,
                num_gpus=num_gpus,
                input_len=self.workload.isl_mean,
                output_len=self.workload.osl_mean,
                constraints=constraints,
                timeout=self.timeout_seconds,
            )

            if not metrics:
                logger.warning("No results from llm-optimizer for %d GPUs", num_gpus)
                continue

            throughput = metrics.get(
                "constrained_throughput_tok_s",
                metrics.get("max_throughput_tok_s", 0),
            )
            ttft = metrics.get("ttft_mean_ms", 0)
            tpot = metrics.get("tpot_mean_ms")
            qps = metrics.get("max_throughput_qps", 0)
            concurrency = metrics.get(
                "concurrency",
                metrics.get("optimal_concurrency", 128),
            )

            # llm-optimizer internally picks TP; default to num_gpus as TP
            tp = metrics.get("tp", num_gpus)
            pp = metrics.get("pp", 1)
            dp = num_gpus // (tp * pp) if tp * pp <= num_gpus else 1

            vllm_args = VllmArgs(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                num_replicas=1,
                data_parallel_size=dp,
                max_num_seqs=concurrency,
                max_num_batched_tokens=concurrency * (self.workload.isl_mean + self.workload.osl_mean),
                enable_chunked_prefill=False,
                block_size=16,
            )

            tool_cfg = ToolConfig(
                scheduler="vllm-default",
                max_concurrency=concurrency,
            )
            config_hash = compute_config_hash(vllm_args, None, tool_cfg)
            cost = num_gpus * self.gpu_cost_per_hour
            meets_slo = ttft <= self.workload.slo_ttft_mean_ms

            cr = ConfigResult(
                tool=self.tool_name,
                workload=self.workload,
                vllm_args=vllm_args,
                routing_config=None,
                tool_config=tool_cfg,
                results=Results(
                    max_throughput_tok_s=throughput,
                    max_throughput_qps=qps,
                    ttft_mean_ms=ttft,
                    ttft_p50_ms=None,
                    ttft_p99_ms=None,
                    tpot_mean_ms=tpot,
                    meets_slo=meets_slo,
                    cost_per_hour=cost,
                    cost_per_1k_tokens=(cost / throughput * 1000 / 3600)
                    if throughput > 0 else None,
                ),
                metadata=Metadata(
                    status="ok",
                    config_hash=config_hash,
                ),
            )
            self.append_result(cr)
            total_written += 1

            logger.info(
                "  %d GPUs (TP=%d PP=%d DP=%d): %.0f tok/s, TTFT %.1fms, $%.2f/hr",
                num_gpus, tp, pp, dp, throughput, ttft, cost,
            )

        return total_written
