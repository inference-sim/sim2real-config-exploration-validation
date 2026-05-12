import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

logger = logging.getLogger(__name__)

SLO_TTFT_MEAN_MS = 300
BINARY_SEARCH_ITERATIONS = 8
INITIAL_RATE = 10.0
MAX_RATE = 10000.0


def build_blis_args(
    config: dict,
    model: str,
    rate: float,
    metrics_path: str = "/dev/stdout",
    workload_spec: str | None = None,
    defaults_filepath: str | None = None,
    num_requests: int = 10000,
    seed: int = 42,
) -> list[str]:
    args = [
        "--model", model,
        "--hardware", "H100",
        "--tp", str(config["tp"]),
        "--num-instances", str(config["replicas"]),
        "--rate", str(rate),
        "--num-requests", str(num_requests),
        "--seed", str(seed),
        "--latency-model", "trained-physics",
        "--max-num-running-reqs", str(config["max_num_seqs"]),
        "--max-num-scheduled-tokens", str(config["max_batched_tokens"]),
        "--long-prefill-token-threshold", str(config["chunked_prefill_threshold"]),
        "--block-size-in-tokens", str(config["block_size"]),
        "--scheduler", config["scheduler"],
        "--preemption-policy", config["preemption_policy"],
        "--metrics-path", metrics_path,
    ]

    if defaults_filepath:
        args.extend(["--defaults-filepath", defaults_filepath])

    if workload_spec:
        args.extend(["--workload-spec", workload_spec])

    if config["routing_policy"] is not None:
        args.extend(["--routing-policy", config["routing_policy"]])
        if config.get("routing_scorers"):
            args.extend(["--routing-scorers", config["routing_scorers"]])

    if config["admission_policy"] is not None:
        args.extend(["--admission-policy", config["admission_policy"]])

    return args


class BLISRunner(BaseRunner):
    tool_name = "inference-sim"
    timeout_seconds = 60

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        blis_binary: str = "./estimators/inference-sim/blis",
        defaults_filepath: str = "./estimators/inference-sim/defaults.yaml",
        workload_spec: str | None = None,
    ):
        super().__init__(workload, output_path)
        self.blis_binary = blis_binary
        self.defaults_filepath = defaults_filepath
        self.workload_spec = workload_spec

    def _run_single(self, config: dict, rate: float) -> dict | None:
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".json", delete=True
        ) as metrics_file:
            args = build_blis_args(
                config, model=self.workload.model, rate=rate,
                metrics_path=metrics_file.name,
                defaults_filepath=self.defaults_filepath,
                num_requests=self.workload.num_requests,
                seed=self.workload.seed,
            )
            try:
                result = subprocess.run(
                    [self.blis_binary, "run"] + args,
                    capture_output=True, text=True,
                    timeout=self.timeout_seconds,
                )
                if result.returncode != 0:
                    logger.warning("blis run failed at rate=%.1f: %s", rate, result.stderr[:200])
                    return None
                return json.loads(metrics_file.read())
            except subprocess.TimeoutExpired:
                logger.warning("blis run timed out at rate=%.1f", rate)
                return None
            except json.JSONDecodeError:
                logger.warning("Failed to parse blis output at rate=%.1f", rate)
                return None

    def evaluate_config(self, config: dict) -> ConfigResult:
        start = time.monotonic()

        lo, hi = 0.0, INITIAL_RATE
        best_metrics = None
        best_rate = 0.0
        probes = 0

        while hi <= MAX_RATE:
            metrics = self._run_single(config, hi)
            probes += 1
            if metrics is None or metrics.get("ttft_mean_ms", 0) > SLO_TTFT_MEAN_MS:
                break
            best_metrics = metrics
            best_rate = hi
            lo = hi
            hi *= 2
        else:
            hi = MAX_RATE

        for _ in range(BINARY_SEARCH_ITERATIONS):
            mid = (lo + hi) / 2
            metrics = self._run_single(config, mid)
            probes += 1
            if metrics and metrics.get("ttft_mean_ms", 0) <= SLO_TTFT_MEAN_MS:
                best_metrics = metrics
                best_rate = mid
                lo = mid
            else:
                hi = mid

        elapsed = time.monotonic() - start

        vllm_args = VllmArgs(
            tensor_parallel_size=config["tp"],
            pipeline_parallel_size=1,
            num_instances=config["replicas"],
            data_parallel_size=1,
            max_num_seqs=config["max_num_seqs"],
            max_num_batched_tokens=config["max_batched_tokens"],
            enable_chunked_prefill=config["chunked_prefill_threshold"] > 0,
            block_size=config["block_size"],
        )

        routing = None
        if config["routing_policy"] is not None:
            if config["routing_policy"] == "weighted" and config.get("routing_scorers"):
                routing = RoutingConfig(
                    strategy="weighted-scoring",
                    scorers=config["routing_scorers"],
                    picker="max-score",
                )
            elif config["routing_policy"] == "round-robin":
                routing = RoutingConfig(strategy="round-robin")
            elif config["routing_policy"] == "least-loaded":
                routing = RoutingConfig(
                    strategy="least-loaded",
                    scorers="queue-depth:1",
                    picker="max-score",
                )

        tool_cfg = ToolConfig(
            scheduler=config["scheduler"],
            admission_policy=config.get("admission_policy"),
            preemption_policy=config["preemption_policy"],
        )

        config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

        if best_metrics is None:
            return ConfigResult(
                tool=self.tool_name, workload=self.workload,
                vllm_args=vllm_args, routing_config=routing, tool_config=tool_cfg,
                results=None,
                metadata=Metadata(
                    status="unconverged", wall_clock_seconds=elapsed,
                    num_rate_probes=probes, config_hash=config_hash,
                ),
            )

        cost = config["tp"] * config["replicas"] * 3.20
        throughput_tok_s = best_metrics.get("tokens_per_sec", 0)

        return ConfigResult(
            tool=self.tool_name, workload=self.workload,
            vllm_args=vllm_args, routing_config=routing, tool_config=tool_cfg,
            results=Results(
                max_throughput_tok_s=throughput_tok_s,
                max_throughput_qps=best_rate,
                ttft_mean_ms=best_metrics.get("ttft_mean_ms", 0),
                ttft_p50_ms=best_metrics.get("ttft_p50_ms"),
                ttft_p99_ms=best_metrics.get("ttft_p99_ms"),
                tpot_mean_ms=best_metrics.get("itl_mean_ms"),
                meets_slo=best_metrics.get("ttft_mean_ms", 999) <= SLO_TTFT_MEAN_MS,
                cost_per_hour=cost,
                cost_per_1k_tokens=(cost / throughput_tok_s * 1000 / 3600)
                if throughput_tok_s > 0 else None,
            ),
            metadata=Metadata(
                status="ok", wall_clock_seconds=elapsed,
                num_rate_probes=probes, config_hash=config_hash,
            ),
        )
