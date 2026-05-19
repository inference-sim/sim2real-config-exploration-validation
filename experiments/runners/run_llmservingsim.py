import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

logger = logging.getLogger(__name__)

LLMSERVINGSIM_MODEL = "meta-llama/Meta-Llama-3-8B"
LLMSERVINGSIM_HARDWARE = "H100"
LLMSERVINGSIM_DIR = Path(__file__).resolve().parents[2] / "estimators" / "LLMServingSim"

# H100 SXM 80GB specs
H100_MEM_SIZE_GB = 80
H100_MEM_BW_GBS = 3350
H100_LINK_BW_GBS = 900

AVAILABLE_TP_DEGREES = [1, 2, 4]

BINARY_SEARCH_MAX_RATE = 500.0
BINARY_SEARCH_MIN_RATE = 1.0

LEAN_TP = [1, 2, 4]
LEAN_PP = [1]
LEAN_BATCH_SIZES = [128, 512]
LEAN_ROUTING = ["LOAD"]

FULL_TP = [1, 2, 4]
FULL_PP = [1, 2, 4]
FULL_BATCH_SIZES = [32, 64, 128, 256, 512]
FULL_MAX_BATCHED_TOKENS = [2048, 4096, 8192]
FULL_THRESHOLDS = [0, 1024, 2048, 4096]
FULL_BLOCK_SIZES = [16, 32]
FULL_ROUTING = ["LOAD", "RR", "RAND"]


class LLMServingSimRunner(BaseRunner):
    tool_name = "LLMServingSim"
    timeout_seconds = 300

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        gpu_cost_per_hour: float = 3.20,
        max_gpus: int = 8,
        num_requests: int = 500,
        start_rate: float = 10.0,
        max_bisect_iters: int = 8,
        lean: bool = True,
    ):
        super().__init__(workload, output_path)
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.max_gpus = max_gpus
        self.num_requests = num_requests
        self.start_rate = start_rate
        self.max_bisect_iters = max_bisect_iters
        self.lean = lean

    def evaluate_config(self, config: dict) -> ConfigResult:
        tp = config["tp"]
        pp = config.get("pp", 1)
        replicas = config["replicas"]
        num_gpus = tp * pp * replicas
        cost = num_gpus * self.gpu_cost_per_hour

        vllm_args = self._config_to_vllm_args(config)
        routing = self._config_to_routing(config)
        tool_cfg = ToolConfig()
        config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

        t0 = time.time()
        max_rate, metrics = self._binary_search_rate(config)
        wall = time.time() - t0

        if metrics is None:
            return ConfigResult(
                tool=self.tool_name,
                workload=self.workload,
                vllm_args=vllm_args,
                routing_config=routing,
                tool_config=tool_cfg,
                results=None,
                metadata=Metadata(
                    status="crashed",
                    config_hash=config_hash,
                    wall_clock_seconds=wall,
                ),
            )

        throughput_tok_s = max_rate * self.workload.osl_mean
        meets_slo = metrics["ttft_mean_ms"] <= self.workload.slo_ttft_mean_ms

        return ConfigResult(
            tool=self.tool_name,
            workload=self.workload,
            vllm_args=vllm_args,
            routing_config=routing,
            tool_config=tool_cfg,
            results=Results(
                max_throughput_tok_s=throughput_tok_s,
                max_throughput_qps=max_rate,
                ttft_mean_ms=metrics["ttft_mean_ms"],
                ttft_p50_ms=metrics.get("ttft_p50_ms"),
                ttft_p99_ms=metrics.get("ttft_p99_ms"),
                tpot_mean_ms=metrics.get("tpot_mean_ms"),
                meets_slo=meets_slo,
                cost_per_hour=cost,
                cost_per_1k_tokens=(cost / throughput_tok_s * 1000 / 3600)
                if throughput_tok_s > 0 else None,
            ),
            metadata=Metadata(
                status="ok",
                config_hash=config_hash,
                wall_clock_seconds=wall,
                num_rate_probes=self.max_bisect_iters + 4,
            ),
        )

    def _binary_search_rate(self, config: dict) -> tuple[float, dict | None]:
        lo = 0.0
        hi = self.start_rate
        last_good_metrics = None

        while hi <= BINARY_SEARCH_MAX_RATE:
            metrics = self._run_simulation(config, hi)
            if metrics is None:
                return lo, last_good_metrics
            if metrics["ttft_mean_ms"] > self.workload.slo_ttft_mean_ms:
                break
            lo = hi
            last_good_metrics = metrics
            hi *= 2.0

        if last_good_metrics is None:
            metrics = self._run_simulation(config, BINARY_SEARCH_MIN_RATE)
            if metrics and metrics["ttft_mean_ms"] <= self.workload.slo_ttft_mean_ms:
                return BINARY_SEARCH_MIN_RATE, metrics
            return 0.0, None

        hi = min(hi, BINARY_SEARCH_MAX_RATE)

        for _ in range(self.max_bisect_iters):
            mid = (lo + hi) / 2.0
            metrics = self._run_simulation(config, mid)
            if metrics is None:
                hi = mid
                continue
            if metrics["ttft_mean_ms"] <= self.workload.slo_ttft_mean_ms:
                lo = mid
                last_good_metrics = metrics
            else:
                hi = mid

        return lo, last_good_metrics

    def _run_simulation(self, config: dict, rate: float) -> dict | None:
        tmp_dir = tempfile.mkdtemp(prefix="llmservingsim_", dir=str(LLMSERVINGSIM_DIR))
        try:
            tmp_rel = os.path.relpath(tmp_dir, LLMSERVINGSIM_DIR)
            cluster_path = Path(tmp_dir) / "cluster.json"
            workload_path = Path(tmp_dir) / "workload.jsonl"
            output_path = Path(tmp_dir) / "output.csv"

            cluster_cfg = self._build_cluster_config(config)
            cluster_path.write_text(json.dumps(cluster_cfg, indent=2))

            self._generate_workload_jsonl(workload_path, rate, self.num_requests)

            args = self._build_sim_args(
                config,
                Path(tmp_rel) / "cluster.json",
                Path(tmp_rel) / "workload.jsonl",
                Path(tmp_rel) / "output.csv",
            )

            result = subprocess.run(
                ["python", "-m", "serving"] + args,
                capture_output=True, text=True,
                timeout=self.timeout_seconds,
                cwd=str(LLMSERVINGSIM_DIR),
            )

            if result.returncode != 0:
                logger.debug(
                    "LLMServingSim failed (rate=%.1f): %s",
                    rate, result.stderr[:300],
                )
                return None

            if not output_path.exists():
                return None

            return self._parse_output_csv(output_path)
        except subprocess.TimeoutExpired:
            logger.warning("LLMServingSim timed out at rate=%.1f", rate)
            return None
        except Exception as e:
            logger.warning("LLMServingSim error at rate=%.1f: %s", rate, e)
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _build_cluster_config(self, config: dict) -> dict:
        tp = config["tp"]
        pp = config.get("pp", 1)
        replicas = config["replicas"]

        instance = {
            "model_name": LLMSERVINGSIM_MODEL,
            "hardware": LLMSERVINGSIM_HARDWARE,
            "npu_mem": {
                "mem_size": H100_MEM_SIZE_GB,
                "mem_bw": H100_MEM_BW_GBS,
                "mem_latency": 0,
            },
            "tp_size": tp,
            "num_npus": tp * pp,
            "pd_type": None,
        }

        return {
            "num_nodes": 1,
            "link_bw": H100_LINK_BW_GBS,
            "link_latency": 0,
            "nodes": [{
                "num_instances": replicas,
                "cpu_mem": {
                    "mem_size": 1024,
                    "mem_bw": 512,
                    "mem_latency": 0,
                },
                "instances": [dict(instance) for _ in range(replicas)],
            }],
        }

    def _generate_workload_jsonl(
        self, output_path: Path, rate: float, num_requests: int,
    ) -> None:
        rng = np.random.default_rng(self.workload.seed)
        inter_arrivals = rng.exponential(1.0 / rate, size=num_requests)
        arrival_times_s = np.cumsum(inter_arrivals)

        with open(output_path, "w") as f:
            for i in range(num_requests):
                arrival_ns = int(arrival_times_s[i] * 1_000_000_000)
                record = {
                    "input_toks": self.workload.isl_mean,
                    "output_toks": self.workload.osl_mean,
                    "arrival_time_ns": arrival_ns,
                }
                f.write(json.dumps(record) + "\n")

    def _build_sim_args(
        self, config: dict, cluster_path: Path, workload_path: Path, output_path: Path,
    ) -> list[str]:
        args = [
            "--cluster-config", str(cluster_path),
            "--dataset", str(workload_path),
            "--output", str(output_path),
            "--max-num-seqs", str(config.get("max_num_seqs", 128)),
            "--max-num-batched-tokens", str(config.get("max_batched_tokens", 4096)),
            "--block-size", str(config.get("block_size", 16)),
            "--num-reqs", str(self.num_requests),
            "--dtype", "bfloat16",
            "--log-level", "WARNING",
        ]

        threshold = config.get("long_prefill_token_threshold", 0)
        if threshold > 0:
            args += ["--long-prefill-token-threshold", str(threshold)]

        if config.get("prefix_caching", False):
            args.append("--enable-prefix-caching")
        else:
            args.append("--no-enable-prefix-caching")

        if config["replicas"] > 1:
            routing = config.get("routing_policy", "LOAD")
            args += ["--request-routing-policy", routing]

        return args

    @staticmethod
    def _parse_output_csv(csv_path: Path) -> dict | None:
        try:
            lines = csv_path.read_text().strip().split("\n")
        except Exception:
            return None

        if len(lines) < 2:
            return None

        header = lines[0].split(",")
        try:
            ttft_idx = header.index("TTFT")
            tpot_idx = header.index("TPOT")
        except ValueError:
            return None

        ttft_values = []
        tpot_values = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(ttft_idx, tpot_idx):
                continue
            try:
                ttft_ns = float(parts[ttft_idx])
                tpot_ns = float(parts[tpot_idx])
                ttft_values.append(ttft_ns)
                tpot_values.append(tpot_ns)
            except (ValueError, IndexError):
                continue

        if not ttft_values:
            return None

        ttft_arr = np.array(ttft_values)
        tpot_arr = np.array(tpot_values)

        return {
            "ttft_mean_ms": float(np.mean(ttft_arr)) / 1_000_000.0,
            "ttft_p50_ms": float(np.median(ttft_arr)) / 1_000_000.0,
            "ttft_p99_ms": float(np.percentile(ttft_arr, 99)) / 1_000_000.0,
            "tpot_mean_ms": float(np.mean(tpot_arr)) / 1_000_000.0,
        }

    def _enumerate_configs(self) -> list[dict]:
        if self.lean:
            tp_vals, pp_vals = LEAN_TP, LEAN_PP
            batch_sizes = LEAN_BATCH_SIZES
            routing_list = LEAN_ROUTING
            mbt_list = [4096]
            threshold_list = [0]
            block_sizes = [16]
            prefix_opts = [False]
        else:
            tp_vals, pp_vals = FULL_TP, FULL_PP
            batch_sizes = FULL_BATCH_SIZES
            routing_list = FULL_ROUTING
            mbt_list = FULL_MAX_BATCHED_TOKENS
            threshold_list = FULL_THRESHOLDS
            block_sizes = FULL_BLOCK_SIZES
            prefix_opts = [False, True]

        configs = []
        for tp in tp_vals:
            if tp not in AVAILABLE_TP_DEGREES:
                continue
            for pp in pp_vals:
                if tp * pp > self.max_gpus:
                    continue
                max_r = self.max_gpus // (tp * pp)
                replica_vals = [1] + ([max_r] if max_r > 1 else [])

                for replicas in replica_vals:
                    for bs in batch_sizes:
                        for mbt in mbt_list:
                            for threshold in threshold_list:
                                if threshold >= mbt:
                                    continue
                                for block_size in block_sizes:
                                    for prefix in prefix_opts:
                                        base = {
                                            "tp": tp, "pp": pp,
                                            "replicas": replicas,
                                            "max_num_seqs": bs,
                                            "max_batched_tokens": mbt,
                                            "long_prefill_token_threshold": threshold,
                                            "block_size": block_size,
                                            "prefix_caching": prefix,
                                        }
                                        if replicas > 1:
                                            for r in routing_list:
                                                configs.append(
                                                    {**base, "routing_policy": r}
                                                )
                                        else:
                                            configs.append(base)
        return configs

    def _config_to_vllm_args(self, config: dict) -> VllmArgs:
        threshold = config.get("long_prefill_token_threshold", 0)
        return VllmArgs(
            tensor_parallel_size=config["tp"],
            pipeline_parallel_size=config.get("pp", 1),
            num_replicas=config["replicas"],
            data_parallel_size=1,
            max_num_seqs=config.get("max_num_seqs", 128),
            max_num_batched_tokens=config.get("max_batched_tokens", 4096),
            enable_chunked_prefill=threshold > 0,
            block_size=config.get("block_size", 16),
            enable_prefix_caching=config.get("prefix_caching", False),
        )

    def _config_to_routing(self, config: dict) -> RoutingConfig | None:
        if config.get("replicas", 1) <= 1:
            return None
        routing = config.get("routing_policy", "LOAD")
        strategy_map = {
            "LOAD": "least-loaded",
            "RR": "round-robin",
            "RAND": "random",
        }
        return RoutingConfig(strategy=strategy_map.get(routing, routing))

    def run_full_sweep(self) -> int:
        configs = self._enumerate_configs()
        completed = self.load_completed_hashes()
        total_written = 0

        logger.info(
            "LLMServingSim sweep: %d configs (%s mode)",
            len(configs), "lean" if self.lean else "full",
        )

        for i, config in enumerate(configs):
            vllm_args = self._config_to_vllm_args(config)
            routing = self._config_to_routing(config)
            tool_cfg = ToolConfig()
            config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

            if config_hash in completed:
                continue

            logger.info(
                "[%d/%d] LLMServingSim: TP=%d PP=%d R=%d batch=%d",
                i + 1, len(configs),
                config["tp"], config.get("pp", 1), config["replicas"],
                config.get("max_num_seqs", 128),
            )

            try:
                result = self.evaluate_config(config)
                result.metadata.config_hash = config_hash
                self.append_result(result)
                completed.add(config_hash)
                total_written += 1

                if result.results:
                    logger.info(
                        "  -> %.0f tok/s, TTFT %.1fms, QPS %.1f",
                        result.results.max_throughput_tok_s,
                        result.results.ttft_mean_ms,
                        result.results.max_throughput_qps,
                    )
            except Exception as e:
                logger.error("Config %s failed: %s", config_hash, e)

        logger.info("LLMServingSim: %d results written", total_written)
        return total_written
