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

VIDUR_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
VIDUR_DEVICE = "h100"
VIDUR_DIR = Path(__file__).resolve().parents[2] / "estimators" / "vidur"

BINARY_SEARCH_MAX_QPS = 500.0
BINARY_SEARCH_MIN_QPS = 0.1

LEAN_TP = [1, 2, 4, 8]
LEAN_PP = [1]
LEAN_SCHEDULERS = ["vllm", "sarathi"]
LEAN_BATCH_SIZES = [128, 512]
LEAN_ROUTING = ["round_robin"]

FULL_TP = [1, 2, 4, 8]
FULL_PP = [1, 2, 4]
FULL_SCHEDULERS = ["vllm", "sarathi", "orca"]
FULL_BATCH_SIZES = [32, 64, 128, 256, 512]
FULL_MAX_BATCHED_TOKENS = [2048, 4096, 8192]
FULL_CHUNK_SIZES = [1024, 2048, 4096]
FULL_BLOCK_SIZES = [16, 32]
FULL_ROUTING = ["round_robin", "lor", "random"]


class VidurRunner(BaseRunner):
    tool_name = "Vidur"
    timeout_seconds = 600

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        gpu_cost_per_hour: float = 3.20,
        max_gpus: int = 8,
        num_requests: int = 500,
        start_qps: float = 10.0,
        max_bisect_iters: int = 8,
        lean: bool = True,
        slo_multiplier: float = 1.0,
    ):
        super().__init__(workload, output_path)
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.max_gpus = max_gpus
        self.num_requests = num_requests
        self.start_qps = start_qps
        self.max_bisect_iters = max_bisect_iters
        self.lean = lean
        self.slo_multiplier = slo_multiplier
        self._effective_slo_ms = workload.slo_ttft_mean_ms * slo_multiplier

    def evaluate_config(self, config: dict) -> ConfigResult:
        tp = config["tp"]
        pp = config["pp"]
        replicas = config["replicas"]
        num_gpus = tp * pp * replicas
        cost = num_gpus * self.gpu_cost_per_hour

        vllm_args = self._config_to_vllm_args(config)
        routing = self._config_to_routing(config)
        tool_cfg = ToolConfig(vidur_scheduler_type=config.get("scheduler"))
        config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

        t0 = time.time()
        max_qps, metrics = self._binary_search_qps(config)
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

        throughput_tok_s = max_qps * self.workload.osl_mean
        meets_slo = metrics["ttft_mean_ms"] <= self._effective_slo_ms

        return ConfigResult(
            tool=self.tool_name,
            workload=self.workload,
            vllm_args=vllm_args,
            routing_config=routing,
            tool_config=tool_cfg,
            results=Results(
                max_throughput_tok_s=throughput_tok_s,
                max_throughput_qps=max_qps,
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

    def _binary_search_qps(self, config: dict) -> tuple[float, dict | None]:
        lo = 0.0
        hi = self.start_qps
        last_good_metrics = None

        while hi <= BINARY_SEARCH_MAX_QPS:
            metrics = self._run_simulation(config, hi)
            if metrics is None:
                return lo, last_good_metrics
            if metrics["ttft_mean_ms"] > self._effective_slo_ms:
                break
            lo = hi
            last_good_metrics = metrics
            hi *= 2.0

        if last_good_metrics is None:
            metrics = self._run_simulation(config, BINARY_SEARCH_MIN_QPS)
            if metrics and metrics["ttft_mean_ms"] <= self._effective_slo_ms:
                return BINARY_SEARCH_MIN_QPS, metrics
            return 0.0, None

        hi = min(hi, BINARY_SEARCH_MAX_QPS)

        for _ in range(self.max_bisect_iters):
            mid = (lo + hi) / 2.0
            metrics = self._run_simulation(config, mid)
            if metrics is None:
                hi = mid
                continue
            if metrics["ttft_mean_ms"] <= self._effective_slo_ms:
                lo = mid
                last_good_metrics = metrics
            else:
                hi = mid

        return lo, last_good_metrics

    def _run_simulation(self, config: dict, qps: float) -> dict | None:
        tmp_dir = tempfile.mkdtemp(prefix="vidur_")
        try:
            args = self._build_vidur_args(config, qps, tmp_dir)
            env = os.environ.copy()
            env["WANDB_MODE"] = "disabled"

            result = subprocess.run(
                ["python", "-m", "vidur.main"] + args,
                capture_output=True, text=True,
                timeout=self.timeout_seconds,
                cwd=str(VIDUR_DIR),
                env=env,
            )

            if result.returncode != 0:
                logger.debug("Vidur failed (QPS=%.1f): %s", qps, result.stderr[:300])
                return None

            return self._parse_output(tmp_dir)
        except subprocess.TimeoutExpired:
            logger.warning("Vidur timed out at QPS=%.1f", qps)
            return None
        except Exception as e:
            logger.warning("Vidur error at QPS=%.1f: %s", qps, e)
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _build_vidur_args(self, config: dict, qps: float, output_dir: str) -> list[str]:
        tp = config["tp"]
        pp = config["pp"]
        replicas = config["replicas"]
        scheduler = config.get("scheduler", "vllm")
        max_num_seqs = config.get("max_num_seqs", 128)

        args = [
            "--replica_config_device", VIDUR_DEVICE,
            "--replica_config_model_name", VIDUR_MODEL_NAME,
            "--replica_config_tensor_parallel_size", str(tp),
            "--replica_config_num_pipeline_stages", str(pp),
            "--cluster_config_num_replicas", str(replicas),
            "--replica_scheduler_config_type", scheduler,
            "--request_generator_config_type", "synthetic",
            "--synthetic_request_generator_config_num_requests", str(self.num_requests),
            "--length_generator_config_type", "fixed",
            "--fixed_request_length_generator_config_prefill_tokens",
            str(self.workload.isl_mean),
            "--fixed_request_length_generator_config_decode_tokens",
            str(self.workload.osl_mean),
            "--interval_generator_config_type", "poisson",
            "--poisson_request_interval_generator_config_qps", f"{qps:.2f}",
            "--metrics_config_output_dir", output_dir,
        ]

        if scheduler == "vllm":
            args += ["--vllm_scheduler_config_batch_size_cap", str(max_num_seqs)]
            mbt = config.get("max_batched_tokens", 4096)
            args += ["--vllm_scheduler_config_max_tokens_in_batch", str(mbt)]
        elif scheduler == "sarathi":
            args += ["--sarathi_scheduler_config_batch_size_cap", str(max_num_seqs)]
            cs = config.get("chunk_size", 2048)
            args += ["--sarathi_scheduler_config_chunk_size", str(cs)]

        if replicas > 1:
            routing = config.get("routing", "round_robin")
            args += ["--global_scheduler_config_type", routing]

        return args

    def _parse_output(self, output_dir: str) -> dict | None:
        output_path = Path(output_dir)
        subdirs = [d for d in output_path.iterdir() if d.is_dir()]
        if not subdirs:
            return None
        run_dir = subdirs[0]

        metrics_file = run_dir / "request_metrics.csv"
        if not metrics_file.exists():
            return None

        return self._parse_request_metrics(metrics_file)

    @staticmethod
    def _parse_request_metrics(metrics_file: Path) -> dict | None:
        try:
            lines = metrics_file.read_text().strip().split("\n")
            if len(lines) < 2:
                return None
            header = lines[0].split(",")
            ttft_idx = header.index("prefill_e2e_time")
        except (ValueError, IndexError):
            return None

        ttft_values = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) > ttft_idx:
                try:
                    ttft_values.append(float(parts[ttft_idx]))
                except ValueError:
                    continue

        if not ttft_values:
            return None

        ttft_arr = np.array(ttft_values)

        return {
            "ttft_mean_ms": float(np.mean(ttft_arr)) * 1000.0,
            "ttft_p50_ms": float(np.median(ttft_arr)) * 1000.0,
            "ttft_p99_ms": float(np.percentile(ttft_arr, 99)) * 1000.0,
        }

    def _enumerate_configs(self) -> list[dict]:
        if self.lean:
            tp_vals, pp_vals = LEAN_TP, LEAN_PP
            schedulers, batch_sizes = LEAN_SCHEDULERS, LEAN_BATCH_SIZES
            routing_list = LEAN_ROUTING
        else:
            tp_vals, pp_vals = FULL_TP, FULL_PP
            schedulers, batch_sizes = FULL_SCHEDULERS, FULL_BATCH_SIZES
            routing_list = FULL_ROUTING

        configs = []
        for tp in tp_vals:
            for pp in pp_vals:
                if tp * pp > self.max_gpus:
                    continue
                max_r = self.max_gpus // (tp * pp)
                replica_vals = [1] + ([max_r] if max_r > 1 else [])

                for replicas in replica_vals:
                    for scheduler in schedulers:
                        for bs in batch_sizes:
                            base = {
                                "tp": tp, "pp": pp, "replicas": replicas,
                                "scheduler": scheduler, "max_num_seqs": bs,
                            }
                            variants = self._scheduler_variants(base, scheduler)
                            for v in variants:
                                if replicas > 1:
                                    for r in routing_list:
                                        configs.append({**v, "routing": r})
                                else:
                                    configs.append(v)
        return configs

    def _scheduler_variants(self, base: dict, scheduler: str) -> list[dict]:
        if self.lean:
            if scheduler == "vllm":
                return [{**base, "max_batched_tokens": 4096}]
            elif scheduler == "sarathi":
                return [{**base, "chunk_size": 2048}]
            return [base]

        if scheduler == "vllm":
            return [{**base, "max_batched_tokens": mbt} for mbt in FULL_MAX_BATCHED_TOKENS]
        elif scheduler == "sarathi":
            return [{**base, "chunk_size": cs} for cs in FULL_CHUNK_SIZES]
        return [base]

    def _config_to_vllm_args(self, config: dict) -> VllmArgs:
        scheduler = config.get("scheduler", "vllm")
        enable_chunked = scheduler == "sarathi"
        mbt = config.get("max_batched_tokens", 4096)
        if scheduler == "sarathi":
            mbt = config.get("chunk_size", 2048)

        return VllmArgs(
            tensor_parallel_size=config["tp"],
            pipeline_parallel_size=config["pp"],
            num_replicas=config["replicas"],
            data_parallel_size=1,
            max_num_seqs=config.get("max_num_seqs", 128),
            max_num_batched_tokens=mbt,
            enable_chunked_prefill=enable_chunked,
            block_size=config.get("block_size", 16),
        )

    def _config_to_routing(self, config: dict) -> RoutingConfig | None:
        if config.get("replicas", 1) <= 1:
            return None
        routing = config.get("routing", "round_robin")
        strategy_map = {
            "round_robin": "round-robin",
            "lor": "least-outstanding",
            "random": "random",
        }
        return RoutingConfig(strategy=strategy_map.get(routing, routing))

    def run_full_sweep(self) -> int:
        configs = self._enumerate_configs()
        completed = self.load_completed_hashes()
        total_written = 0

        logger.info(
            "Vidur sweep: %d configs (%s mode)",
            len(configs), "lean" if self.lean else "full",
        )

        for i, config in enumerate(configs):
            vllm_args = self._config_to_vllm_args(config)
            routing = self._config_to_routing(config)
            tool_cfg = ToolConfig(vidur_scheduler_type=config.get("scheduler"))
            config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

            if config_hash in completed:
                continue

            logger.info(
                "[%d/%d] Vidur: TP=%d PP=%d R=%d sched=%s batch=%d",
                i + 1, len(configs),
                config["tp"], config["pp"], config["replicas"],
                config.get("scheduler", "vllm"), config.get("max_num_seqs", 128),
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

        logger.info("Vidur: %d results written", total_written)
        return total_written
