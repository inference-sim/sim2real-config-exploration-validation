import json
import logging
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

SLO_TTFT_MEAN_MS = 300
BINARY_SEARCH_ITERATIONS = 5
INITIAL_RATE = 10.0
MAX_RATE = 10000.0
SEARCH_NUM_REQUESTS = 1000
CONFIRM_BISECT_ITERS = 4

HALVING_STAGES = [
    {"num_requests": 100, "bisect_iters": 3, "keep_ratio": 0.25},
    {"num_requests": 500, "bisect_iters": 4, "keep_ratio": 0.5},
    {"num_requests": 1000, "bisect_iters": 5, "keep_ratio": 1.0},
]
HALVING_MIN_SURVIVORS_PER_TOPO = 1
MAX_PARALLEL_WORKERS = 8

# Search-mode constants (from search_blis.py)
SEARCH_TP_OPTIONS = [1, 2, 4, 8]
SEARCH_VALID_TP_INSTANCES = [
    (tp, ni)
    for tp in SEARCH_TP_OPTIONS
    for ni in range(1, 8 // tp + 1)
]
SEARCH_TP_CONFIGS_ALL = [(1, 8), (2, 4), (4, 2), (8, 1)]
SEARCH_TP_CONFIGS_LEAN = [(4, 2), (8, 1)]
SEARCH_SCHEDULERS = ["fcfs", "priority-fcfs", "sjf", "reverse-priority"]
SEARCH_MAX_RUNNING = [32, 64, 128, 256, 512]
SEARCH_MAX_TOKENS = [2048, 4096, 8192]
SEARCH_PREFILL_THRESHOLDS = [0, 1024, 2048, 4096]
SEARCH_BLOCK_SIZES = [16, 32]
SEARCH_ROUTING = ["round-robin", "least-loaded"]
SEARCH_ADMISSION = ["always-admit", "tier-shed"]
SEARCH_PREEMPTION = ["fcfs", "priority"]
DEFAULT_FITNESS_WEIGHTS = "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"


def build_blis_args(
    config: dict,
    model: str,
    rate: float,
    metrics_path: str = "/dev/stdout",
    workload_spec: str | None = None,
    defaults_filepath: str | None = None,
    hardware_config: str | None = None,
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

    if hardware_config:
        args.extend(["--hardware-config", hardware_config])

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
    timeout_seconds = 180

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        blis_binary: str = "./estimators/inference-sim/blis",
        defaults_filepath: str = "./estimators/inference-sim/defaults.yaml",
        hardware_config: str | None = None,
        workload_spec: str | None = None,
        search_budget: int = 20,
        phase1_k: int = 1,
        phase1_profiles: str = "bracket",
        tp_candidates: str = "lean",
        skip_phase2: bool = False,
        fitness_weights: str = DEFAULT_FITNESS_WEIGHTS,
    ):
        super().__init__(workload, output_path)
        self.blis_binary = blis_binary
        self.defaults_filepath = defaults_filepath
        self.hardware_config = hardware_config
        self.workload_spec = workload_spec
        self.search_budget = search_budget
        self.phase1_k = phase1_k
        self.phase1_profiles = phase1_profiles
        self.tp_candidates = tp_candidates
        self.skip_phase2 = skip_phase2
        self.fitness_weights = fitness_weights

    def _run_single(
        self, config: dict, rate: float,
        num_requests: int | None = None,
        timeout: int | None = None,
    ) -> dict | None:
        nr = num_requests if num_requests is not None else self.workload.num_requests
        to = timeout if timeout is not None else self.timeout_seconds
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".json", delete=True
        ) as metrics_file:
            args = build_blis_args(
                config, model=self.workload.model, rate=rate,
                metrics_path=metrics_file.name,
                defaults_filepath=self.defaults_filepath,
                hardware_config=self.hardware_config,
                num_requests=nr,
                seed=self.workload.seed,
            )
            try:
                result = subprocess.run(
                    [self.blis_binary, "run"] + args,
                    capture_output=True, text=True,
                    timeout=to,
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
        search_nr = min(SEARCH_NUM_REQUESTS, self.workload.num_requests)

        lo, hi = 0.0, INITIAL_RATE
        best_metrics = None
        best_rate = 0.0
        probes = 0
        timed_out = False

        # Phase 1: exponential ramp with fast probes
        while hi <= MAX_RATE:
            metrics = self._run_single(
                config, hi, num_requests=search_nr, timeout=60,
            )
            probes += 1
            if metrics is None:
                if probes == 1:
                    timed_out = True
                break
            if metrics.get("ttft_mean_ms", 0) > SLO_TTFT_MEAN_MS:
                break
            best_metrics = metrics
            best_rate = hi
            lo = hi
            hi *= 2
        else:
            hi = MAX_RATE

        # Phase 2: binary search with fast probes (skip if first probe timed out)
        if not timed_out:
            for _ in range(BINARY_SEARCH_ITERATIONS):
                mid = (lo + hi) / 2
                metrics = self._run_single(
                    config, mid, num_requests=search_nr, timeout=60,
                )
                probes += 1
                if metrics and metrics.get("ttft_mean_ms", 0) <= SLO_TTFT_MEAN_MS:
                    best_metrics = metrics
                    best_rate = mid
                    lo = mid
                else:
                    hi = mid

            # Phase 3: confirm at full num_requests, bisect down from
            # a conservative starting point (half the cheap-search rate)
            if best_rate > 0:
                lo_full, hi_full = 0.0, best_rate * 0.5
                for _ in range(CONFIRM_BISECT_ITERS):
                    mid_full = (lo_full + hi_full) / 2
                    confirm = self._run_single(config, mid_full)
                    probes += 1
                    if confirm and confirm.get("ttft_mean_ms", 0) <= SLO_TTFT_MEAN_MS:
                        best_metrics = confirm
                        best_rate = mid_full
                        lo_full = mid_full
                    else:
                        hi_full = mid_full

        elapsed = time.monotonic() - start

        vllm_args = VllmArgs(
            tensor_parallel_size=config["tp"],
            pipeline_parallel_size=1,
            num_replicas=config["replicas"],
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

    def _find_max_rate(
        self, config: dict, num_requests: int, bisect_iters: int,
    ) -> tuple[float, dict | None]:
        """Find max sustainable rate via exponential ramp + bisection.

        Returns (best_rate, best_metrics). Uses early termination on first
        timeout. Lighter than evaluate_config (no ConfigResult construction).
        """
        lo, hi = 0.0, INITIAL_RATE
        best_metrics = None
        best_rate = 0.0

        while hi <= MAX_RATE:
            metrics = self._run_single(config, hi, num_requests=num_requests)
            if metrics is None:
                if best_rate == 0.0:
                    return 0.0, None
                break
            if metrics.get("ttft_mean_ms", 0) > SLO_TTFT_MEAN_MS:
                break
            best_metrics = metrics
            best_rate = hi
            lo = hi
            hi *= 2
        else:
            hi = MAX_RATE

        for _ in range(bisect_iters):
            mid = (lo + hi) / 2
            metrics = self._run_single(config, mid, num_requests=num_requests)
            if metrics and metrics.get("ttft_mean_ms", 0) <= SLO_TTFT_MEAN_MS:
                best_metrics = metrics
                best_rate = mid
                lo = mid
            else:
                hi = mid

        return best_rate, best_metrics

    def _eval_rate_worker(
        self, config: dict, num_requests: int, bisect_iters: int,
    ) -> tuple[float, dict, dict | None]:
        """Worker function for parallel rate evaluation."""
        rate, metrics = self._find_max_rate(config, num_requests, bisect_iters)
        throughput = 0.0
        if metrics:
            throughput = metrics.get("tokens_per_sec", rate * 768)
        return throughput, config, metrics

    def run_hierarchical_sweep(
        self,
        configs_by_topo: dict[tuple[int, int], list[dict]],
        seed: int = 42,
        max_workers: int = MAX_PARALLEL_WORKERS,
    ) -> int:
        """Three-stage hierarchical halving across all topologies (parallelized).

        Evaluates ALL configs in the pool using parallel blis invocations.
        Stage 1: Screen with cheap probes (100 requests), keep top 25%
        Stage 2: Re-evaluate survivors (500 requests), keep top 50%
        Stage 3: Final screening (1000 requests), then full 10k-request confirmation

        Guarantees at least 1 survivor per topology at each stage to maintain
        Pareto coverage across cost levels.
        """
        total_written = 0

        candidates = {k: list(v) for k, v in configs_by_topo.items()}
        total_candidates = sum(len(v) for v in candidates.values())
        logger.info(
            "Hierarchical sweep: %d candidates across %d topologies (workers=%d)",
            total_candidates, len(candidates), max_workers,
        )

        for stage_idx, stage in enumerate(HALVING_STAGES):
            nr = stage["num_requests"]
            bisect = stage["bisect_iters"]
            keep = stage["keep_ratio"]
            is_final = (stage_idx == len(HALVING_STAGES) - 1)

            stage_total = sum(len(v) for v in candidates.values())
            logger.info(
                "Stage %d: evaluating %d configs (num_requests=%d, bisect=%d, workers=%d)",
                stage_idx + 1, stage_total, nr, bisect, max_workers,
            )

            scored: dict[tuple[int, int], list[tuple[float, dict]]] = {
                k: [] for k in candidates
            }

            # Parallel evaluation of all configs in this stage
            all_tasks = []
            for topo_key, configs in candidates.items():
                for config in configs:
                    all_tasks.append((topo_key, config))

            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._eval_rate_worker, config, nr, bisect,
                    ): topo_key
                    for topo_key, config in all_tasks
                }
                for future in as_completed(futures):
                    topo_key = futures[future]
                    throughput, config, _ = future.result()
                    scored[topo_key].append((throughput, config))
                    completed += 1
                    if completed % 100 == 0:
                        logger.info(
                            "  Stage %d: %d/%d evaluated",
                            stage_idx + 1, completed, stage_total,
                        )

            logger.info("  Stage %d: %d/%d evaluated (done)", stage_idx + 1, completed, stage_total)

            if is_final:
                # Final stage: confirm top configs at full fidelity
                finalists = []
                for topo_key, results in scored.items():
                    results.sort(key=lambda x: x[0], reverse=True)
                    for throughput, config in results:
                        if throughput > 0:
                            finalists.append((topo_key, throughput, config))

                logger.info(
                    "Stage %d: %d finalists with throughput > 0, running full evaluation (workers=%d)",
                    stage_idx + 1, len(finalists), max_workers,
                )

                # Parallel full evaluation of finalists
                completed_finals = 0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self.evaluate_config, config): (topo_key, throughput)
                        for topo_key, throughput, config in finalists
                    }
                    for future in as_completed(futures):
                        topo_key, screening_tput = futures[future]
                        cr = future.result()
                        self.append_result(cr)
                        total_written += 1
                        completed_finals += 1
                        if completed_finals % 10 == 0:
                            logger.info(
                                "  Finalists: %d/%d confirmed",
                                completed_finals, len(finalists),
                            )

                logger.info("  Finalists: %d/%d confirmed (done)", completed_finals, len(finalists))
            else:
                # Halving: keep top-k per topology
                next_candidates = {}
                for topo_key, results in scored.items():
                    results.sort(key=lambda x: x[0], reverse=True)
                    n_keep = max(
                        HALVING_MIN_SURVIVORS_PER_TOPO,
                        int(len(results) * keep),
                    )
                    next_candidates[topo_key] = [cfg for _, cfg in results[:n_keep]]
                    logger.info(
                        "  TP=%d R=%d: %d -> %d survivors (best=%.0f tok/s)",
                        topo_key[0], topo_key[1], len(results), n_keep,
                        results[0][0] if results else 0,
                    )
                candidates = next_candidates

        logger.info("Hierarchical sweep: %d total results written", total_written)
        return total_written

    # ── Adaptive-hierarchical search (from search_blis.py) ────────────────

    def _hardware_flag(self) -> str:
        hw = self.workload.hardware
        if "H100" in hw:
            return "H100"
        if "A100" in hw:
            return "A100-SXM"
        if "L40" in hw:
            return "L40S"
        return "H100"

    def _build_search_args(
        self, config: dict, rate: float, num_requests: int, seed: int,
    ) -> list[str]:
        args = [
            "--model", self.workload.model,
            "--hardware", self._hardware_flag(),
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
        ]

        if config.get("routing_policy"):
            args.extend(["--routing-policy", config["routing_policy"]])
            if config.get("routing_scorers"):
                args.extend(["--routing-scorers", config["routing_scorers"]])

        if config.get("admission_policy"):
            args.extend(["--admission-policy", config["admission_policy"]])

        if self.workload.preset and self.workload.preset != "distribution":
            args.extend(["--workload", self.workload.preset])

        if self.fitness_weights:
            args.extend(["--fitness-weights", self.fitness_weights])

        if self.defaults_filepath:
            args.extend(["--defaults-filepath", self.defaults_filepath])

        if self.hardware_config:
            args.extend(["--hardware-config", self.hardware_config])

        return args

    @staticmethod
    def _parse_cluster_metrics(stdout: str) -> dict:
        blocks = re.findall(
            r"=== Simulation Metrics ===\s*(\{.*?\})", stdout, re.DOTALL,
        )
        for block_str in blocks:
            try:
                data = json.loads(block_str)
                if data.get("instance_id") == "cluster" or len(blocks) == 1:
                    return data
            except json.JSONDecodeError:
                continue
        return {}

    def _run_scored_eval(
        self, config: dict, rate: float, num_requests: int, seed: int,
    ) -> tuple[float, dict] | None:
        """Run a single BLIS eval with fitness scoring. Returns (score, metrics) or None."""
        args = self._build_search_args(config, rate, num_requests, seed)
        try:
            result = subprocess.run(
                [self.blis_binary, "run"] + args,
                capture_output=True, text=True,
                timeout=self.timeout_seconds,
                cwd=Path(self.blis_binary).parent.parent.parent,
            )
            if result.returncode != 0:
                logger.warning("blis search eval failed: %s", result.stderr[:200])
                return None

            score_match = re.search(r"Score:\s+([\d.]+)", result.stdout)
            if not score_match:
                logger.warning("No fitness score in blis output")
                return None
            score = float(score_match.group(1))

            metrics = self._parse_cluster_metrics(result.stdout)
            return score, metrics
        except subprocess.TimeoutExpired:
            logger.warning("blis search eval timed out")
            return None
        except Exception as e:
            logger.warning("blis search eval error: %s", e)
            return None

    @staticmethod
    def _make_bracket_profiles(k: int) -> list[dict]:
        max_profile = {
            "max_num_seqs": max(SEARCH_MAX_RUNNING),
            "max_batched_tokens": max(SEARCH_MAX_TOKENS),
            "chunked_prefill_threshold": max(SEARCH_PREFILL_THRESHOLDS),
        }
        min_profile = {
            "max_num_seqs": min(SEARCH_MAX_RUNNING),
            "max_batched_tokens": min(SEARCH_MAX_TOKENS),
            "chunked_prefill_threshold": min(SEARCH_PREFILL_THRESHOLDS),
        }
        return [max_profile, min_profile][:k]

    @staticmethod
    def _make_strategic_profiles(k: int) -> list[dict]:
        profiles = [
            {"max_num_seqs": 256, "max_batched_tokens": 4096, "chunked_prefill_threshold": 1024},
            {"max_num_seqs": 512, "max_batched_tokens": 8192, "chunked_prefill_threshold": 4096},
            {"max_num_seqs": 64, "max_batched_tokens": 2048, "chunked_prefill_threshold": 0},
        ]
        return profiles[:k]

    @staticmethod
    def _make_random_profiles(k: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        profiles = []
        for _ in range(k):
            profiles.append({
                "max_num_seqs": SEARCH_MAX_RUNNING[rng.integers(0, len(SEARCH_MAX_RUNNING))],
                "max_batched_tokens": SEARCH_MAX_TOKENS[rng.integers(0, len(SEARCH_MAX_TOKENS))],
                "chunked_prefill_threshold": SEARCH_PREFILL_THRESHOLDS[
                    rng.integers(0, len(SEARCH_PREFILL_THRESHOLDS))
                ],
            })
        return profiles

    @staticmethod
    def _make_phase1_config(tp: int, replicas: int, profile: dict) -> dict:
        routing = "least-loaded" if replicas > 1 else None
        return {
            "tp": tp,
            "replicas": replicas,
            "scheduler": "fcfs",
            "max_num_seqs": profile["max_num_seqs"],
            "max_batched_tokens": profile["max_batched_tokens"],
            "chunked_prefill_threshold": profile["chunked_prefill_threshold"],
            "block_size": 16,
            "routing_policy": routing,
            "routing_scorers": None,
            "admission_policy": "always-admit" if replicas > 1 else None,
            "preemption_policy": "fcfs",
        }

    def _build_config_result(
        self, config: dict, score: float, metrics: dict,
        rate: float, phase: int, profile_idx: int | None = None,
    ) -> ConfigResult:
        vllm_args = VllmArgs(
            tensor_parallel_size=config["tp"],
            pipeline_parallel_size=1,
            num_replicas=config["replicas"],
            data_parallel_size=1,
            max_num_seqs=config["max_num_seqs"],
            max_num_batched_tokens=config["max_batched_tokens"],
            enable_chunked_prefill=config["chunked_prefill_threshold"] > 0,
            block_size=config["block_size"],
        )

        routing = None
        if config.get("routing_policy"):
            if config["routing_policy"] == "least-loaded":
                routing = RoutingConfig(
                    strategy="least-loaded", scorers="queue-depth:1", picker="max-score",
                )
            elif config["routing_policy"] == "round-robin":
                routing = RoutingConfig(strategy="round-robin")

        tool_cfg = ToolConfig(
            scheduler=config["scheduler"],
            admission_policy=config.get("admission_policy"),
            preemption_policy=config["preemption_policy"],
        )

        config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

        throughput_tok_s = metrics.get("tokens_per_sec", 0.0)
        throughput_qps = metrics.get("responses_per_sec", rate)
        ttft_mean = metrics.get("ttft_mean_ms", 0.0)
        cost = config["tp"] * config["replicas"] * 3.20

        results_obj = None
        if throughput_tok_s > 0 or score > 0:
            results_obj = Results(
                max_throughput_tok_s=throughput_tok_s,
                max_throughput_qps=throughput_qps,
                ttft_mean_ms=ttft_mean,
                ttft_p50_ms=metrics.get("ttft_p50_ms"),
                ttft_p99_ms=metrics.get("ttft_p99_ms"),
                tpot_mean_ms=metrics.get("itl_mean_ms"),
                meets_slo=ttft_mean <= self.workload.slo_ttft_mean_ms,
                cost_per_hour=cost,
                cost_per_1k_tokens=(cost / throughput_tok_s * 1000 / 3600)
                if throughput_tok_s > 0 else None,
            )

        return ConfigResult(
            tool=self.tool_name,
            workload=self.workload,
            vllm_args=vllm_args,
            routing_config=routing,
            tool_config=tool_cfg,
            results=results_obj,
            metadata=Metadata(
                status="ok" if results_obj else "unconverged",
                config_hash=config_hash,
                num_rate_probes=1,
                fitness_score=score,
                search_phase=phase,
                search_profile_idx=profile_idx,
            ),
        )

    def run_search(self, search_seed: int = 42) -> list[ConfigResult]:
        """Execute adaptive-hierarchical search. Returns all evaluated ConfigResults."""
        start = time.monotonic()
        rate = getattr(self, "_search_rate", None) or 200.0
        num_requests = min(1000, self.workload.num_requests)

        tp_configs = (
            SEARCH_TP_CONFIGS_LEAN if self.tp_candidates == "lean"
            else SEARCH_TP_CONFIGS_ALL
        )

        if self.phase1_profiles == "bracket":
            profiles = self._make_bracket_profiles(self.phase1_k)
        elif self.phase1_profiles == "random":
            profiles = self._make_random_profiles(self.phase1_k, search_seed)
        elif self.phase1_profiles == "strategic":
            profiles = self._make_strategic_profiles(self.phase1_k)
        else:
            profiles = self._make_bracket_profiles(self.phase1_k)

        results: list[ConfigResult] = []
        phase1_tp_scores: dict[int, list[float]] = {tp: [] for tp, _ in tp_configs}
        best_score = -1.0
        eval_count = 0

        logger.info(
            "Search: strategy=adaptive-hierarchical budget=%d tp_candidates=%s "
            "phase1_k=%d profiles=%s skip_phase2=%s rate=%.1f num_requests=%d",
            self.search_budget, self.tp_candidates, self.phase1_k,
            self.phase1_profiles, self.skip_phase2, rate, num_requests,
        )

        # Phase 1: Evaluate each TP level with K bracket profiles
        for profile_idx, profile in enumerate(profiles):
            for tp, replicas in tp_configs:
                if eval_count >= self.search_budget:
                    break

                config = self._make_phase1_config(tp, replicas, profile)
                out = self._run_scored_eval(config, rate, num_requests, self.workload.seed)

                if out is None:
                    score, metrics = 0.0, {}
                else:
                    score, metrics = out

                phase1_tp_scores[tp].append(score)

                cr = self._build_config_result(
                    config, score, metrics, rate, phase=1, profile_idx=profile_idx,
                )
                results.append(cr)
                self.append_result(cr)
                eval_count += 1

                if score > best_score:
                    best_score = score

        # Pick Phase 1 winner
        tp_max = {tp: max(scores) if scores else 0.0
                  for tp, scores in phase1_tp_scores.items()}
        if all(s <= 0 for s in tp_max.values()):
            winning_tp, winning_replicas = tp_configs[-1]
        else:
            winning_tp = max(tp_max, key=lambda t: tp_max[t])
            winning_replicas = next(r for tp, r in tp_configs if tp == winning_tp)

        logger.info(
            "Phase 1 done: winner=TP%d/%dinst (score=%.4f). Scores: %s",
            winning_tp, winning_replicas, tp_max[winning_tp],
            " ".join(f"TP{tp}={s:.4f}" for tp, s in sorted(tp_max.items())),
        )

        # Phase 2: TPE optimization in winner subspace
        remaining = self.search_budget - eval_count
        if not self.skip_phase2 and remaining > 0:
            phase2_results = self._run_phase2_tpe(
                winning_tp, winning_replicas, remaining, rate, num_requests, search_seed,
            )
            for cr in phase2_results:
                results.append(cr)
                self.append_result(cr)

        elapsed = time.monotonic() - start
        logger.info(
            "Search complete: %d evals, best_fitness=%.4f, wall=%.1fs",
            len(results), best_score, elapsed,
        )
        return results

    def _run_phase2_tpe(
        self, tp: int, replicas: int, budget: int,
        rate: float, num_requests: int, search_seed: int,
    ) -> list[ConfigResult]:
        results: list[ConfigResult] = []

        def objective(trial: optuna.Trial) -> float:
            config = {
                "tp": tp,
                "replicas": replicas,
                "scheduler": trial.suggest_categorical("scheduler", SEARCH_SCHEDULERS),
                "max_num_seqs": trial.suggest_categorical("max_num_seqs", SEARCH_MAX_RUNNING),
                "max_batched_tokens": trial.suggest_categorical("max_batched_tokens", SEARCH_MAX_TOKENS),
                "chunked_prefill_threshold": trial.suggest_categorical(
                    "chunked_prefill_threshold", SEARCH_PREFILL_THRESHOLDS,
                ),
                "block_size": trial.suggest_categorical("block_size", SEARCH_BLOCK_SIZES),
                "routing_policy": (
                    trial.suggest_categorical("routing", SEARCH_ROUTING)
                    if replicas > 1 else None
                ),
                "routing_scorers": None,
                "admission_policy": (
                    trial.suggest_categorical("admission", SEARCH_ADMISSION)
                    if replicas > 1 else None
                ),
                "preemption_policy": trial.suggest_categorical("preemption", SEARCH_PREEMPTION),
            }

            out = self._run_scored_eval(config, rate, num_requests, self.workload.seed)
            if out is None:
                return 0.0

            score, metrics = out
            cr = self._build_config_result(config, score, metrics, rate, phase=2)
            results.append(cr)
            return score

        sampler = optuna.samplers.TPESampler(seed=search_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=budget, show_progress_bar=False)

        logger.info(
            "Phase 2 TPE: %d trials, best=%.4f",
            len(results), study.best_value if study.best_trial else 0.0,
        )
        return results

    def run_tpe_search(self, search_seed: int = 42) -> list[ConfigResult]:
        """TPE search over full config space including varying GPU counts.

        Mirrors search_blis.py's run_tpe_search: samples from VALID_TP_INSTANCES
        (all (tp, replicas) pairs where tp * replicas <= max_gpus), so the search
        explores configs at different cost levels.
        """
        start = time.monotonic()
        rate = getattr(self, "_search_rate", None) or 200.0
        num_requests = min(1000, self.workload.num_requests)
        results: list[ConfigResult] = []

        logger.info(
            "TPE search: budget=%d rate=%.1f num_requests=%d",
            self.search_budget, rate, num_requests,
        )

        def objective(trial: optuna.Trial) -> float:
            tp_ni_idx = trial.suggest_int(
                "tp_ni_idx", 0, len(SEARCH_VALID_TP_INSTANCES) - 1,
            )
            tp, replicas = SEARCH_VALID_TP_INSTANCES[tp_ni_idx]

            config = {
                "tp": tp,
                "replicas": replicas,
                "scheduler": trial.suggest_categorical("scheduler", SEARCH_SCHEDULERS),
                "max_num_seqs": trial.suggest_categorical("max_num_seqs", SEARCH_MAX_RUNNING),
                "max_batched_tokens": trial.suggest_categorical(
                    "max_batched_tokens", SEARCH_MAX_TOKENS,
                ),
                "chunked_prefill_threshold": trial.suggest_categorical(
                    "chunked_prefill_threshold", SEARCH_PREFILL_THRESHOLDS,
                ),
                "block_size": trial.suggest_categorical("block_size", SEARCH_BLOCK_SIZES),
                "routing_policy": (
                    trial.suggest_categorical("routing", SEARCH_ROUTING)
                    if replicas > 1 else None
                ),
                "routing_scorers": None,
                "admission_policy": (
                    trial.suggest_categorical("admission", SEARCH_ADMISSION)
                    if replicas > 1 else None
                ),
                "preemption_policy": trial.suggest_categorical("preemption", SEARCH_PREEMPTION),
            }

            out = self._run_scored_eval(config, rate, num_requests, self.workload.seed)
            if out is None:
                return 0.0

            score, metrics = out
            cr = self._build_config_result(config, score, metrics, rate, phase=0)
            results.append(cr)
            self.append_result(cr)
            return score

        sampler = optuna.samplers.TPESampler(seed=search_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.search_budget, show_progress_bar=False)

        elapsed = time.monotonic() - start
        logger.info(
            "TPE search complete: %d evals, best=%.4f, wall=%.1fs",
            len(results),
            study.best_value if study.best_trial else 0.0,
            elapsed,
        )
        return results
