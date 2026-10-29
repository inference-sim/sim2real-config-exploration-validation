import json
import logging
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import yaml

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, ToolConfig,
    Results, Metadata, WorkloadInfo, compute_config_hash,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

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

# NSGA-II multi-objective search parameter space (from search.py)
NSGA2_PARAM_SPACE = {
    'tp': [1, 2, 4, 8],
    'scheduler': ['fcfs', 'sjf'],
    'max_num_running_reqs': [32, 64, 128, 256, 512],
    'total_kv_blocks': [2000, 3000, 4000, 5000, 7500, 10000],
    'block_size_in_tokens': [16, 32],
}
NSGA2_DEFAULT_POP_SIZE = 40
NSGA2_DEFAULT_N_GENS = 4
NSGA2_DEFAULT_WORKERS = 8
NSGA2_COMMON_FLAGS = [
    "--latency-model", "trained-physics",
    "--num-requests", "200",
    "--prefix-tokens", "512",
    "--seed", "42",
    "--max-num-scheduled-tokens", "4096",
    "--long-prefill-token-threshold", "0",
    "--admission-policy", "always-admit",
    "--preemption-policy", "fcfs",
    "--routing-policy", "round-robin",
    "--gpu-memory-utilization", "0.9",
]


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


# ──────────────────────────────────────────────────────────
# NSGA-II algorithm (from inference-sim/search.py)
# ──────────────────────────────────────────────────────────

def _nsga2_get_max_instances(tp: int) -> int:
    return 8 // tp


def _nsga2_config_key(config: dict) -> tuple:
    return (config['tp'], config['num_instances'], config['scheduler'],
            config['max_num_running_reqs'], config['total_kv_blocks'],
            config['block_size_in_tokens'])


def _nsga2_normalize_config(config: dict) -> dict:
    c = dict(config)
    c['num_instances'] = max(1, min(c['num_instances'], 8 // c['tp']))
    return c


def _nsga2_random_config(rng: random.Random, param_space: dict) -> dict:
    tp = rng.choice(param_space['tp'])
    return {
        'tp': tp,
        'num_instances': rng.randint(1, 8 // tp),
        'scheduler': rng.choice(param_space['scheduler']),
        'max_num_running_reqs': rng.choice(param_space['max_num_running_reqs']),
        'total_kv_blocks': rng.choice(param_space['total_kv_blocks']),
        'block_size_in_tokens': rng.choice(param_space['block_size_in_tokens']),
    }


def _nsga2_pareto_dominates(a: list, b: list) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _nsga2_get_nondominated(points: list[list]) -> list[list]:
    nd = []
    for p in points:
        if not any(_nsga2_pareto_dominates(q, p) for q in nd):
            nd = [q for q in nd if not _nsga2_pareto_dominates(p, q)]
            nd.append(p)
    return nd


def _nsga2_fast_nondominated_sort(objectives: list[list]) -> list[list[int]]:
    n = len(objectives)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _nsga2_pareto_dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _nsga2_pareto_dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _nsga2_crowding_distance(objectives: list[list]) -> list[float]:
    n = len(objectives)
    if n <= 2:
        return [float('inf')] * n
    d = len(objectives[0])
    distances = [0.0] * n
    for m in range(d):
        sorted_idx = sorted(range(n), key=lambda i: objectives[i][m])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        mn = objectives[sorted_idx[0]][m]
        mx = objectives[sorted_idx[-1]][m]
        if mx == mn:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                objectives[sorted_idx[k + 1]][m] - objectives[sorted_idx[k - 1]][m]
            ) / (mx - mn)
    return distances


def _nsga2_tournament_select(
    objectives: list[list], fronts: list[list[int]], rng: random.Random,
) -> int:
    ranks = {}
    for rank, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank
    cd_map = {}
    for front in fronts:
        fobj = [objectives[i] for i in front]
        fcd = _nsga2_crowding_distance(fobj)
        for k, i in enumerate(front):
            cd_map[i] = fcd[k]

    n = len(objectives)
    a, b = rng.randrange(n), rng.randrange(n)
    ra, rb = ranks.get(a, 999), ranks.get(b, 999)
    if ra < rb:
        return a
    if rb < ra:
        return b
    return a if cd_map.get(a, 0) >= cd_map.get(b, 0) else b


def _nsga2_crossover(
    p1: dict, p2: dict, rng: random.Random,
) -> tuple[dict, dict]:
    c1, c2 = dict(p1), dict(p2)
    params = ['tp', 'num_instances', 'scheduler', 'max_num_running_reqs',
              'total_kv_blocks', 'block_size_in_tokens']
    for param in params:
        if rng.random() < 0.5:
            c1[param], c2[param] = c2[param], c1[param]
    return _nsga2_normalize_config(c1), _nsga2_normalize_config(c2)


def _nsga2_mutate(
    config: dict, rng: random.Random, param_space: dict, mutation_rate: float = 0.15,
) -> dict:
    c = dict(config)
    if rng.random() < mutation_rate:
        c['tp'] = rng.choice(param_space['tp'])
    if rng.random() < mutation_rate:
        c['num_instances'] = rng.randint(1, 8 // c['tp'])
    if rng.random() < mutation_rate:
        c['scheduler'] = rng.choice(param_space['scheduler'])
    if rng.random() < mutation_rate:
        c['max_num_running_reqs'] = rng.choice(param_space['max_num_running_reqs'])
    if rng.random() < mutation_rate:
        c['total_kv_blocks'] = rng.choice(param_space['total_kv_blocks'])
    if rng.random() < mutation_rate:
        c['block_size_in_tokens'] = rng.choice(param_space['block_size_in_tokens'])
    return _nsga2_normalize_config(c)


def run_nsga2(
    eval_fn,
    param_space: dict | None = None,
    pop_size: int = NSGA2_DEFAULT_POP_SIZE,
    n_gens: int = NSGA2_DEFAULT_N_GENS,
    seed: int = 42,
) -> tuple[list[tuple[dict, list]], dict]:
    """Run NSGA-II search. Returns (pareto_configs, all_evaluated).

    Args:
        eval_fn: callable(config) -> objectives list or None
        param_space: dict of parameter -> value lists
        pop_size: population size per generation
        n_gens: number of generations after initial population
        seed: random seed

    Returns:
        pareto_configs: list of (config, objectives) on the Pareto front
        all_evaluated: dict of config_key -> (config, objectives)
    """
    if param_space is None:
        param_space = NSGA2_PARAM_SPACE
    rng = random.Random(seed)
    evaluated = {}

    def get_unique_config():
        for _ in range(1000):
            cfg = _nsga2_random_config(rng, param_space)
            k = _nsga2_config_key(cfg)
            if k not in evaluated:
                return cfg
        return None

    population = []
    pop_objectives = []
    for _ in range(pop_size):
        cfg = get_unique_config()
        if cfg is None:
            break
        obj = eval_fn(cfg)
        if obj is None:
            continue
        evaluated[_nsga2_config_key(cfg)] = (cfg, obj)
        population.append(cfg)
        pop_objectives.append(obj)

    for gen in range(n_gens):
        fronts = _nsga2_fast_nondominated_sort(pop_objectives)
        offspring = []
        offspring_obj = []

        attempts = 0
        while len(offspring) < pop_size and attempts < pop_size * 20:
            attempts += 1
            p1_idx = _nsga2_tournament_select(pop_objectives, fronts, rng)
            p2_idx = _nsga2_tournament_select(pop_objectives, fronts, rng)
            child1, child2 = _nsga2_crossover(population[p1_idx], population[p2_idx], rng)
            for child in [child1, child2]:
                child = _nsga2_mutate(child, rng, param_space)
                child = _nsga2_normalize_config(child)
                k = _nsga2_config_key(child)
                if k not in evaluated:
                    obj = eval_fn(child)
                    if obj is not None:
                        evaluated[k] = (child, obj)
                        offspring.append(child)
                        offspring_obj.append(obj)
                if len(offspring) >= pop_size:
                    break

        while len(offspring) < pop_size:
            cfg = get_unique_config()
            if cfg is None:
                break
            obj = eval_fn(cfg)
            if obj is None:
                continue
            evaluated[_nsga2_config_key(cfg)] = (cfg, obj)
            offspring.append(cfg)
            offspring_obj.append(obj)

        combined = population + offspring
        combined_obj = pop_objectives + offspring_obj
        combined_fronts = _nsga2_fast_nondominated_sort(combined_obj)

        new_pop = []
        new_pop_obj = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for i in front:
                    new_pop.append(combined[i])
                    new_pop_obj.append(combined_obj[i])
            else:
                remaining = pop_size - len(new_pop)
                fobj = [combined_obj[i] for i in front]
                fcd = _nsga2_crowding_distance(fobj)
                sorted_front = sorted(range(len(front)), key=lambda k: -fcd[k])
                for k in sorted_front[:remaining]:
                    new_pop.append(combined[front[k]])
                    new_pop_obj.append(combined_obj[front[k]])
                break

        population = new_pop
        pop_objectives = new_pop_obj

        logger.info("  NSGA-II gen %d/%d: %d evaluated", gen + 1, n_gens, len(evaluated))

    all_objs = [v[1] for v in evaluated.values()]
    pareto_objs = _nsga2_get_nondominated(all_objs)
    pareto_set = set(tuple(obj) for obj in pareto_objs)

    pareto_configs = []
    for cfg, obj in evaluated.values():
        if tuple(obj) in pareto_set:
            pareto_configs.append((cfg, obj))

    return pareto_configs, evaluated


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
        search_script: str | None = None,
        model_config_folder: str | None = None,
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
        self.search_script = search_script
        self.model_config_folder = model_config_folder

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
                workload_spec=self.workload_spec,
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
            if metrics.get("ttft_mean_ms", 0) > self.workload.slo_ttft_mean_ms:
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
                if metrics and metrics.get("ttft_mean_ms", 0) <= self.workload.slo_ttft_mean_ms:
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
                    if confirm and confirm.get("ttft_mean_ms", 0) <= self.workload.slo_ttft_mean_ms:
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
                meets_slo=best_metrics.get("ttft_mean_ms", 999) <= self.workload.slo_ttft_mean_ms,
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
            if metrics.get("ttft_mean_ms", 0) > self.workload.slo_ttft_mean_ms:
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
            if metrics and metrics.get("ttft_mean_ms", 0) <= self.workload.slo_ttft_mean_ms:
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

        if self.workload_spec:
            args.extend(["--workload-spec", self.workload_spec])
        elif self.workload.preset and self.workload.preset != "distribution":
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

    # ── Native search via inference-sim/search.py ────────────────────────────

    def _generate_fallback_workload_spec(self) -> str:
        """Generate a temporary workload spec YAML from WorkloadInfo ISL/OSL.

        Used when no explicit workload_spec file is provided, to ensure
        BLIS evaluates with the correct sequence length distribution
        rather than its internal defaults.
        """
        spec = {
            "api": {"streaming": True, "type": "completion"},
            "data": {
                "type": "fixed_length",
                "input_tokens": self.workload.isl_mean,
                "output_tokens": self.workload.osl_mean,
            },
            "load": {
                "type": "constant",
                "rate": 10,
            },
        }
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="blis_workload_",
        )
        yaml.dump(spec, f, default_flow_style=False)
        f.close()
        return f.name

    def run_native_search(
        self,
        search_seed: int = 42,
        pop_size: int = 40,
        n_gens: int = 4,
        workers: int = 8,
        objectives: list[str] | None = None,
        constraints: list[str] | None = None,
        params: dict | None = None,
        scorer_space: dict | None = None,
        fixed_rate: float | None = None,
        max_model_len: int | None = None,
    ) -> list[ConfigResult]:
        """Run NSGA-II search via inference-sim's native search.py.

        Generates a spec YAML, shells out to search.py, and translates
        the output into ConfigResult objects.
        """
        start = time.monotonic()
        rate = fixed_rate or getattr(self, "_search_rate", None) or 200.0

        if objectives is None:
            objectives = ["throughput", "ttft_p99", "gpu_count"]
        if constraints is None:
            constraints = []

        default_params = {
            "tp": [2, 4, 8],
            "num_instances": [1, 2, 4],
            "scheduler": ["fcfs", "sjf"],
            "max_num_running_reqs": [32, 64, 128, 256, 512],
            "total_kv_blocks": [2000, 3000, 4000, 5000, 7500, 10000],
            "block_size_in_tokens": [16, 32],
        }
        search_params = params if params is not None else default_params

        generated_workload_spec = None
        workload_spec_path = self.workload_spec
        if not workload_spec_path:
            generated_workload_spec = self._generate_fallback_workload_spec()
            workload_spec_path = generated_workload_spec
            logger.warning(
                "No workload_spec provided; generated fallback with ISL=%d OSL=%d at %s",
                self.workload.isl_mean, self.workload.osl_mean, generated_workload_spec,
            )

        fixed = {
            "model": self.workload.model,
            "hardware": self._hardware_flag(),
            "rate": rate,
            "num_requests": min(200, self.workload.num_requests),
            "seed": self.workload.seed,
            "latency_model": "trained-physics",
            "gpu_memory_utilization": 0.9,
            "workload_spec": workload_spec_path,
        }
        if max_model_len is not None:
            fixed["max_model_len"] = max_model_len
        if self.defaults_filepath:
            fixed["defaults_filepath"] = self.defaults_filepath
        if self.hardware_config:
            fixed["hardware_config"] = self.hardware_config
        if self.model_config_folder:
            fixed["model_config_folder"] = self.model_config_folder

        spec = {
            "fixed": fixed,
            "search_space": search_params,
            "objectives": objectives,
            "algorithm": {
                "pop_size": pop_size,
                "n_gens": n_gens,
                "seed": self.workload.seed,
                "workers": workers,
            },
        }
        if constraints:
            spec["constraints"] = constraints
        if scorer_space:
            ss = dict(scorer_space)
            min_scorers = ss.pop("min_scorers", 1)
            spec["scorer_space"] = {
                "scorers": ss,
                "min_scorers": min_scorers,
            }

        if self.search_script:
            search_py = self.search_script
        else:
            search_py = str(Path(self.blis_binary).parent / "search.py")

        if not Path(search_py).exists():
            logger.error(
                "search.py not found at %s. Set search_script path or update submodule.",
                search_py,
            )
            if generated_workload_spec:
                try:
                    os.remove(generated_workload_spec)
                except OSError:
                    pass
            return []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="blis_spec_",
        ) as spec_file:
            yaml.dump(spec, spec_file, default_flow_style=False)
            spec_path = spec_file.name

        output_path = spec_path.replace(".yaml", "_results.json")

        logger.info(
            "Native NSGA-II search: model=%s, rate=%.1f, pop=%d, gens=%d, "
            "workers=%d, objectives=%s, spec=%s",
            self.workload.model, rate, pop_size, n_gens, workers,
            objectives, spec_path,
        )

        try:
            env = os.environ.copy()
            env["BLIS_BIN"] = self.blis_binary

            result = subprocess.run(
                ["python3", search_py, "--spec", spec_path, "-o", output_path],
                capture_output=True, text=True,
                timeout=3600,
                env=env,
            )

            if result.returncode != 0:
                logger.error("search.py failed (rc=%d): %s", result.returncode, result.stderr[:500])
                return []

            with open(output_path) as f:
                search_output = json.load(f)

        finally:
            cleanup_paths = [spec_path, output_path]
            if generated_workload_spec:
                cleanup_paths.append(generated_workload_spec)
            for p in cleanup_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        pareto_keys = set()
        for entry in search_output.get("pareto_configs", []):
            cfg = entry["config"]
            key = (cfg.get("tp"), cfg.get("num_instances"), cfg.get("scheduler"),
                   cfg.get("max_num_running_reqs"), cfg.get("total_kv_blocks"),
                   cfg.get("block_size_in_tokens"))
            pareto_keys.add(key)

        results: list[ConfigResult] = []
        for entry in search_output.get("all_configs", []):
            cfg = entry["config"]
            obj = entry.get("objectives", {})
            feasible = entry.get("feasible", True)

            key = (cfg.get("tp"), cfg.get("num_instances"), cfg.get("scheduler"),
                   cfg.get("max_num_running_reqs"), cfg.get("total_kv_blocks"),
                   cfg.get("block_size_in_tokens"))
            is_pareto = key in pareto_keys

            cr = self._translate_native_result(cfg, obj, feasible, is_pareto)
            results.append(cr)
            self.append_result(cr)

        elapsed = time.monotonic() - start
        pareto_count = sum(1 for r in results if r.metadata.pareto_optimal)
        logger.info(
            "Native NSGA-II complete: %d evaluated, %d Pareto-optimal, wall=%.1fs",
            len(results), pareto_count, elapsed,
        )
        return results

    def _translate_native_result(
        self, cfg: dict, objectives: dict, feasible: bool, pareto_optimal: bool,
    ) -> ConfigResult:
        """Translate a search.py result entry into a ConfigResult."""
        tp = cfg.get("tp", 1)
        num_instances = cfg.get("num_instances", 1)

        max_scheduled = cfg.get("max_num_scheduled_tokens", 4096)
        prefill_threshold = cfg.get("long_prefill_token_threshold", 0)

        vllm_args = VllmArgs(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            num_replicas=num_instances,
            data_parallel_size=1,
            max_num_seqs=cfg.get("max_num_running_reqs", 256),
            max_num_batched_tokens=max_scheduled,
            enable_chunked_prefill=prefill_threshold > 0,
            block_size=cfg.get("block_size_in_tokens", 16),
            gpu_memory_utilization=0.9,
        )

        routing_policy = cfg.get("routing_policy", "round-robin")
        routing_scorers = cfg.get("routing_scorers")
        if routing_policy == "weighted" and routing_scorers:
            routing = RoutingConfig(
                strategy="weighted-scoring",
                scorers=routing_scorers,
                picker="max-score",
            )
        elif routing_policy == "least-loaded":
            routing = RoutingConfig(
                strategy="least-loaded",
                scorers="queue-depth:1",
                picker="max-score",
            )
        else:
            routing = RoutingConfig(strategy="round-robin")

        tool_cfg = ToolConfig(
            scheduler=cfg.get("scheduler", "fcfs"),
            admission_policy=cfg.get("admission_policy", "always-admit"),
            preemption_policy=cfg.get("preemption_policy", "fcfs"),
        )

        config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

        throughput_rps = objectives.get("throughput", 0.0)
        ttft_p99 = objectives.get("ttft_p99", 0.0)
        gpu_count = objectives.get("gpu_count", float(tp * num_instances))
        cost = gpu_count * 3.20

        results_obj = None
        if feasible and throughput_rps > 0:
            results_obj = Results(
                max_throughput_tok_s=throughput_rps * 768,
                max_throughput_qps=throughput_rps,
                ttft_mean_ms=objectives.get("ttft_mean", 0.0),
                ttft_p99_ms=ttft_p99,
                tpot_mean_ms=objectives.get("itl_mean"),
                meets_slo=True,
                cost_per_hour=cost,
                cost_per_1k_tokens=(cost / (throughput_rps * 768) * 1000 / 3600)
                if throughput_rps > 0 else None,
            )

        return ConfigResult(
            tool=self.tool_name,
            workload=self.workload,
            vllm_args=vllm_args,
            routing_config=routing,
            tool_config=tool_cfg,
            results=results_obj,
            metadata=Metadata(
                status="ok" if results_obj else "infeasible",
                config_hash=config_hash,
                num_rate_probes=1,
                nsga2_objectives=objectives,
                pareto_optimal=pareto_optimal,
            ),
        )

    # ── NSGA-II multi-objective search (legacy reimplementation) ──────────────

    def _run_nsga2_eval(
        self, config: dict, rate: float, seed: int,
    ) -> list[float] | None:
        """Run a single BLIS evaluation for NSGA-II. Returns [neg_rps, ttft_p99, gpu_count, kv_blocks] or None."""
        tmpdir = os.environ.get('TMPDIR', '/tmp')
        tid = threading.get_ident() % 100000
        metrics_path = os.path.join(
            tmpdir, f"blis_nsga2_{os.getpid()}_{tid}.json",
        )

        cmd = [self.blis_binary, "run"] + list(NSGA2_COMMON_FLAGS) + [
            "--hardware", self._hardware_flag(),
            "--rate", str(rate),
            "--model", self.workload.model,
            "--tp", str(config['tp']),
            "--num-instances", str(config['num_instances']),
            "--scheduler", config['scheduler'],
            "--max-num-running-reqs", str(config['max_num_running_reqs']),
            "--total-kv-blocks", str(config['total_kv_blocks']),
            "--block-size-in-tokens", str(config['block_size_in_tokens']),
            "--metrics-path", metrics_path,
        ]

        if self.defaults_filepath:
            cmd.extend(["--defaults-filepath", self.defaults_filepath])
        if self.hardware_config:
            cmd.extend(["--hardware-config", self.hardware_config])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                logger.debug("NSGA-II eval failed: %s", result.stderr[:200])
                return None

            with open(metrics_path) as f:
                metrics = json.load(f)
            try:
                os.remove(metrics_path)
            except Exception:
                pass

            rps = metrics['responses_per_sec']
            ttft_p99 = metrics['ttft_p99_ms']
            gpu_count = float(config['tp'] * config['num_instances'])
            kv_blocks = float(config['total_kv_blocks'])
            config['_metrics'] = {
                'tokens_per_sec': metrics.get('tokens_per_sec', 0.0),
                'ttft_mean_ms': metrics.get('ttft_mean_ms', 0.0),
                'ttft_p99_ms': ttft_p99,
                'responses_per_sec': rps,
                'itl_mean_ms': metrics.get('itl_mean_ms'),
            }
            return [-rps, ttft_p99, gpu_count, kv_blocks]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.debug("NSGA-II eval error: %s", e)
            return None

    def _nsga2_config_to_result(
        self, config: dict, objectives: list[float], pareto_optimal: bool,
    ) -> ConfigResult:
        """Convert an NSGA-II config + objectives into a ConfigResult."""
        cached = config.get('_metrics', {})

        vllm_args = VllmArgs(
            tensor_parallel_size=config['tp'],
            pipeline_parallel_size=1,
            num_replicas=config['num_instances'],
            data_parallel_size=1,
            max_num_seqs=config['max_num_running_reqs'],
            max_num_batched_tokens=4096,
            enable_chunked_prefill=False,
            block_size=config['block_size_in_tokens'],
            gpu_memory_utilization=0.9,
        )

        routing = RoutingConfig(strategy="round-robin")

        tool_cfg = ToolConfig(
            scheduler=config['scheduler'],
            admission_policy="always-admit",
            preemption_policy="fcfs",
        )

        config_hash = compute_config_hash(vllm_args, routing, tool_cfg)

        throughput_rps = -objectives[0]
        throughput_tok_s = cached.get('tokens_per_sec', 0.0)
        ttft_mean = cached.get('ttft_mean_ms', 0.0)
        ttft_p99 = objectives[1]
        gpu_count = objectives[2]
        cost = gpu_count * 3.20

        results_obj = Results(
            max_throughput_tok_s=throughput_tok_s,
            max_throughput_qps=throughput_rps,
            ttft_mean_ms=ttft_mean,
            ttft_p99_ms=ttft_p99,
            tpot_mean_ms=cached.get('itl_mean_ms'),
            meets_slo=True,
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
                status="ok",
                config_hash=config_hash,
                num_rate_probes=1,
                nsga2_objectives={
                    "throughput_rps": throughput_rps,
                    "ttft_p99_ms": ttft_p99,
                    "gpu_count": gpu_count,
                    "total_kv_blocks": objectives[3],
                },
                pareto_optimal=pareto_optimal,
            ),
        )

    def run_nsga2_search(
        self,
        search_seed: int = 42,
        pop_size: int | None = None,
        n_gens: int | None = None,
        workers: int | None = None,
        param_space: dict | None = None,
    ) -> list[ConfigResult]:
        """Run NSGA-II multi-objective search over the BLIS config space.

        Returns all evaluated ConfigResults with Pareto membership marked.
        """
        start = time.monotonic()
        rate = getattr(self, "_search_rate", None) or 200.0
        pop = pop_size or getattr(self, "_nsga2_pop_size", NSGA2_DEFAULT_POP_SIZE)
        gens = n_gens or getattr(self, "_nsga2_n_gens", NSGA2_DEFAULT_N_GENS)
        n_workers = workers or getattr(self, "_nsga2_workers", NSGA2_DEFAULT_WORKERS)

        space = param_space if param_space is not None else NSGA2_PARAM_SPACE

        logger.info(
            "NSGA-II search: model=%s, hardware=%s, rate=%.1f, pop=%d, gens=%d, workers=%d, tp=%s",
            self.workload.model, self._hardware_flag(), rate, pop, gens, n_workers, space['tp'],
        )

        total_budget = pop * (gens + 1)
        logger.info("Max evaluations: %d", total_budget)

        lock = threading.Lock()
        eval_counter = [0]

        def eval_fn(config):
            with lock:
                eval_counter[0] += 1
            return self._run_nsga2_eval(config, rate, self.workload.seed)

        pareto_configs, all_evaluated = run_nsga2(
            eval_fn,
            param_space=space,
            pop_size=pop,
            n_gens=gens,
            seed=search_seed,
        )

        pareto_keys = set(_nsga2_config_key(cfg) for cfg, _ in pareto_configs)

        results: list[ConfigResult] = []
        for cfg, obj in all_evaluated.values():
            is_pareto = _nsga2_config_key(cfg) in pareto_keys
            cr = self._nsga2_config_to_result(cfg, obj, pareto_optimal=is_pareto)
            results.append(cr)
            self.append_result(cr)

        elapsed = time.monotonic() - start
        logger.info(
            "NSGA-II search complete: %d evaluated, %d Pareto-optimal, wall=%.1fs",
            len(all_evaluated), len(pareto_configs), elapsed,
        )
        return results
