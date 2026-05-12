# Config Exploration Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the config exploration experiment that sweeps 5 LLM serving estimators across a shared config space, normalizes their outputs to a common schema, computes Pareto fronts, and prepares top-3 configs per tool for validation on real llm-d deployments.

**Architecture:** A shared config generation library enumerates valid (TP, PP, replicas) topology triples and per-tool parameter expansions. Five independent runner scripts invoke each estimator, perform binary search or native search for max throughput at SLO, and emit JSONL in a common output schema. An analysis pipeline computes Pareto fronts, selects top-3 configs per tool, and generates paper figures. Validation scripts deploy configs on llm-d and measure sim2real drift.

**Tech Stack:** Python 3.10+, Go 1.21+ (inference-sim build), pytest, matplotlib, PyYAML, pandas, numpy

**Spec:** `docs/superpowers/specs/2026-05-07-config-exploration-experiment-design.md`

---

## File Structure

```
experiments/
  __init__.py
  config/
    __init__.py
    topology.py             # Enumerate valid (TP, PP, replicas) triples with GPU constraint
    blis_configs.py         # inference-sim parameter expansion (no PP, custom routing/admission)
    llmservingsim_configs.py # LLMServingSim parameter expansion (PP, routing, prefix caching)
    aiconfigurator_configs.py # AIConfigurator parameter expansion (PP, internal batch sweep)
    vidur_configs.py        # Vidur parameter expansion (PP, scheduler variants, routing)
    llm_optimizer_configs.py # llm-optimizer parameter expansion (PP, DP, grid search syntax)
  workloads/
    __init__.py
    generate.py             # Generate canonical trace via blis convert, convert to per-tool formats
  schema/
    __init__.py
    output.py               # Pydantic models for the common output JSON schema
  runners/
    __init__.py
    base.py                 # Base runner: checkpointing, timeout, error handling, JSONL append
    run_blis.py             # inference-sim runner with binary rate search
    run_llmservingsim.py    # LLMServingSim runner with binary rate search
    run_aiconfigurator.py   # AIConfigurator runner (cli estimate, analytical)
    run_vidur.py            # Vidur runner (native config_optimizer, TTFT post-processing)
    run_llm_optimizer.py    # llm-optimizer runner (native grid search invocation)
  analysis/
    __init__.py
    pareto.py               # Compute Pareto fronts from JSONL results
    select.py               # Select top-3 configs per tool from Pareto front
    charts.py               # Generate paper figures (Charts 1-5)
    drift.py                # Compute drift metrics from validated results
  validation/
    __init__.py
    deploy.py               # Deploy vLLM config on llm-d cluster via kubectl/helm
    benchmark.py            # Send workload to deployed instance, collect metrics
    measure.py              # Compute drift between predicted and actual metrics
  results/
    raw/                    # Per-tool JSONL output (one file per tool)
    processed/              # Merged, normalized results
    validated/              # Real deployment measurements
    figures/                # Generated charts
tests/
  __init__.py
  config/
    __init__.py
    test_topology.py        # Test topology enumeration and validity constraints
    test_blis_configs.py    # Test inference-sim config expansion
    test_llmservingsim_configs.py
    test_aiconfigurator_configs.py
    test_vidur_configs.py
    test_llm_optimizer_configs.py
  workloads/
    __init__.py
    test_generate.py        # Test workload generation and format conversion
  schema/
    __init__.py
    test_output.py          # Test output schema validation and serialization
  runners/
    __init__.py
    test_base.py            # Test checkpointing, timeout, error handling
  analysis/
    __init__.py
    test_pareto.py          # Test Pareto front computation
    test_select.py          # Test top-3 selection logic
    test_drift.py           # Test drift metric calculation
```

Each file has a single responsibility. Config generation is split per-tool because each tool has different parameter dimensions and validity rules. Runners share a base class for checkpointing and error handling. Analysis is separate from runners so it can re-run on cached results.

---

## Chunk 1: Config Generation Core

### Task 1: Topology Enumeration

**Files:**
- Create: `experiments/config/__init__.py`
- Create: `experiments/config/topology.py`
- Create: `experiments/__init__.py`
- Test: `tests/config/test_topology.py`
- Test: `tests/__init__.py`, `tests/config/__init__.py`

This is the foundation: enumerate all valid (TP, PP, replicas) triples satisfying `TP * PP * replicas <= max_gpus`. Two variants: one for PP-supporting tools (25 triples) and one for inference-sim (15 triples, PP fixed at 1). Also generates the llm-optimizer variant where replicas are replaced by `data_parallel_size`.

- [ ] **Step 1: Create test file with topology enumeration tests**

```python
# tests/__init__.py - empty
# tests/config/__init__.py - empty
# tests/config/test_topology.py

from experiments.config.topology import (
    enumerate_topologies,
    enumerate_topologies_no_pp,
    enumerate_topologies_dp,
)


def test_pp_topologies_count():
    """PP-supporting tools: 25 valid triples with max 8 GPUs."""
    triples = enumerate_topologies(max_gpus=8)
    assert len(triples) == 25


def test_pp_topologies_validity():
    """Every triple satisfies TP * PP * replicas <= max_gpus."""
    for tp, pp, replicas in enumerate_topologies(max_gpus=8):
        assert tp * pp * replicas <= 8
        assert tp in (1, 2, 4, 8)
        assert pp in (1, 2, 4)
        assert replicas >= 1


def test_pp_single_replica_count():
    """9 single-replica triples (one per valid TP*PP pair)."""
    triples = enumerate_topologies(max_gpus=8)
    single = [t for t in triples if t[2] == 1]
    assert len(single) == 9


def test_pp_multi_replica_count():
    """16 multi-replica triples."""
    triples = enumerate_topologies(max_gpus=8)
    multi = [t for t in triples if t[2] > 1]
    assert len(multi) == 16


def test_no_pp_topologies_count():
    """inference-sim (no PP): 15 valid triples with max 8 GPUs."""
    triples = enumerate_topologies_no_pp(max_gpus=8)
    assert len(triples) == 15


def test_no_pp_topologies_validity():
    for tp, replicas in triples:
        assert tp * replicas <= 8
        assert tp in (1, 2, 4, 8)
        assert replicas >= 1


def test_no_pp_single_replica_count():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    single = [t for t in triples if t[1] == 1]
    assert len(single) == 4


def test_no_pp_multi_replica_count():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    multi = [t for t in triples if t[1] > 1]
    assert len(multi) == 11


def test_dp_topologies_same_count_as_pp():
    """llm-optimizer DP triples should equal PP topology count (25)."""
    dp_triples = enumerate_topologies_dp(max_gpus=8)
    pp_triples = enumerate_topologies(max_gpus=8)
    assert len(dp_triples) == len(pp_triples)


def test_dp_topologies_validity():
    for tp, pp, dp in enumerate_topologies_dp(max_gpus=8):
        assert tp * pp * dp <= 8


def test_topology_deterministic_order():
    """Same call twice returns same order."""
    a = enumerate_topologies(max_gpus=8)
    b = enumerate_topologies(max_gpus=8)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jchen/go/src/inference-sim/sim2real-config-exploration-validation && python -m pytest tests/config/test_topology.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`

- [ ] **Step 3: Implement topology.py**

```python
# experiments/__init__.py - empty
# experiments/config/__init__.py - empty
# experiments/config/topology.py

from dataclasses import dataclass

TP_VALUES = (1, 2, 4, 8)
PP_VALUES = (1, 2, 4)


@dataclass(frozen=True)
class Topology:
    tp: int
    pp: int
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.replicas

    @property
    def cost_per_hour(self) -> float:
        return self.total_gpus * 3.20

    @property
    def is_multi_replica(self) -> bool:
        return self.replicas > 1


@dataclass(frozen=True)
class TopologyDP:
    tp: int
    pp: int
    dp: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    @property
    def cost_per_hour(self) -> float:
        return self.total_gpus * 3.20

    @property
    def is_multi_dp(self) -> bool:
        return self.dp > 1


@dataclass(frozen=True)
class TopologyNoPP:
    tp: int
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.tp * self.replicas

    @property
    def cost_per_hour(self) -> float:
        return self.total_gpus * 3.20

    @property
    def is_multi_replica(self) -> bool:
        return self.replicas > 1


def enumerate_topologies(max_gpus: int = 8) -> list[Topology]:
    result = []
    for tp in TP_VALUES:
        for pp in PP_VALUES:
            if tp * pp > max_gpus:
                continue
            max_replicas = max_gpus // (tp * pp)
            for r in range(1, max_replicas + 1):
                result.append(Topology(tp=tp, pp=pp, replicas=r))
    return result


def enumerate_topologies_no_pp(max_gpus: int = 8) -> list[TopologyNoPP]:
    result = []
    for tp in TP_VALUES:
        if tp > max_gpus:
            continue
        max_replicas = max_gpus // tp
        for r in range(1, max_replicas + 1):
            result.append(TopologyNoPP(tp=tp, replicas=r))
    return result


def enumerate_topologies_dp(max_gpus: int = 8) -> list[TopologyDP]:
    result = []
    for tp in TP_VALUES:
        for pp in PP_VALUES:
            if tp * pp > max_gpus:
                continue
            max_dp = max_gpus // (tp * pp)
            for dp in range(1, max_dp + 1):
                result.append(TopologyDP(tp=tp, pp=pp, dp=dp))
    return result
```

- [ ] **Step 4: Fix test_no_pp_topologies_validity (missing variable)**

The test references `triples` before assignment. Fix:

```python
def test_no_pp_topologies_validity():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    for tp, replicas in triples:
        assert tp * replicas <= 8
        assert tp in (1, 2, 4, 8)
        assert replicas >= 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/jchen/go/src/inference-sim/sim2real-config-exploration-validation && python -m pytest tests/config/test_topology.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/ tests/
git commit -m "feat: add topology enumeration for config space generation

Enumerates valid (TP, PP, replicas) triples satisfying GPU budget constraint.
Three variants: PP-supporting (25 triples), no-PP/inference-sim (15 triples),
and DP/llm-optimizer (25 triples with data_parallel_size)."
```

---

### Task 2: Output Schema (Pydantic Models)

**Files:**
- Create: `experiments/schema/__init__.py`
- Create: `experiments/schema/output.py`
- Test: `tests/schema/__init__.py`
- Test: `tests/schema/test_output.py`

The common output schema from the spec (lines 276-340). All tool runners emit this format. Pydantic ensures validation and serialization consistency.

- [ ] **Step 1: Write test for output schema**

```python
# tests/schema/__init__.py - empty
# tests/schema/test_output.py

import json
from experiments.schema.output import (
    ConfigResult,
    WorkloadInfo,
    VllmArgs,
    RoutingConfig,
    ToolConfig,
    Results,
    Metadata,
)


def test_full_config_result_serialization():
    """A complete config result round-trips through JSON."""
    result = ConfigResult(
        tool="inference-sim",
        workload=WorkloadInfo(
            model="meta-llama/Llama-3.1-8B",
            hardware="H100_SXM_80GB",
            preset="servegen_m-mid",
            num_requests=10000,
            isl_mean=512,
            isl_max=2048,
            osl_mean=256,
            osl_max=1024,
            arrival_pattern="poisson",
            slo_ttft_mean_ms=300,
            seed=42,
            trace_file="workloads/canonical_servegen_m-mid.yaml",
        ),
        vllm_args=VllmArgs(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            num_instances=4,
            data_parallel_size=1,
            max_num_seqs=128,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
            block_size=16,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_prefix_caching=False,
            enforce_eager=False,
            swap_space=4,
        ),
        routing_config=RoutingConfig(
            strategy="weighted-scoring",
            scorers="prefix-cache:2,queue-depth:1,kv-utilization:1",
            picker="max-score",
        ),
        tool_config=ToolConfig(
            scheduler="priority-fcfs",
            admission_policy="tier-shed",
            preemption_policy="priority",
        ),
        results=Results(
            max_throughput_tok_s=850.0,
            max_throughput_qps=12.5,
            ttft_mean_ms=245.0,
            ttft_p50_ms=220.0,
            ttft_p99_ms=380.0,
            tpot_mean_ms=18.5,
            meets_slo=True,
            cost_per_hour=25.60,
            cost_per_1k_tokens=0.0084,
        ),
        metadata=Metadata(
            status="ok",
            tool_version="v0.4.2",
            wall_clock_seconds=2.3,
            num_rate_probes=8,
            config_hash="a3f8c1d2",
        ),
    )
    data = json.loads(result.model_dump_json())
    assert data["tool"] == "inference-sim"
    assert data["vllm_args"]["tensor_parallel_size"] == 2
    assert data["routing_config"]["strategy"] == "weighted-scoring"
    assert data["results"]["meets_slo"] is True
    assert data["metadata"]["status"] == "ok"
    roundtrip = ConfigResult.model_validate(data)
    assert roundtrip == result


def test_failed_config_has_null_results():
    """Failed configs have status != 'ok' and null results."""
    result = ConfigResult(
        tool="llm-optimizer",
        workload=WorkloadInfo(
            model="meta-llama/Llama-3.1-8B",
            hardware="H100_SXM_80GB",
            preset="servegen_m-mid",
            num_requests=10000,
            isl_mean=512,
            isl_max=2048,
            osl_mean=256,
            osl_max=1024,
            arrival_pattern="poisson",
            slo_ttft_mean_ms=300,
            seed=42,
        ),
        vllm_args=VllmArgs(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            num_instances=1,
            data_parallel_size=1,
            max_num_seqs=512,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=False,
            block_size=16,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_prefix_caching=False,
            enforce_eager=False,
            swap_space=4,
        ),
        results=None,
        metadata=Metadata(status="oom", wall_clock_seconds=5.2),
    )
    data = json.loads(result.model_dump_json())
    assert data["results"] is None
    assert data["metadata"]["status"] == "oom"
    assert data["routing_config"] is None
    assert data["tool_config"] is None


def test_single_replica_has_null_routing():
    """Single-replica configs have routing_config=None."""
    result = ConfigResult(
        tool="vidur",
        workload=WorkloadInfo(
            model="meta-llama/Llama-3.1-8B",
            hardware="H100_SXM_80GB",
            preset="servegen_m-mid",
            num_requests=10000,
            isl_mean=512,
            isl_max=2048,
            osl_mean=256,
            osl_max=1024,
            arrival_pattern="poisson",
            slo_ttft_mean_ms=300,
            seed=42,
        ),
        vllm_args=VllmArgs(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            num_instances=1,
            data_parallel_size=1,
            max_num_seqs=128,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=False,
            block_size=16,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            enable_prefix_caching=False,
            enforce_eager=False,
            swap_space=4,
        ),
        routing_config=None,
        tool_config=ToolConfig(vidur_scheduler_type="vllm"),
        results=Results(
            max_throughput_tok_s=600.0,
            max_throughput_qps=9.0,
            ttft_mean_ms=180.0,
            meets_slo=True,
            cost_per_hour=12.80,
            cost_per_1k_tokens=0.006,
        ),
        metadata=Metadata(status="ok", wall_clock_seconds=45.0, config_hash="b1c2d3e4"),
    )
    assert result.routing_config is None
    data = json.loads(result.model_dump_json())
    assert data["routing_config"] is None


def test_config_hash_generation():
    """config_hash is deterministic from vllm_args + routing_config + tool_config."""
    from experiments.schema.output import compute_config_hash

    vllm = VllmArgs(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        num_instances=4,
        data_parallel_size=1,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        block_size=16,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        enforce_eager=False,
        swap_space=4,
    )
    routing = RoutingConfig(
        strategy="weighted-scoring",
        scorers="prefix-cache:2,queue-depth:1",
        picker="max-score",
    )
    tool = ToolConfig(scheduler="fcfs")
    h1 = compute_config_hash(vllm, routing, tool)
    h2 = compute_config_hash(vllm, routing, tool)
    assert h1 == h2
    assert len(h1) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/schema/test_output.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement output.py**

```python
# experiments/schema/__init__.py - empty
# experiments/schema/output.py

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class WorkloadInfo(BaseModel):
    model: str
    hardware: str
    preset: str
    num_requests: int
    isl_mean: int
    isl_max: int
    osl_mean: int
    osl_max: int
    arrival_pattern: str
    slo_ttft_mean_ms: int
    seed: int
    trace_file: Optional[str] = None


class VllmArgs(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    num_instances: int
    data_parallel_size: int = 1
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_chunked_prefill: bool
    block_size: int
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    swap_space: int = 4


class RoutingConfig(BaseModel):
    strategy: str
    scorers: Optional[str] = None
    picker: Optional[str] = None


class ToolConfig(BaseModel):
    scheduler: Optional[str] = None
    admission_policy: Optional[str] = None
    preemption_policy: Optional[str] = None
    max_concurrency: Optional[int] = None
    vidur_scheduler_type: Optional[str] = None


class Results(BaseModel):
    max_throughput_tok_s: float
    max_throughput_qps: float
    ttft_mean_ms: float
    ttft_p50_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_mean_ms: Optional[float] = None
    meets_slo: bool
    cost_per_hour: float
    cost_per_1k_tokens: Optional[float] = None


class Metadata(BaseModel):
    status: str = "ok"
    tool_version: Optional[str] = None
    wall_clock_seconds: Optional[float] = None
    num_rate_probes: Optional[int] = None
    config_hash: Optional[str] = None
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class ConfigResult(BaseModel):
    tool: str
    workload: WorkloadInfo
    vllm_args: VllmArgs
    routing_config: Optional[RoutingConfig] = None
    tool_config: Optional[ToolConfig] = None
    results: Optional[Results] = None
    metadata: Metadata = Field(default_factory=Metadata)


def compute_config_hash(
    vllm_args: VllmArgs,
    routing_config: Optional[RoutingConfig] = None,
    tool_config: Optional[ToolConfig] = None,
) -> str:
    blob = json.dumps(
        {
            "vllm_args": vllm_args.model_dump(exclude_none=True),
            "routing_config": routing_config.model_dump(exclude_none=True) if routing_config else None,
            "tool_config": tool_config.model_dump(exclude_none=True) if tool_config else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:8]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/schema/test_output.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/schema/ tests/schema/
git commit -m "feat: add Pydantic output schema for config results

Common JSON schema for all tool runners: workload info, vllm_args,
routing_config, tool_config, results, metadata with status field.
Supports null results for failed configs."
```

---

### Task 3: inference-sim Config Expansion

**Files:**
- Create: `experiments/config/blis_configs.py`
- Test: `tests/config/test_blis_configs.py`

Expands inference-sim's parameter space: no PP, 15 topology triples, 4 schedulers, 2 admission policies, 2 preemption policies, 6 routing configs (multi-replica only), chunked prefill with threshold pruning. Emits list of dicts ready for the runner.

- [ ] **Step 1: Write tests for inference-sim config expansion**

```python
# tests/config/test_blis_configs.py

from experiments.config.blis_configs import generate_blis_configs


def test_generates_configs():
    configs = generate_blis_configs()
    assert len(configs) > 0


def test_all_configs_have_required_fields():
    configs = generate_blis_configs()
    required = {"tp", "replicas", "max_num_seqs", "max_batched_tokens",
                "chunked_prefill_threshold", "block_size", "scheduler",
                "preemption_policy"}
    for c in configs:
        assert required.issubset(c.keys()), f"Missing fields in {c}"


def test_single_replica_configs_have_no_routing():
    configs = generate_blis_configs()
    single = [c for c in configs if c["replicas"] == 1]
    assert len(single) > 0
    for c in single:
        assert "routing_policy" not in c or c["routing_policy"] is None
        assert "admission_policy" not in c or c["admission_policy"] is None


def test_multi_replica_configs_have_routing():
    configs = generate_blis_configs()
    multi = [c for c in configs if c["replicas"] > 1]
    assert len(multi) > 0
    for c in multi:
        assert c["routing_policy"] is not None
        assert c["admission_policy"] is not None


def test_no_pp_in_any_config():
    """inference-sim does not support PP."""
    configs = generate_blis_configs()
    for c in configs:
        assert "pp" not in c


def test_chunked_prefill_threshold_pruning():
    """Threshold must be < max_batched_tokens when chunked prefill is enabled."""
    configs = generate_blis_configs()
    for c in configs:
        threshold = c["chunked_prefill_threshold"]
        if threshold > 0:
            assert threshold < c["max_batched_tokens"], (
                f"threshold {threshold} >= max_batched_tokens {c['max_batched_tokens']}"
            )


def test_config_count_in_expected_range():
    """Spec says ~25,000 after pruning."""
    configs = generate_blis_configs()
    assert 20_000 <= len(configs) <= 30_000, f"Got {len(configs)} configs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/config/test_blis_configs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement blis_configs.py**

```python
# experiments/config/blis_configs.py

from experiments.config.topology import enumerate_topologies_no_pp

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
MAX_BATCHED_TOKENS = (2048, 4096, 8192)
CHUNKED_PREFILL_THRESHOLDS = (1024, 2048, 4096)
BLOCK_SIZES = (16, 32)
SCHEDULERS = ("fcfs", "priority-fcfs", "sjf", "reverse-priority")
ADMISSION_POLICIES = ("always-admit", "tier-shed")
PREEMPTION_POLICIES = ("fcfs", "priority")
SIMPLE_ROUTING = ("round-robin", "least-loaded")
WEIGHTED_ROUTING = (
    "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    "queue-depth:1,kv-utilization:1",
    "precise-prefix-cache:2,load-balance:1",
    "vllm-dp:1",
)


def _batching_combos() -> list[tuple[int, int, int]]:
    combos = []
    for seqs in MAX_NUM_SEQS:
        for tokens in MAX_BATCHED_TOKENS:
            # Disabled chunked prefill: threshold=0
            combos.append((seqs, tokens, 0))
            # Enabled: each threshold must be < max_batched_tokens
            for thresh in CHUNKED_PREFILL_THRESHOLDS:
                if thresh < tokens:
                    combos.append((seqs, tokens, thresh))
    return combos


def generate_blis_configs(max_gpus: int = 8) -> list[dict]:
    topologies = enumerate_topologies_no_pp(max_gpus)
    batching = _batching_combos()
    configs = []

    for topo in topologies:
        for seqs, tokens, threshold in batching:
            base = {
                "tp": topo.tp,
                "replicas": topo.replicas,
                "max_num_seqs": seqs,
                "max_batched_tokens": tokens,
                "chunked_prefill_threshold": threshold,
            }
            if topo.is_multi_replica:
                for sched in SCHEDULERS:
                    for admission in ADMISSION_POLICIES:
                        for preemption in PREEMPTION_POLICIES:
                            for block in BLOCK_SIZES:
                                for policy in SIMPLE_ROUTING:
                                    configs.append({
                                        **base,
                                        "block_size": block,
                                        "scheduler": sched,
                                        "admission_policy": admission,
                                        "preemption_policy": preemption,
                                        "routing_policy": policy,
                                        "routing_scorers": None,
                                    })
                                for scorers in WEIGHTED_ROUTING:
                                    configs.append({
                                        **base,
                                        "block_size": block,
                                        "scheduler": sched,
                                        "admission_policy": admission,
                                        "preemption_policy": preemption,
                                        "routing_policy": "weighted",
                                        "routing_scorers": scorers,
                                    })
            else:
                for sched in SCHEDULERS:
                    for preemption in PREEMPTION_POLICIES:
                        for block in BLOCK_SIZES:
                            configs.append({
                                **base,
                                "block_size": block,
                                "scheduler": sched,
                                "admission_policy": None,
                                "preemption_policy": preemption,
                                "routing_policy": None,
                                "routing_scorers": None,
                            })
    return configs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_blis_configs.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/config/blis_configs.py tests/config/test_blis_configs.py
git commit -m "feat: add inference-sim config expansion

15 topology triples (no PP), batching combos with threshold pruning,
scheduler/admission/preemption/routing cross-product for multi-replica.
~25k configs after pruning."
```

---

### Task 4: LLMServingSim Config Expansion

**Files:**
- Create: `experiments/config/llmservingsim_configs.py`
- Test: `tests/config/test_llmservingsim_configs.py`

PP-supporting (25 triples), 3 routing policies for multi-replica, chunked prefill with 4 thresholds (0=uncapped, 1024, 2048, 4096), block size, prefix caching.

- [ ] **Step 1: Write tests**

```python
# tests/config/test_llmservingsim_configs.py

from experiments.config.llmservingsim_configs import generate_llmservingsim_configs


def test_generates_configs():
    configs = generate_llmservingsim_configs()
    assert len(configs) > 0


def test_all_configs_have_required_fields():
    configs = generate_llmservingsim_configs()
    required = {"tp", "pp", "replicas", "max_num_seqs", "max_batched_tokens",
                "enable_chunked_prefill", "chunked_prefill_threshold",
                "block_size", "prefix_caching"}
    for c in configs:
        assert required.issubset(c.keys()), f"Missing: {required - c.keys()}"


def test_multi_replica_has_routing():
    configs = generate_llmservingsim_configs()
    multi = [c for c in configs if c["replicas"] > 1]
    assert all(c["routing_policy"] in ("LOAD", "RR", "RAND") for c in multi)


def test_single_replica_no_routing():
    configs = generate_llmservingsim_configs()
    single = [c for c in configs if c["replicas"] == 1]
    assert all(c.get("routing_policy") is None for c in single)


def test_threshold_zero_only_when_enabled():
    """Threshold=0 (uncapped) is only valid when chunked prefill is enabled."""
    configs = generate_llmservingsim_configs()
    for c in configs:
        if not c["enable_chunked_prefill"]:
            assert c["chunked_prefill_threshold"] is None


def test_threshold_pruning():
    configs = generate_llmservingsim_configs()
    for c in configs:
        thresh = c["chunked_prefill_threshold"]
        if thresh is not None and thresh > 0:
            assert thresh < c["max_batched_tokens"]


def test_config_count_in_expected_range():
    """Spec says ~4,700 after pruning."""
    configs = generate_llmservingsim_configs()
    assert 4_000 <= len(configs) <= 5_500, f"Got {len(configs)} configs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/config/test_llmservingsim_configs.py -v`

- [ ] **Step 3: Implement llmservingsim_configs.py**

```python
# experiments/config/llmservingsim_configs.py

from experiments.config.topology import enumerate_topologies

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
MAX_BATCHED_TOKENS = (2048, 4096, 8192)
CHUNKED_PREFILL_THRESHOLDS_ENABLED = (0, 1024, 2048, 4096)  # 0 = uncapped
BLOCK_SIZES = (16, 32)
PREFIX_CACHING = (True, False)
ROUTING_POLICIES = ("LOAD", "RR", "RAND")


def generate_llmservingsim_configs(max_gpus: int = 8) -> list[dict]:
    topologies = enumerate_topologies(max_gpus)
    configs = []

    for topo in topologies:
        for seqs in MAX_NUM_SEQS:
            for tokens in MAX_BATCHED_TOKENS:
                for block in BLOCK_SIZES:
                    for prefix in PREFIX_CACHING:
                        # Disabled chunked prefill
                        base = {
                            "tp": topo.tp,
                            "pp": topo.pp,
                            "replicas": topo.replicas,
                            "max_num_seqs": seqs,
                            "max_batched_tokens": tokens,
                            "enable_chunked_prefill": False,
                            "chunked_prefill_threshold": None,
                            "block_size": block,
                            "prefix_caching": prefix,
                        }
                        if topo.is_multi_replica:
                            for routing in ROUTING_POLICIES:
                                configs.append({**base, "routing_policy": routing})
                        else:
                            configs.append({**base, "routing_policy": None})

                        # Enabled chunked prefill with each threshold
                        for thresh in CHUNKED_PREFILL_THRESHOLDS_ENABLED:
                            if thresh > 0 and thresh >= tokens:
                                continue
                            enabled_base = {
                                "tp": topo.tp,
                                "pp": topo.pp,
                                "replicas": topo.replicas,
                                "max_num_seqs": seqs,
                                "max_batched_tokens": tokens,
                                "enable_chunked_prefill": True,
                                "chunked_prefill_threshold": thresh,
                                "block_size": block,
                                "prefix_caching": prefix,
                            }
                            if topo.is_multi_replica:
                                for routing in ROUTING_POLICIES:
                                    configs.append({**enabled_base, "routing_policy": routing})
                            else:
                                configs.append({**enabled_base, "routing_policy": None})

    return configs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_llmservingsim_configs.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/config/llmservingsim_configs.py tests/config/test_llmservingsim_configs.py
git commit -m "feat: add LLMServingSim config expansion

25 topology triples with PP, chunked prefill thresholds (0=uncapped),
3 routing policies for multi-replica, block size and prefix caching.
~4,700 configs after pruning."
```

---

### Task 5: AIConfigurator Config Expansion

**Files:**
- Create: `experiments/config/aiconfigurator_configs.py`
- Test: `tests/config/test_aiconfigurator_configs.py`

AIConfigurator uses `cli estimate` per (TP, PP) pair. Its internal sweep handles batch sizes and ctx_tokens (~80-100 values). We enumerate topology triples (25) with deployment replicas; AIConfigurator applies linear throughput scaling for multi-replica.

- [ ] **Step 1: Write tests**

```python
# tests/config/test_aiconfigurator_configs.py

from experiments.config.aiconfigurator_configs import generate_aiconfigurator_configs


def test_generates_configs():
    configs = generate_aiconfigurator_configs()
    assert len(configs) > 0


def test_configs_are_topology_triples():
    """AIConfigurator sweeps internally; we only pass topology + ISL/OSL."""
    configs = generate_aiconfigurator_configs()
    required = {"tp", "pp", "replicas"}
    for c in configs:
        assert required.issubset(c.keys())


def test_count_matches_topology():
    """One config per topology triple (internal sweep handles rest)."""
    configs = generate_aiconfigurator_configs()
    assert len(configs) == 25


def test_no_routing_config():
    """AIConfigurator has no routing model."""
    configs = generate_aiconfigurator_configs()
    for c in configs:
        assert c.get("routing_policy") is None
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement aiconfigurator_configs.py**

```python
# experiments/config/aiconfigurator_configs.py

from experiments.config.topology import enumerate_topologies


def generate_aiconfigurator_configs(max_gpus: int = 8) -> list[dict]:
    configs = []
    for topo in enumerate_topologies(max_gpus):
        configs.append({
            "tp": topo.tp,
            "pp": topo.pp,
            "replicas": topo.replicas,
            "routing_policy": None,
        })
    return configs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_aiconfigurator_configs.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/config/aiconfigurator_configs.py tests/config/test_aiconfigurator_configs.py
git commit -m "feat: add AIConfigurator config expansion

25 topology triples; AIConfigurator handles internal batch/ctx_tokens sweep.
Linear throughput scaling for multi-replica."
```

---

### Task 6: Vidur Config Expansion

**Files:**
- Create: `experiments/config/vidur_configs.py`
- Test: `tests/config/test_vidur_configs.py`

25 topology triples, 7 scheduler variants (3 vllm + 3 sarathi + 1 orca), block size, 3 routing policies for multi-replica.

- [ ] **Step 1: Write tests**

```python
# tests/config/test_vidur_configs.py

from experiments.config.vidur_configs import generate_vidur_configs


def test_generates_configs():
    configs = generate_vidur_configs()
    assert len(configs) > 0


def test_scheduler_variants():
    configs = generate_vidur_configs()
    schedulers = {c["scheduler_type"] for c in configs}
    assert "vllm" in schedulers
    assert "sarathi" in schedulers
    assert "orca" in schedulers


def test_vllm_scheduler_has_max_tokens_in_batch():
    configs = generate_vidur_configs()
    vllm_configs = [c for c in configs if c["scheduler_type"] == "vllm"]
    assert all(c["max_tokens_in_batch"] is not None for c in vllm_configs)


def test_sarathi_scheduler_has_chunk_size():
    configs = generate_vidur_configs()
    sarathi_configs = [c for c in configs if c["scheduler_type"] == "sarathi"]
    assert all(c["chunk_size"] is not None for c in sarathi_configs)


def test_orca_scheduler_minimal():
    configs = generate_vidur_configs()
    orca_configs = [c for c in configs if c["scheduler_type"] == "orca"]
    assert all(c.get("max_tokens_in_batch") is None for c in orca_configs)
    assert all(c.get("chunk_size") is None for c in orca_configs)


def test_multi_replica_has_routing():
    configs = generate_vidur_configs()
    multi = [c for c in configs if c["replicas"] > 1]
    assert all(c["routing"] in ("round_robin", "lor", "random") for c in multi)


def test_config_count_in_expected_range():
    """Spec says ~3,500 after pruning."""
    configs = generate_vidur_configs()
    assert 3_000 <= len(configs) <= 4_500, f"Got {len(configs)} configs"
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement vidur_configs.py**

```python
# experiments/config/vidur_configs.py

from experiments.config.topology import enumerate_topologies

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
BLOCK_SIZES = (16, 32)
VLLM_MAX_TOKENS = (2048, 4096, 8192)
SARATHI_CHUNK_SIZES = (1024, 2048, 4096)
ROUTING_POLICIES = ("round_robin", "lor", "random")


def _scheduler_variants() -> list[dict]:
    variants = []
    for tokens in VLLM_MAX_TOKENS:
        variants.append({
            "scheduler_type": "vllm",
            "max_tokens_in_batch": tokens,
            "chunk_size": None,
        })
    for chunk in SARATHI_CHUNK_SIZES:
        variants.append({
            "scheduler_type": "sarathi",
            "max_tokens_in_batch": None,
            "chunk_size": chunk,
        })
    variants.append({
        "scheduler_type": "orca",
        "max_tokens_in_batch": None,
        "chunk_size": None,
    })
    return variants


def generate_vidur_configs(max_gpus: int = 8) -> list[dict]:
    topologies = enumerate_topologies(max_gpus)
    scheduler_variants = _scheduler_variants()
    configs = []

    for topo in topologies:
        for seqs in MAX_NUM_SEQS:
            for sched in scheduler_variants:
                for block in BLOCK_SIZES:
                    base = {
                        "tp": topo.tp,
                        "pp": topo.pp,
                        "replicas": topo.replicas,
                        "max_num_seqs": seqs,
                        "block_size": block,
                        **sched,
                    }
                    if topo.is_multi_replica:
                        for routing in ROUTING_POLICIES:
                            configs.append({**base, "routing": routing})
                    else:
                        configs.append({**base, "routing": None})

    return configs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_vidur_configs.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/config/vidur_configs.py tests/config/test_vidur_configs.py
git commit -m "feat: add Vidur config expansion

25 topology triples, 7 scheduler variants (vllm/sarathi/orca),
block size, 3 routing policies for multi-replica. ~3,500 configs."
```

---

### Task 7: llm-optimizer Config Expansion and Grid Search Syntax

**Files:**
- Create: `experiments/config/llm_optimizer_configs.py`
- Test: `tests/config/test_llm_optimizer_configs.py`

Uses `data_parallel_size` instead of deployment replicas. Generates both a config list (for counting/analysis) and the native grid search CLI arguments (for actual execution via `--server-args` / `--client-args`).

- [ ] **Step 1: Write tests**

```python
# tests/config/test_llm_optimizer_configs.py

from experiments.config.llm_optimizer_configs import (
    generate_llm_optimizer_configs,
    build_grid_search_args,
)


def test_generates_configs():
    configs = generate_llm_optimizer_configs()
    assert len(configs) > 0


def test_uses_dp_not_replicas():
    configs = generate_llm_optimizer_configs()
    for c in configs:
        assert "dp" in c
        assert "replicas" not in c


def test_dp_validity():
    configs = generate_llm_optimizer_configs()
    for c in configs:
        assert c["tp"] * c["pp"] * c["dp"] <= 8


def test_no_routing_config():
    """llm-optimizer has no external routing."""
    configs = generate_llm_optimizer_configs()
    for c in configs:
        assert c.get("routing_policy") is None


def test_config_count_in_expected_range():
    """Spec says ~11,200 after pruning."""
    configs = generate_llm_optimizer_configs()
    assert 10_000 <= len(configs) <= 13_000, f"Got {len(configs)} configs"


def test_grid_search_args_structure():
    """Native grid search args should have server_args and client_args."""
    args = build_grid_search_args()
    assert "server_args" in args
    assert "client_args" in args
    assert isinstance(args["server_args"], list)
    assert isinstance(args["client_args"], list)


def test_grid_search_args_contain_tp_dp_pairs():
    """Server args should include paired TP*PP*DP combinations."""
    args = build_grid_search_args()
    server_str = " ".join(args["server_args"])
    assert "tensor_parallel_size" in server_str or "tp" in server_str.lower()
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement llm_optimizer_configs.py**

```python
# experiments/config/llm_optimizer_configs.py

from experiments.config.topology import enumerate_topologies_dp

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
MAX_BATCHED_TOKENS = (2048, 4096, 8192)
CHUNKED_PREFILL = (True, False)
BLOCK_SIZES = (16, 32)
PREFIX_CACHING = (True, False)
MAX_CONCURRENCY = (32, 64, 128, 256)


def generate_llm_optimizer_configs(max_gpus: int = 8) -> list[dict]:
    topologies = enumerate_topologies_dp(max_gpus)
    configs = []

    for topo in topologies:
        for seqs in MAX_NUM_SEQS:
            for tokens in MAX_BATCHED_TOKENS:
                for chunked in CHUNKED_PREFILL:
                    for block in BLOCK_SIZES:
                        for prefix in PREFIX_CACHING:
                            for concurrency in MAX_CONCURRENCY:
                                configs.append({
                                    "tp": topo.tp,
                                    "pp": topo.pp,
                                    "dp": topo.dp,
                                    "max_num_seqs": seqs,
                                    "max_batched_tokens": tokens,
                                    "enable_chunked_prefill": chunked,
                                    "block_size": block,
                                    "prefix_caching": prefix,
                                    "max_concurrency": concurrency,
                                    "routing_policy": None,
                                })

    return configs


def build_grid_search_args(max_gpus: int = 8) -> dict:
    topologies = enumerate_topologies_dp(max_gpus)
    tp_pp_dp_pairs = [(t.tp, t.pp, t.dp) for t in topologies]
    pairs_str = ",".join(f"({tp},{pp},{dp})" for tp, pp, dp in tp_pp_dp_pairs)

    seqs_str = ",".join(str(s) for s in MAX_NUM_SEQS)
    tokens_str = ",".join(str(t) for t in MAX_BATCHED_TOKENS)
    block_str = ",".join(str(b) for b in BLOCK_SIZES)
    concurrency_str = ",".join(str(c) for c in MAX_CONCURRENCY)

    server_args = [
        f"tensor_parallel_size*pipeline_parallel_size*data_parallel_size=[{pairs_str}]",
        f"max_num_seqs=[{seqs_str}]",
        f"max_num_batched_tokens=[{tokens_str}]",
        "enable_chunked_prefill=[true,false]",
        f"block_size=[{block_str}]",
        "enable_prefix_caching=[true,false]",
    ]

    client_args = [
        f"max_concurrency=[{concurrency_str}]",
        "num_prompts=10000",
        "dataset_name=sharegpt",
    ]

    return {
        "server_args": server_args,
        "client_args": client_args,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_llm_optimizer_configs.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/config/llm_optimizer_configs.py tests/config/test_llm_optimizer_configs.py
git commit -m "feat: add llm-optimizer config expansion and grid search syntax

25 DP topology triples, native grid search arg generation for
--server-args and --client-args. ~11,200 configs."
```

---

## Chunk 2: Runner Infrastructure and Workload Generation

### Task 8: Base Runner with Checkpointing and Error Handling

**Files:**
- Create: `experiments/runners/__init__.py`
- Create: `experiments/runners/base.py`
- Test: `tests/runners/__init__.py`
- Test: `tests/runners/test_base.py`

Shared infrastructure: JSONL append, config hash dedup on resume, subprocess timeout, error status recording. All 5 tool runners inherit from this.

- [ ] **Step 1: Write tests**

```python
# tests/runners/__init__.py - empty
# tests/runners/test_base.py

import json
import tempfile
from pathlib import Path

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, WorkloadInfo, VllmArgs, Metadata, Results,
)


def _make_workload():
    return WorkloadInfo(
        model="meta-llama/Llama-3.1-8B",
        hardware="H100_SXM_80GB",
        preset="servegen_m-mid",
        num_requests=10000,
        isl_mean=512, isl_max=2048,
        osl_mean=256, osl_max=1024,
        arrival_pattern="poisson",
        slo_ttft_mean_ms=300,
        seed=42,
    )


def _make_vllm_args(**overrides):
    defaults = dict(
        tensor_parallel_size=2, pipeline_parallel_size=1,
        num_instances=1, data_parallel_size=1,
        max_num_seqs=128, max_num_batched_tokens=4096,
        enable_chunked_prefill=False, block_size=16,
        gpu_memory_utilization=0.9, dtype="bfloat16",
        kv_cache_dtype="auto", enable_prefix_caching=False,
        enforce_eager=False, swap_space=4,
    )
    defaults.update(overrides)
    return VllmArgs(**defaults)


class DummyRunner(BaseRunner):
    tool_name = "test-tool"
    timeout_seconds = 5

    def evaluate_config(self, config: dict) -> ConfigResult:
        return ConfigResult(
            tool=self.tool_name,
            workload=self.workload,
            vllm_args=_make_vllm_args(**{k: config[k] for k in ("tensor_parallel_size",) if k in config}),
            results=Results(
                max_throughput_tok_s=100.0, max_throughput_qps=1.0,
                ttft_mean_ms=200.0, meets_slo=True,
                cost_per_hour=6.40, cost_per_1k_tokens=0.01,
            ),
            metadata=Metadata(status="ok", config_hash="abcd1234"),
        )


def test_append_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.jsonl"
        runner = DummyRunner(workload=_make_workload(), output_path=output)
        result = runner.evaluate_config({"tensor_parallel_size": 2})
        runner.append_result(result)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["tool"] == "test-tool"


def test_resume_skips_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.jsonl"
        runner = DummyRunner(workload=_make_workload(), output_path=output)
        result = runner.evaluate_config({"tensor_parallel_size": 2})
        runner.append_result(result)

        completed = runner.load_completed_hashes()
        assert "abcd1234" in completed


def test_run_batch_skips_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.jsonl"
        runner = DummyRunner(workload=_make_workload(), output_path=output)

        configs = [{"tensor_parallel_size": 2}, {"tensor_parallel_size": 4}]
        runner.run_batch(configs, hash_fn=lambda c: f"hash-{c['tensor_parallel_size']}")
        assert output.read_text().count("\n") == 2

        runner2 = DummyRunner(workload=_make_workload(), output_path=output)
        runner2.run_batch(configs, hash_fn=lambda c: f"hash-{c['tensor_parallel_size']}")
        assert output.read_text().count("\n") == 2  # no duplicates
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement base.py**

```python
# experiments/runners/__init__.py - empty
# experiments/runners/base.py

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from experiments.schema.output import ConfigResult, WorkloadInfo

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
            data = json.loads(line)
            h = data.get("metadata", {}).get("config_hash")
            if h:
                hashes.add(h)
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

            logger.info(f"[{i+1}/{total}] Evaluating config {config_hash}")
            try:
                result = self.evaluate_config(config)
                if result.metadata.config_hash is None:
                    result.metadata.config_hash = config_hash
                self.append_result(result)
                completed.add(config_hash)
            except Exception as e:
                logger.error(f"Config {config_hash} failed: {e}")
                from experiments.schema.output import Metadata
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
            logger.info(f"Skipped {skipped} already-completed configs")

    def _config_to_vllm_args(self, config: dict):
        # Config dicts use tool-native keys: tp, pp, replicas/dp, max_num_seqs,
        # max_batched_tokens, enable_chunked_prefill, block_size, prefix_caching.
        # Defaults handle missing keys for tools with fewer dimensions.
        from experiments.schema.output import VllmArgs
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/runners/test_base.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/runners/ tests/runners/
git commit -m "feat: add base runner with JSONL checkpointing and error handling

Append-only JSONL output, config hash dedup on resume, batch execution
with automatic error recording. All tool runners inherit from this."
```

---

### Task 9: Workload Generation and Format Conversion

**Files:**
- Create: `experiments/workloads/__init__.py`
- Create: `experiments/workloads/generate.py`
- Test: `tests/workloads/__init__.py`
- Test: `tests/workloads/test_generate.py`

Generates canonical trace via `blis convert preset`, then converts to each tool's native format. The canonical trace is generated once; conversions are deterministic. inference-sim reads the canonical v2 WorkloadSpec YAML directly via `--workload-spec` (no conversion needed). AIConfigurator uses analytical ISL/OSL parameters (no trace).

- [ ] **Step 1: Write tests**

```python
# tests/workloads/__init__.py - empty
# tests/workloads/test_generate.py

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from experiments.workloads.generate import (
    generate_canonical_trace,
    convert_to_llmservingsim,
    convert_to_vidur,
    convert_to_llm_optimizer,
    CANONICAL_NUM_REQUESTS,
)


def test_canonical_trace_calls_blis():
    """Verify blis convert is invoked with correct args."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "canonical.yaml"
        with patch("experiments.workloads.generate.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="yaml content")
            generate_canonical_trace(output, blis_binary="./blis")
            args = mock_run.call_args[0][0]
            assert "./blis" in args[0]
            assert "convert" in args
            assert "preset" in args


def test_convert_to_llmservingsim_format():
    """JSONL output has required fields."""
    requests = [
        {"arrival_time": 0.0, "input_tokens": 512, "output_tokens": 256},
        {"arrival_time": 0.1, "input_tokens": 128, "output_tokens": 64},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "trace.jsonl"
        convert_to_llmservingsim(requests, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "input_toks" in record
        assert "output_toks" in record
        assert "arrival_time" in record


def test_convert_to_vidur_format():
    """CSV output has required columns."""
    requests = [
        {"arrival_time": 0.0, "input_tokens": 512, "output_tokens": 256},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "trace.csv"
        convert_to_vidur(requests, output)
        content = output.read_text()
        assert "request_id" in content
        assert "arrival_time" in content
        assert "prefill_tokens" in content
        assert "decode_tokens" in content


def test_convert_to_llm_optimizer_format():
    """ShareGPT-format JSON with pre-tokenized lengths."""
    requests = [
        {"arrival_time": 0.0, "input_tokens": 512, "output_tokens": 256},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "dataset.json"
        convert_to_llm_optimizer(requests, output)
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert "conversations" in data[0] or "input_len" in data[0]
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement generate.py**

```python
# experiments/workloads/__init__.py - empty
# experiments/workloads/generate.py

import csv
import json
import subprocess
from pathlib import Path

CANONICAL_NUM_REQUESTS = 10000
CANONICAL_SEED = 42


def generate_canonical_trace(
    output_path: Path,
    blis_binary: str = "./estimators/inference-sim/blis",
    preset: str = "servegen",
    variant: str = "m-mid",
    num_requests: int = CANONICAL_NUM_REQUESTS,
    rate: float = 10.0,
    seed: int = CANONICAL_SEED,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            blis_binary, "convert", "preset",
            "--name", preset,
            "--variant", variant,
            "--num-requests", str(num_requests),
            "--rate", str(rate),
            "--seed", str(seed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    output_path.write_text(result.stdout)


def parse_canonical_trace(trace_path: Path) -> list[dict]:
    import yaml
    with open(trace_path) as f:
        spec = yaml.safe_load(f)
    requests = []
    if "requests" in spec:
        for r in spec["requests"]:
            requests.append({
                "arrival_time": r.get("arrival_time", 0.0),
                "input_tokens": r.get("input_tokens", r.get("isl", 512)),
                "output_tokens": r.get("output_tokens", r.get("osl", 256)),
            })
    return requests


def convert_to_llmservingsim(requests: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in requests:
            record = {
                "input_toks": r["input_tokens"],
                "output_toks": r["output_tokens"],
                "arrival_time": r["arrival_time"],
            }
            f.write(json.dumps(record) + "\n")


def convert_to_vidur(requests: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "arrival_time", "prefill_tokens", "decode_tokens"])
        for i, r in enumerate(requests):
            writer.writerow([i, r["arrival_time"], r["input_tokens"], r["output_tokens"]])


def convert_to_llm_optimizer(requests: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = []
    for r in requests:
        dataset.append({
            "input_len": r["input_tokens"],
            "output_len": r["output_tokens"],
        })
    output_path.write_text(json.dumps(dataset, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/workloads/test_generate.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/workloads/ tests/workloads/
git commit -m "feat: add workload generation and per-tool format conversion

Generates canonical trace via blis convert preset, converts to
LLMServingSim JSONL, Vidur CSV, and llm-optimizer sharegpt JSON.
AIConfigurator uses analytical ISL/OSL directly (no trace)."
```

---

## Chunk 3: Analysis Pipeline

### Task 10: Pareto Front Computation

**Files:**
- Create: `experiments/analysis/__init__.py`
- Create: `experiments/analysis/pareto.py`
- Test: `tests/analysis/__init__.py`
- Test: `tests/analysis/test_pareto.py`

Given a list of ConfigResults, compute the Pareto front on (cost_per_hour, max_throughput_tok_s). Filter to SLO-meeting configs first.

- [ ] **Step 1: Write tests**

```python
# tests/analysis/__init__.py - empty
# tests/analysis/test_pareto.py

from experiments.analysis.pareto import compute_pareto_front


def _make_point(cost: float, throughput: float, meets_slo: bool = True):
    return {"cost_per_hour": cost, "max_throughput_tok_s": throughput, "meets_slo": meets_slo}


def test_simple_pareto():
    points = [
        _make_point(10, 500),   # dominated by (10, 800)
        _make_point(10, 800),   # Pareto
        _make_point(20, 1000),  # Pareto
        _make_point(20, 600),   # dominated by (20, 1000)
        _make_point(30, 900),   # dominated by (20, 1000)
    ]
    front = compute_pareto_front(points)
    assert len(front) == 2
    costs = {p["cost_per_hour"] for p in front}
    assert costs == {10, 20}


def test_filters_slo_violations():
    points = [
        _make_point(10, 800, meets_slo=True),
        _make_point(5, 1200, meets_slo=False),
    ]
    front = compute_pareto_front(points)
    assert len(front) == 1
    assert front[0]["cost_per_hour"] == 10


def test_empty_input():
    assert compute_pareto_front([]) == []


def test_all_slo_violations():
    points = [_make_point(10, 500, meets_slo=False)]
    assert compute_pareto_front(points) == []


def test_single_point():
    points = [_make_point(10, 500)]
    front = compute_pareto_front(points)
    assert len(front) == 1
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement pareto.py**

```python
# experiments/analysis/__init__.py - empty
# experiments/analysis/pareto.py


def compute_pareto_front(points: list[dict]) -> list[dict]:
    viable = [p for p in points if p.get("meets_slo", False)]
    if not viable:
        return []

    # Sort by cost ascending, throughput descending (for tie-breaking)
    viable.sort(key=lambda p: (p["cost_per_hour"], -p["max_throughput_tok_s"]))

    front = []
    max_throughput = float("-inf")
    for p in viable:
        if p["max_throughput_tok_s"] > max_throughput:
            front.append(p)
            max_throughput = p["max_throughput_tok_s"]

    return front
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_pareto.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/analysis/ tests/analysis/
git commit -m "feat: add Pareto front computation

Filters to SLO-meeting configs, computes non-dominated front on
(cost_per_hour, max_throughput_tok_s). Used by select.py for top-3."
```

---

### Task 11: Top-3 Selection

**Files:**
- Create: `experiments/analysis/select.py`
- Test: `tests/analysis/test_select.py`

Select top-3 cheapest configs from Pareto front that exceed minimum throughput threshold.

- [ ] **Step 1: Write tests**

```python
# tests/analysis/test_select.py

from experiments.analysis.select import select_top_k


def _make_point(cost, throughput):
    return {"cost_per_hour": cost, "max_throughput_tok_s": throughput, "meets_slo": True}


def test_selects_cheapest_3():
    front = [
        _make_point(6.40, 300),
        _make_point(12.80, 600),
        _make_point(19.20, 900),
        _make_point(25.60, 1100),
    ]
    selected = select_top_k(front, k=3, min_throughput=200)
    assert len(selected) == 3
    assert selected[0]["cost_per_hour"] == 6.40


def test_filters_below_min_throughput():
    front = [
        _make_point(6.40, 50),   # below threshold
        _make_point(12.80, 600),
        _make_point(19.20, 900),
    ]
    selected = select_top_k(front, k=3, min_throughput=200)
    assert len(selected) == 2


def test_fewer_than_k():
    front = [_make_point(6.40, 300)]
    selected = select_top_k(front, k=3, min_throughput=200)
    assert len(selected) == 1


def test_empty_front():
    assert select_top_k([], k=3, min_throughput=200) == []
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement select.py**

```python
# experiments/analysis/select.py


def select_top_k(
    pareto_front: list[dict],
    k: int = 3,
    min_throughput: float = 200.0,
) -> list[dict]:
    eligible = [
        p for p in pareto_front
        if p["max_throughput_tok_s"] >= min_throughput
    ]
    eligible.sort(key=lambda p: p["cost_per_hour"])
    return eligible[:k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_select.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/analysis/select.py tests/analysis/test_select.py
git commit -m "feat: add top-3 config selection from Pareto front

Selects cheapest k configs exceeding minimum throughput threshold."
```

---

### Task 12: Drift Metric Computation

**Files:**
- Create: `experiments/analysis/drift.py`
- Test: `tests/analysis/test_drift.py`

Computes sim2real drift between predicted and actual measurements.

- [ ] **Step 1: Write tests**

```python
# tests/analysis/test_drift.py

from experiments.analysis.drift import compute_drift


def test_compute_drift():
    predicted = {"ttft_mean_ms": 200.0, "max_throughput_tok_s": 800.0}
    actual = {"ttft_mean_ms": 250.0, "max_throughput_tok_s": 720.0}
    drift = compute_drift(predicted, actual, slo_ttft_mean_ms=300)
    assert drift["drift_ttft_ms"] == 50.0
    assert drift["drift_throughput_tok_s"] == -80.0
    assert drift["slo_violation"] is False


def test_slo_violation_detected():
    predicted = {"ttft_mean_ms": 280.0, "max_throughput_tok_s": 500.0}
    actual = {"ttft_mean_ms": 320.0, "max_throughput_tok_s": 450.0}
    drift = compute_drift(predicted, actual, slo_ttft_mean_ms=300)
    assert drift["slo_violation"] is True


def test_negative_drift_means_better_than_predicted():
    predicted = {"ttft_mean_ms": 250.0, "max_throughput_tok_s": 600.0}
    actual = {"ttft_mean_ms": 220.0, "max_throughput_tok_s": 650.0}
    drift = compute_drift(predicted, actual, slo_ttft_mean_ms=300)
    assert drift["drift_ttft_ms"] == -30.0
    assert drift["drift_throughput_tok_s"] == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement drift.py**

```python
# experiments/analysis/drift.py


def compute_drift(
    predicted: dict,
    actual: dict,
    slo_ttft_mean_ms: float = 300.0,
) -> dict:
    return {
        "drift_ttft_ms": actual["ttft_mean_ms"] - predicted["ttft_mean_ms"],
        "drift_throughput_tok_s": actual["max_throughput_tok_s"] - predicted["max_throughput_tok_s"],
        "slo_violation": actual["ttft_mean_ms"] > slo_ttft_mean_ms,
        "predicted_ttft_mean_ms": predicted["ttft_mean_ms"],
        "actual_ttft_mean_ms": actual["ttft_mean_ms"],
        "predicted_throughput_tok_s": predicted["max_throughput_tok_s"],
        "actual_throughput_tok_s": actual["max_throughput_tok_s"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_drift.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/analysis/drift.py tests/analysis/test_drift.py
git commit -m "feat: add drift metric computation

Computes TTFT drift, throughput drift, and SLO violation from
predicted vs. actual measurements."
```

---

## Chunk 4: Tool Runner Stubs

Each runner implements the tool-specific invocation logic. These are stubs that define the interface and CLI argument construction; the actual tool invocation requires the estimator binaries/environments to be installed. Tests verify argument construction and output schema compliance.

### Task 13: inference-sim Runner

**Files:**
- Create: `experiments/runners/run_blis.py`
- Test: `tests/runners/test_run_blis.py`

Binary search over `--rate`: start at 10, double until mean TTFT > 300ms, then bisect to convergence (8 iterations). Each probe invokes `blis run` with the config's parameters.

- [ ] **Step 1: Write tests**

```python
# tests/runners/test_run_blis.py

from experiments.runners.run_blis import BLISRunner, build_blis_args


def test_build_blis_args():
    config = {
        "tp": 2, "replicas": 4, "max_num_seqs": 128,
        "max_batched_tokens": 4096, "chunked_prefill_threshold": 1024,
        "block_size": 16, "scheduler": "priority-fcfs",
        "admission_policy": "tier-shed", "preemption_policy": "priority",
        "routing_policy": "weighted",
        "routing_scorers": "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=50.0)
    assert "--model" in args
    assert "--tp" in args
    assert "--num-instances" in args
    assert "--rate" in args
    assert "50" in args or "50.0" in args
    assert "--routing-scorers" in args


def test_build_blis_args_single_replica():
    config = {
        "tp": 4, "replicas": 1, "max_num_seqs": 256,
        "max_batched_tokens": 8192, "chunked_prefill_threshold": 0,
        "block_size": 32, "scheduler": "fcfs",
        "admission_policy": None, "preemption_policy": "fcfs",
        "routing_policy": None, "routing_scorers": None,
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=10.0)
    assert "--routing-policy" not in args
    assert "--admission-policy" not in args


def test_build_blis_args_disabled_chunked_prefill():
    config = {
        "tp": 1, "replicas": 1, "max_num_seqs": 64,
        "max_batched_tokens": 2048, "chunked_prefill_threshold": 0,
        "block_size": 16, "scheduler": "fcfs",
        "admission_policy": None, "preemption_policy": "fcfs",
        "routing_policy": None, "routing_scorers": None,
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=10.0)
    assert "--long-prefill-token-threshold" in args
    idx = args.index("--long-prefill-token-threshold")
    assert args[idx + 1] == "0"
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement run_blis.py**

```python
# experiments/runners/run_blis.py

import json
import logging
import subprocess
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


def build_blis_args(
    config: dict,
    model: str,
    rate: float,
    workload_spec: str | None = None,
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
        "--metrics-path", "/dev/stdout",
    ]

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
    timeout_seconds = 30

    def __init__(
        self,
        workload: WorkloadInfo,
        output_path: Path,
        blis_binary: str = "./estimators/inference-sim/blis",
        workload_spec: str | None = None,
    ):
        super().__init__(workload, output_path)
        self.blis_binary = blis_binary
        self.workload_spec = workload_spec

    def _run_single(self, config: dict, rate: float) -> dict | None:
        args = build_blis_args(
            config, model=self.workload.model, rate=rate,
            workload_spec=self.workload_spec,
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
                logger.warning(f"blis run failed at rate={rate}: {result.stderr[:200]}")
                return None
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning(f"blis run timed out at rate={rate}")
            return None
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse blis output at rate={rate}")
            return None

    def evaluate_config(self, config: dict) -> ConfigResult:
        start = time.monotonic()

        # Binary search for max rate at SLO
        lo, hi = 0.0, INITIAL_RATE
        best_metrics = None
        best_rate = 0.0
        probes = 0

        # Phase 1: exponential ramp to find upper bound
        while True:
            metrics = self._run_single(config, hi)
            probes += 1
            if metrics is None or metrics.get("ttft_mean_ms", 0) > SLO_TTFT_MEAN_MS:
                break
            best_metrics = metrics
            best_rate = hi
            lo = hi
            hi *= 2

        # Phase 2: bisection
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/runners/test_run_blis.py -v`

- [ ] **Step 5: Commit**

```bash
git add experiments/runners/run_blis.py tests/runners/test_run_blis.py
git commit -m "feat: add inference-sim runner with binary rate search

Exponential ramp + bisection over --rate to find max throughput at
mean TTFT < 300ms. Builds CLI args from config dict, handles timeout
and parse errors."
```

---

### Task 14: LLMServingSim Runner (stub)

**Files:**
- Create: `experiments/runners/run_llmservingsim.py`
- Test: `tests/runners/test_run_llmservingsim.py`

Similar binary search pattern. Generates cluster config JSON per topology, invokes LLMServingSim at increasing arrival rates.

- [ ] **Step 1: Write test for CLI arg construction**

```python
# tests/runners/test_run_llmservingsim.py

from experiments.runners.run_llmservingsim import build_llmservingsim_args


def test_build_args_basic():
    config = {
        "tp": 2, "pp": 1, "replicas": 2, "max_num_seqs": 128,
        "max_batched_tokens": 4096, "enable_chunked_prefill": True,
        "chunked_prefill_threshold": 1024, "block_size": 16,
        "prefix_caching": False, "routing_policy": "LOAD",
    }
    args = build_llmservingsim_args(config, dataset_path="trace.jsonl")
    assert "--enable-chunked-prefill" in args
    assert "--long-prefill-token-threshold" in args


def test_build_args_no_chunked_prefill():
    config = {
        "tp": 1, "pp": 1, "replicas": 1, "max_num_seqs": 64,
        "max_batched_tokens": 2048, "enable_chunked_prefill": False,
        "chunked_prefill_threshold": None, "block_size": 16,
        "prefix_caching": True, "routing_policy": None,
    }
    args = build_llmservingsim_args(config, dataset_path="trace.jsonl")
    assert "--enable-chunked-prefill" not in args
```

- [ ] **Step 2: Implement run_llmservingsim.py**

```python
# experiments/runners/run_llmservingsim.py

import logging
from pathlib import Path

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, VllmArgs, RoutingConfig, Metadata, Results,
    WorkloadInfo, compute_config_hash,
)

logger = logging.getLogger(__name__)


def build_llmservingsim_args(config: dict, dataset_path: str) -> list[str]:
    args = [
        "--dataset", dataset_path,
        "--num-instances", str(config["replicas"]),
        "--tp-size", str(config["tp"]),
        "--pp-size", str(config["pp"]),
        "--max-batch-size", str(config["max_num_seqs"]),
        "--max-tokens-in-batch", str(config["max_batched_tokens"]),
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
        args.extend(["--load-balancing-policy", config["routing_policy"]])
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
        # Placeholder: actual implementation invokes LLMServingSim binary
        # with binary search over arrival rate, similar to BLISRunner
        raise NotImplementedError(
            "LLMServingSim runner requires LLMServingSim installation. "
            "See estimators/LLMSERVINGSIM.md for setup."
        )
```

- [ ] **Step 3: Run tests, commit**

Run: `python -m pytest tests/runners/test_run_llmservingsim.py -v`

```bash
git add experiments/runners/run_llmservingsim.py tests/runners/test_run_llmservingsim.py
git commit -m "feat: add LLMServingSim runner with CLI arg builder

Builds LLMServingSim CLI arguments from config dict. evaluate_config
is a stub pending LLMServingSim installation."
```

---

### Task 15: AIConfigurator Runner (stub)

**Files:**
- Create: `experiments/runners/run_aiconfigurator.py`
- Test: `tests/runners/test_run_aiconfigurator.py`

Invokes `cli estimate` per topology triple. AIConfigurator's internal sweep handles batch sizes. Linear throughput scaling for replicas.

- [ ] **Step 1: Write test and implement**

```python
# tests/runners/test_run_aiconfigurator.py

from experiments.runners.run_aiconfigurator import build_aiconfigurator_args


def test_build_args():
    config = {"tp": 2, "pp": 1, "replicas": 4}
    args = build_aiconfigurator_args(
        config, model="meta-llama/Llama-3.1-8B", isl=512, osl=256,
    )
    assert "--model-path" in args or "--model" in args
    assert "--tp-size" in args
    assert "--pp-size" in args
```

```python
# experiments/runners/run_aiconfigurator.py

import logging
from pathlib import Path

from experiments.runners.base import BaseRunner
from experiments.schema.output import ConfigResult, WorkloadInfo

logger = logging.getLogger(__name__)


def build_aiconfigurator_args(
    config: dict, model: str, isl: int = 512, osl: int = 256,
) -> list[str]:
    return [
        "estimate",
        "--model-path", model,
        "--system", "SILICON",
        "--tp-size", str(config["tp"]),
        "--pp-size", str(config["pp"]),
        "--isl", str(isl),
        "--osl", str(osl),
        "--mode", "agg",
        "--backend", "vllm",
    ]


class AIConfiguratorRunner(BaseRunner):
    tool_name = "AIConfigurator"
    timeout_seconds = 30

    def __init__(self, workload: WorkloadInfo, output_path: Path):
        super().__init__(workload, output_path)

    def evaluate_config(self, config: dict) -> ConfigResult:
        raise NotImplementedError(
            "AIConfigurator runner requires aiconfigurator installation. "
            "See estimators/AICONFIGURATOR.md for setup."
        )
```

- [ ] **Step 2: Run tests, commit**

```bash
git add experiments/runners/run_aiconfigurator.py tests/runners/test_run_aiconfigurator.py
git commit -m "feat: add AIConfigurator runner with CLI arg builder

Builds cli estimate arguments. evaluate_config is a stub pending
aiconfigurator installation."
```

---

### Task 16: Vidur Runner (stub)

**Files:**
- Create: `experiments/runners/run_vidur.py`
- Test: `tests/runners/test_run_vidur.py`

Invokes Vidur's native `config_optimizer`. Post-processes cached results to extract TTFT.

- [ ] **Step 1: Write test and implement**

```python
# tests/runners/test_run_vidur.py

from experiments.runners.run_vidur import build_vidur_config_yaml


def test_build_config_yaml():
    config = {
        "tp": 2, "pp": 1, "replicas": 2, "max_num_seqs": 128,
        "scheduler_type": "vllm", "max_tokens_in_batch": 4096,
        "chunk_size": None, "block_size": 16, "routing": "round_robin",
    }
    yaml_str = build_vidur_config_yaml(config, model="meta-llama/Llama-3.1-8B")
    assert "tensor_parallel_size" in yaml_str
    assert "vllm" in yaml_str
```

```python
# experiments/runners/run_vidur.py

import logging
from pathlib import Path

import yaml

from experiments.runners.base import BaseRunner
from experiments.schema.output import ConfigResult, WorkloadInfo

logger = logging.getLogger(__name__)


def build_vidur_config_yaml(config: dict, model: str) -> str:
    vidur_config = {
        "cluster_config": {
            "num_replicas": config["replicas"],
            "replica_config": {
                "model_name": model,
                "tensor_parallel_size": config["tp"],
                "pipeline_parallel_size": config["pp"],
                "device": "h100",
                "network_device": "a100_pairwise_nvlink",
            },
            "replica_scheduler_config": {
                "type": config["scheduler_type"],
                "batch_size_cap": config["max_num_seqs"],
                "block_size": config["block_size"],
            },
        },
    }

    sched = vidur_config["cluster_config"]["replica_scheduler_config"]
    if config["scheduler_type"] == "vllm" and config.get("max_tokens_in_batch"):
        sched["max_tokens_in_batch"] = config["max_tokens_in_batch"]
    if config["scheduler_type"] == "sarathi" and config.get("chunk_size"):
        sched["chunk_size"] = config["chunk_size"]

    if config["replicas"] > 1 and config.get("routing"):
        vidur_config["cluster_config"]["global_scheduler_config"] = {
            "type": config["routing"],
        }

    return yaml.dump(vidur_config, default_flow_style=False)


class VidurRunner(BaseRunner):
    tool_name = "Vidur"
    timeout_seconds = 600

    def __init__(self, workload: WorkloadInfo, output_path: Path):
        super().__init__(workload, output_path)

    def evaluate_config(self, config: dict) -> ConfigResult:
        raise NotImplementedError(
            "Vidur runner requires Vidur installation. "
            "See estimators/VIDUR.md for setup."
        )
```

- [ ] **Step 2: Run tests, commit**

```bash
git add experiments/runners/run_vidur.py tests/runners/test_run_vidur.py
git commit -m "feat: add Vidur runner with config YAML builder

Builds Vidur config_optimizer YAML from config dict. evaluate_config
is a stub pending Vidur installation."
```

---

### Task 17: llm-optimizer Runner (stub)

**Files:**
- Create: `experiments/runners/run_llm_optimizer.py`
- Test: `tests/runners/test_run_llm_optimizer.py`

Uses native grid search. Builds `--server-args` and `--client-args` from config, invokes `llm-optimizer` with `--constraints` and `--continue`.

- [ ] **Step 1: Write test and implement**

```python
# tests/runners/test_run_llm_optimizer.py

from experiments.runners.run_llm_optimizer import build_llm_optimizer_cmd


def test_build_cmd():
    cmd = build_llm_optimizer_cmd(
        model="meta-llama/Llama-3.1-8B",
        output_json="results/raw/llm_optimizer.json",
    )
    assert "llm-optimizer" in cmd[0] or "llm_optimizer" in " ".join(cmd)
    assert "--framework" in cmd
    assert "--constraints" in cmd
    assert "--continue" in cmd
```

```python
# experiments/runners/run_llm_optimizer.py

import logging
from pathlib import Path

from experiments.config.llm_optimizer_configs import build_grid_search_args
from experiments.runners.base import BaseRunner
from experiments.schema.output import ConfigResult, WorkloadInfo

logger = logging.getLogger(__name__)


def build_llm_optimizer_cmd(
    model: str,
    output_json: str = "results/raw/llm_optimizer.json",
    constraints: str = "ttft:mean<300ms",
    dataset_path: str | None = None,
) -> list[str]:
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
    timeout_seconds = 600

    def __init__(self, workload: WorkloadInfo, output_path: Path):
        super().__init__(workload, output_path)

    def evaluate_config(self, config: dict) -> ConfigResult:
        raise NotImplementedError(
            "llm-optimizer runner uses native grid search. "
            "Use build_llm_optimizer_cmd() to generate the CLI invocation, "
            "then post-process results. See estimators/LLM-OPTIMIZER.md."
        )
```

- [ ] **Step 2: Run tests, commit**

```bash
git add experiments/runners/run_llm_optimizer.py tests/runners/test_run_llm_optimizer.py
git commit -m "feat: add llm-optimizer runner with native grid search CLI builder

Builds llm-optimizer command using native --server-args grid syntax,
--constraints for SLO filtering, and --continue for checkpointing."
```

---

## Chunk 5: Integration and Execution Entry Points

### Task 18: Main Orchestration Script

**Files:**
- Create: `experiments/run_all.py`

Top-level script that ties together config generation, workload conversion, and runner invocation. Serves as the entry point for the full experiment.

- [ ] **Step 1: Implement run_all.py**

```python
# experiments/run_all.py

"""
Config Exploration Experiment - Main Orchestration

Usage:
    python -m experiments.run_all --phase setup     # Build tools, generate workloads
    python -m experiments.run_all --phase estimate   # Run all 5 tool sweeps
    python -m experiments.run_all --phase analyze    # Compute Pareto fronts, select top-3
    python -m experiments.run_all --tool blis        # Run single tool only

See docs/superpowers/specs/2026-05-07-config-exploration-experiment-design.md
"""

import argparse
import logging
from pathlib import Path

from experiments.schema.output import WorkloadInfo

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")
WORKLOADS_DIR = Path("experiments/workloads")

DEFAULT_WORKLOAD = WorkloadInfo(
    model="meta-llama/Llama-3.1-8B",
    hardware="H100_SXM_80GB",
    preset="servegen_m-mid",
    num_requests=10000,
    isl_mean=512,
    isl_max=2048,
    osl_mean=256,
    osl_max=1024,
    arrival_pattern="poisson",
    slo_ttft_mean_ms=300,
    seed=42,
)

TOOLS = ("blis", "llmservingsim", "aiconfigurator", "vidur", "llm-optimizer")


def phase_setup():
    logger.info("Phase: Setup")
    logger.info("1. Verify submodules: git submodule update --init --recursive")
    logger.info("2. Build inference-sim: cd estimators/inference-sim && go build -o blis main.go")
    logger.info("3. Install AIConfigurator: pip install -r requirements.txt")
    logger.info("4. Generate canonical workload")

    from experiments.workloads.generate import generate_canonical_trace
    WORKLOADS_DIR.mkdir(parents=True, exist_ok=True)
    canonical = WORKLOADS_DIR / "canonical_servegen_m-mid.yaml"
    if not canonical.exists():
        generate_canonical_trace(canonical)
        logger.info(f"Generated canonical trace: {canonical}")
    else:
        logger.info(f"Canonical trace exists: {canonical}")


def phase_estimate(tool: str | None = None):
    logger.info(f"Phase: Estimate (tool={tool or 'all'})")
    tools = [tool] if tool else TOOLS
    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for t in tools:
        output = raw_dir / f"{t}.jsonl"
        logger.info(f"Running {t} -> {output}")
        if t == "blis":
            from experiments.config.blis_configs import generate_blis_configs
            from experiments.runners.run_blis import BLISRunner
            configs = generate_blis_configs()
            runner = BLISRunner(
                workload=DEFAULT_WORKLOAD,
                output_path=output,
                workload_spec=str(WORKLOADS_DIR / "canonical_servegen_m-mid.yaml"),
            )
            runner.run_batch(configs, hash_fn=lambda c: str(hash(frozenset(c.items()))))
        elif t == "llm-optimizer":
            from experiments.runners.run_llm_optimizer import build_llm_optimizer_cmd
            cmd = build_llm_optimizer_cmd(
                model=DEFAULT_WORKLOAD.model,
                output_json=str(raw_dir / "llm_optimizer.json"),
            )
            logger.info(f"llm-optimizer CLI: {' '.join(cmd)}")
            logger.info("Run this command on a GPU node with --dry-run first to verify config count.")
        else:
            logger.warning(f"Runner for {t} not yet implemented. See experiments/runners/")


def phase_analyze():
    logger.info("Phase: Analyze")
    from experiments.analysis.pareto import compute_pareto_front
    from experiments.analysis.select import select_top_k

    raw_dir = RESULTS_DIR / "raw"
    processed_dir = RESULTS_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    import json
    for tool in TOOLS:
        results_file = raw_dir / f"{tool}.jsonl"
        if not results_file.exists():
            logger.warning(f"No results for {tool}, skipping")
            continue

        results = []
        for line in results_file.read_text().strip().split("\n"):
            if line:
                data = json.loads(line)
                if data.get("results") and data.get("metadata", {}).get("status") == "ok":
                    results.append(data["results"])

        front = compute_pareto_front(results)
        top3 = select_top_k(front, k=3, min_throughput=200)
        logger.info(f"{tool}: {len(results)} results, {len(front)} Pareto, {len(top3)} top-3")

        output = processed_dir / f"{tool}_top3.json"
        output.write_text(json.dumps(top3, indent=2))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Config Exploration Experiment")
    parser.add_argument("--phase", choices=["setup", "estimate", "analyze"], required=True)
    parser.add_argument("--tool", choices=TOOLS, help="Run single tool only (estimate phase)")
    args = parser.parse_args()

    if args.phase == "setup":
        phase_setup()
    elif args.phase == "estimate":
        phase_estimate(args.tool)
    elif args.phase == "analyze":
        phase_analyze()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

Run: `python -m experiments.run_all --phase setup --help` (should show usage)

- [ ] **Step 3: Commit**

```bash
git add experiments/run_all.py
git commit -m "feat: add main orchestration script

Entry point for setup, estimate, and analyze phases.
Wires config generation, workload conversion, and runners together."
```

---

### Task 19: pyproject.toml and Test Configuration

**Files:**
- Create: `pyproject.toml`

Project configuration with dependencies and pytest settings.

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "sim2real-config-exploration"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7",
    "numpy>=1.24",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py310"
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pyproject.toml with dependencies and test config"
```

---

### Task 20: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

End-to-end test: generate configs for all tools, verify counts, verify schema compliance for a sample config.

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py

from experiments.config.blis_configs import generate_blis_configs
from experiments.config.llmservingsim_configs import generate_llmservingsim_configs
from experiments.config.aiconfigurator_configs import generate_aiconfigurator_configs
from experiments.config.vidur_configs import generate_vidur_configs
from experiments.config.llm_optimizer_configs import generate_llm_optimizer_configs


def test_all_tools_generate_configs():
    blis = generate_blis_configs()
    llmsim = generate_llmservingsim_configs()
    aicfg = generate_aiconfigurator_configs()
    vidur = generate_vidur_configs()
    llmopt = generate_llm_optimizer_configs()

    assert len(blis) > 0
    assert len(llmsim) > 0
    assert len(aicfg) > 0
    assert len(vidur) > 0
    assert len(llmopt) > 0


def test_config_counts_summary():
    """Verify input config counts match spec topology/parameter expectations.

    Note: AIConfigurator returns 25 input topology triples; the ~250 output
    points come from its internal batch/ctx_tokens sweep at runtime.
    """
    counts = {
        "inference-sim": len(generate_blis_configs()),
        "LLMServingSim": len(generate_llmservingsim_configs()),
        "AIConfigurator": len(generate_aiconfigurator_configs()),
        "Vidur": len(generate_vidur_configs()),
        "llm-optimizer": len(generate_llm_optimizer_configs()),
    }
    assert 20_000 <= counts["inference-sim"] <= 30_000
    assert 4_000 <= counts["LLMServingSim"] <= 5_500
    assert counts["AIConfigurator"] == 25  # input triples; ~250 output points at runtime
    assert 3_000 <= counts["Vidur"] <= 4_500
    assert 10_000 <= counts["llm-optimizer"] <= 13_000
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for all tool config generation

Verifies all 5 tools generate configs and counts match spec ranges."
```
