# Config Exploration Experiment Design

## Overview

Experiment implementation for the Config Exploration section of the BLIS paper. Five LLM serving estimators search a shared config space, and their recommendations are validated against real llm-d deployments (vLLM on H100) to measure sim2real drift and SLO compliance.

Paper plan: https://github.com/inference-sim/inference-sim/discussions/1237

## Models

Model selection aligned with BLIS accuracy experiments (discussion #1257):

| Chart | Model(s) | Rationale |
|-------|----------|-----------|
| Charts 1-2 (comparative) | meta-llama/Llama-3.1-8B | Primary baseline in accuracy experiments; Vidur supports Llama-3-8B; fits on 1 GPU (TP=1); all tools can evaluate it |
| Charts 3-4 (BLIS-only) | Qwen3-14B | BLIS uses analytical roofline (no profiling); modern model strengthens narrative |
| Chart 5 (model selection) | meta-llama/Llama-3.1-8B, CodeLlama-34B-Instruct, Llama-3.1-70B | Spans TP=1 to TP=4; CodeLlama-34B requires min TP=2, 70B requires min TP=4 |

## Workload

Aligned with BLIS accuracy experiments: `servegen m-mid` as primary workload.

| Parameter | Value | Source |
|-----------|-------|--------|
| Workload type | servegen m-mid | Accuracy experiments baseline; production-representative, no multi-turn queueing confounds |
| ISL distribution | Mean ~512, max 2048 | Typical chat distribution |
| OSL distribution | Mean ~256, max 1024 | Typical chat response |
| SLO | Mean TTFT < 300ms | Paper plan |
| Arrival pattern | Poisson | Standard open-loop benchmark |

### Canonical Workload Generation

To ensure all tools evaluate the same workload, we generate a single canonical trace from the `servegen m-mid` preset and convert it to each tool's native format:

1. Generate canonical trace: `blis convert preset --preset servegen --variant m-mid` produces 10,000 requests with ISL/OSL distributions matching production chat traffic. Poisson arrivals at parameterized rate. Fixed seed for reproducibility.
2. Convert to each tool's format:
   - inference-sim: v2 WorkloadSpec YAML (native; `--workload servegen --workload-variant m-mid`)
   - LLMServingSim: JSONL with input_toks, output_toks, arrival_time fields
   - Vidur: CSV trace (request_id, arrival_time, prefill_tokens, decode_tokens)
   - llm-optimizer: sharegpt-format JSON with pre-tokenized lengths
   - AIConfigurator: pass mean ISL/OSL (`--isl 512 --osl 256`); no trace needed (analytical)

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | H100 SXM 80GB |
| Cost | $3.20/hr per GPU (consistent pricing across tools) |
| Validation cluster | Manual deployment on H100 cluster |
| Deployment platform | llm-d (vLLM serving engine, llm-d-inference-scheduler for routing) |

## Config Space Design

### Philosophy

The experiment has two phases:

**Phase A (apples-to-apples):** All tools search the same shared config space. This isolates prediction accuracy: given identical deployment configs, which tool most accurately predicts real-world performance?

**Phase B (tool-native expressiveness):** Each tool additionally explores parameters it uniquely models. This demonstrates expressiveness: BLIS finds better configs because it CAN explore admission control, priority scheduling, and routing. Other tools physically cannot evaluate these dimensions.

Both phases are presented in the paper. Phase A produces Chart 1's scatter plot with drift arrows (identical configs, different predictions). Phase B shows how BLIS's richer space shifts its Pareto front left (cheaper) relative to other tools.

### Phase A: Shared Base (all 5 tools)

Hardware constraint: **up to 8 GPUs** (single node, H100 SXM)

| Parameter | Values | Count |
|-----------|--------|-------|
| TP (tensor parallelism) | {1, 2, 4, 8} | 4 |
| PP (pipeline parallelism) | {1, 2, 4} | 3 |
| Deployment replicas | {1, 2, ..., floor(8 / (TP * PP))} | varies |
| Max batch size / max_num_seqs | {32, 64, 128, 256, 512} | 5 |

Validity constraint: `TP * PP * deployment_replicas <= 8`

"Deployment replicas" means independent Kubernetes-style vLLM pods, each with its own KV cache, scheduler, and request queue. An external scheduler (llm-d inference scheduler) routes requests across pods. This is distinct from vLLM's internal `--data-parallel-size`, which spawns multiple worker groups within a single pod (see note on llm-optimizer below).

Deployment replicas is an independent sweep dimension, not derived from the GPU budget. Each replica uses `TP * PP` GPUs, so total GPU usage is `TP * PP * deployment_replicas` (may be less than 8). Cost scales with actual GPU usage: `cost_per_hour = TP * PP * deployment_replicas * $3.20`.

This makes routing a first-class variable: holding (TP, PP) constant, more replicas with intelligent routing may outperform fewer replicas at the same or lower cost. This is where BLIS's routing expressiveness (prefix-aware weighted routing, admission control) delivers measurable value.

The implementation enumerates all valid (TP, PP, deployment_replicas) triples satisfying the constraint. For PP-supporting tools:
- TP=1, PP=1: deployment_replicas in {1..8} (8 values)
- TP=1, PP=2: deployment_replicas in {1..4} (4 values)
- TP=1, PP=4: deployment_replicas in {1..2} (2 values)
- TP=2, PP=1: deployment_replicas in {1..4} (4 values)
- TP=2, PP=2: deployment_replicas in {1..2} (2 values)
- TP=2, PP=4: deployment_replicas in {1} (1 value)
- TP=4, PP=1: deployment_replicas in {1..2} (2 values)
- TP=4, PP=2: deployment_replicas in {1} (1 value)
- TP=8, PP=1: deployment_replicas in {1} (1 value)
- Total: 25 valid topology triples (9 single-replica, 16 multi-replica)

llm-optimizer does not model deployment replicas (always 1 pod). Instead, it sweeps vLLM's `--data-parallel-size` within that single pod: `TP * PP * data_parallel_size <= 8`. This yields the same 25 valid (TP, PP, DP) triples, but the parallelism is vLLM-internal (shared process, vLLM-managed routing) rather than Kubernetes-level (separate pods, llm-d routing). For validation, `data_parallel_size` maps directly to vLLM's `--data-parallel-size` flag, which is a real, deployable vLLM configuration. Routing between DP groups is handled internally by vLLM (not by llm-d), which differs architecturally from the external routing used by other tools' multi-replica configs. This difference is inherent to the deployment topology, not a limitation of the experiment.

**Base config count: 25 topology triples x 5 batch sizes = 125 configs** (llm-optimizer: 25 x 5 = 125, using DP instead of deployment replicas)

### Phase B: Per-Tool Parameter Sweep

Each tool sweeps additional parameters beyond the Phase A shared base. Design principles:

- Tools with native config search (Vidur, llm-optimizer) use their native search mechanisms. We post-process their output to extract TTFT and apply our SLO filter.
- Tools without native config search (inference-sim, LLMServingSim) are invoked as single-point evaluators by our external sweep harness with binary search over rate.
- AIConfigurator is invoked via `cli estimate` per config (analytical, returns prediction directly).
- Vidur's native `config_optimizer` runs binary search over QPS using scheduling delay as its internal stopping criterion. The simulation output includes TTFT (`prefill_e2e_time`); we post-process cached results to find max QPS where TTFT < 300ms.
- llm-optimizer's native grid search benchmarks each config independently. We use `--constraints "ttft:mean<300ms"` for SLO filtering.

Source of truth for parameter availability: `estimators/COMPARISON.md`

#### Unified Parameter Sweep Table

Design principle: only sweep knobs that affect serving performance (throughput, latency, cost) without changing model output quality. Precision-reducing options (fp8 weights, fp8 KV cache, quantization) are excluded because they alter generation quality, confounding the sim2real accuracy comparison.

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|-----------|--------------|---------------|----------------|-------|---------------|
| TP | {1, 2, 4, 8} | {1, 2, 4, 8} | {1, 2, 4, 8} | {1, 2, 4, 8} | {1, 2, 4, 8} |
| PP | N/A | {1, 2, 4} | {1, 2, 4} | {1, 2, 4} | {1, 2, 4} |
| Deployment replicas | {1..floor(8/TP)} | {1..floor(8/(TP*PP))} | {1..floor(8/(TP*PP))} | {1..floor(8/(TP*PP))} | N/A (always 1) |
| vLLM data parallel size | N/A | N/A | N/A | N/A | {1..floor(8/(TP*PP))} |
| Scheduler type | {fcfs, priority-fcfs, sjf, reverse-priority} | N/A (mirrors vLLM) | N/A (analytical) | {vllm, sarathi, orca} | N/A |
| Max batch size / max_num_seqs | {32, 64, 128, 256, 512} | {32, 64, 128, 256, 512} | internal sweep (~80-100 values) | {32, 64, 128, 256, 512} | {32, 64, 128, 256, 512} |
| Max batched tokens | {2048, 4096, 8192} | {2048, 4096, 8192} | N/A | {2048, 4096, 8192} (vllm sched only) | {2048, 4096, 8192} |
| Enable chunked prefill | {true, false} | {true, false} | N/A | N/A (use sarathi scheduler) | {true, false} |
| Chunked prefill threshold | {1024, 2048, 4096} (when enabled) | {0 (uncapped), 1024, 2048, 4096} (when enabled) | N/A | {1024, 2048, 4096} (sarathi chunk_size) | N/A (vLLM uses max_batched_tokens as cap) |
| Block size | {16, 32} | {16, 32} | N/A | {16, 32} | {16, 32} |
| Routing configuration | {round-robin, least-loaded} + 4 weighted scorer configs (see notes) | {LOAD, RR, RAND} | N/A | {round_robin, lor, random} | N/A |
| Admission policy | {always-admit, tier-shed} | N/A | N/A | N/A | N/A |
| Preemption policy | {fcfs, priority} | N/A | N/A | N/A | N/A |
| Prefix caching | N/A (always active for routing; see notes) | {enabled, disabled} | N/A | N/A | {enabled, disabled} |
| Max concurrency | N/A | N/A | N/A | N/A | {32, 64, 128, 256} |

**Fixed parameters (not swept):**

| Parameter | Value | Applies to | Rationale |
|-----------|-------|-----------|-----------|
| Latency model | trained-physics | inference-sim | Best accuracy mode |
| Max GPUs | 8 (single node H100 SXM) | All | Hardware constraint; actual usage = TP * PP * deployment_replicas (or TP * PP * data_parallel_size for llm-optimizer) |
| GPU memory utilization | 0.9 | inference-sim, llm-optimizer | Standard default; controls KV cache capacity |
| Model dtype | bfloat16 | All | Quality-neutral baseline; fp8 excluded to avoid quality confounds |
| KV cache dtype | auto (bfloat16) | LLMServingSim, llm-optimizer | Quality-neutral; fp8 KV excluded |
| Database mode | SILICON | AIConfigurator | Hardware-specific predictions |
| Estimate mode | agg | AIConfigurator | Aggregated prefill+decode (not disaggregated) |
| Backend / framework | vllm | AIConfigurator, llm-optimizer | Match validation target |
| Network backend | analytical | LLMServingSim | Only complete backend; ns3 is WIP |
| Chunked prefill (AIConfigurator) | disabled | AIConfigurator | enable_chunked_prefill=false for ctx_tokens sweep |
| Prefill priority | disabled | LLMServingSim | No vLLM equivalent; simulation-only scheduling policy |
| Prefix sharing | disabled | LLMServingSim | No vLLM/llm-d equivalent; models hypothetical cross-instance KV cache pooling |
| Prefix storage | None | LLMServingSim | No vLLM equivalent for CXL; CPU mode differs from vLLM swap semantics |

Notes:
- Up to 8 GPUs total. Validity: `TP * PP * deployment_replicas <= 8` (or `TP * PP * data_parallel_size <= 8` for llm-optimizer). Cost: `TP * PP * (deployment_replicas or data_parallel_size) * $3.20/hr`. Configs may use fewer than 8 GPUs; the Pareto analysis accounts for variable cost.
- Only inference-sim lacks PP support. PP is fixed at 1 for inference-sim, giving 4 valid TP values: {1, 2, 4, 8}. For each TP, deployment_replicas ranges from 1 to floor(8/TP): TP=1 gets {1..8}, TP=2 gets {1..4}, TP=4 gets {1..2}, TP=8 gets {1}. Total: 15 topology triples (4 single-replica, 11 multi-replica).
- For PP-supporting tools, 9 valid (TP, PP) pairs (where TP * PP <= 8) expand to 25 topology triples with independent deployment_replicas (see Phase A table). 9 are single-replica; 16 are multi-replica. llm-optimizer uses the same 25 topology triples but with `data_parallel_size` instead of deployment replicas (vLLM-internal DP within a single pod, not Kubernetes replicas with external routing).
- Chunked prefill implementation varies by tool:
  - inference-sim: `--long-prefill-token-threshold` (0=disabled, >0=chunk size)
  - LLMServingSim: `--enable-chunked-prefill` (boolean) + `--long-prefill-token-threshold` (per-request cap). The 0 value means "uncapped" (not disabled): the scheduler guard is `if 0 < threshold < remaining` (`scheduler.py` line 134), so threshold=0 never triggers the cap. A single prefill can consume the entire `max_num_batched_tokens` budget after decode requests take their 1-token-each share. Values >0 impose a per-request cap, enabling better interleaving of multiple prefills. This differs from inference-sim where 0 means disabled.
  - Vidur: sarathi scheduler has explicit `chunk_size`; vllm/orca schedulers do not chunk
  - llm-optimizer: `enable_chunked_prefill=True` uses `max_num_batched_tokens` as the effective chunk cap (no separate chunk size param in vLLM)
- Max batched tokens applies to vllm scheduler in Vidur (not sarathi/orca)
- Routing/admission policies only apply to multi-replica configs (deployment_replicas > 1). Routing is a single concept per tool; some tools offer composable weighted scoring while others provide fixed-formula policies:
  - inference-sim: 2 simple policies (`round-robin`, `least-loaded`) + composable weighted routing via `--routing-scorers` (comma-separated `scorer:weight` pairs; all scorers output [0, 1], weighted router picks argmax). This is a key BLIS differentiator; no other tool can model weighted multi-signal routing. 4 weighted scorer configs are swept:
    - `precise-prefix-cache:2,queue-depth:1,kv-utilization:1` (llm-d production default: prefix-aware with load/memory tiebreaking)
    - `queue-depth:1,kv-utilization:1` (load-only: no prefix awareness, pure load balancing on queue pressure and memory)
    - `precise-prefix-cache:2,load-balance:1` (prefix + pile-on mitigation: prefix-aware with composite load signal that includes in-flight requests)
    - `vllm-dp:1` (vLLM parity: exact vLLM data-parallel routing formula, QueueDepth*4 + BatchSize; use alone)
  - LLMServingSim: 3 fixed-formula policies. `LOAD` uses `waiting*4 + running` (not configurable weights). `RR` is round-robin. `RAND` is random. No composable scoring.
  - Vidur: `--global_scheduler_config_type` supports `round_robin` (default), `lor` (least-outstanding-requests), and `random`. Only applies to multi-replica configs (num_replicas > 1).
  - AIConfigurator: no routing model; assumes linear throughput scaling with replicas.
  - llm-optimizer: no external routing model; always deployment_replicas=1. Sweeps `data_parallel_size` for vLLM-internal DP, but routing between DP groups is handled internally by vLLM and not configurable at the policy level.
- Available but not swept inference-sim scorers: `prefix-affinity` (cheaper router-side prefix approximation), `no-hit-lru` (cold request spreading), `active-requests` (in-flight only), `running-requests` (batch size only), `load-aware` (threshold-capped, scores in [0, 0.5] sub-range). These are either subsumed by the swept combinations or too narrow to justify additional configs.
- AIConfigurator internally sweeps ~80-100 batch sizes and a ctx_tokens dimension per TP/PP pair (~4,000 internal evaluations), but outputs only the Pareto-optimal configs (~70 points).
- AIConfigurator backend fixed to vllm to match validation target
- Vidur uses native `config_optimizer` with scheduling-delay SLO; TTFT extracted from cached simulation output in post-processing
- llm-optimizer uses native grid search with `--constraints` for direct TTFT filtering
- inference-sim prefix caching: marked N/A in the sweep table because there is no boolean toggle equivalent to vLLM's `--enable-prefix-caching`. However, inference-sim always simulates KV cache state including prefix hit tracking (`sim/prefix_cache_index.go`). Prefix-cache-aware routing is configured via weighted routing scorers (`precise-prefix-cache`, `prefix-affinity`, `no-hit-lru`), which are already swept in the routing configuration. The cache behavior is always active; only the routing scorer decides whether prefix hits influence instance selection.
- Excluded quality-affecting knobs: model dtype fp8, KV cache dtype fp8, quantization (awq/gptq/squeezellm). These reduce precision and alter model outputs, confounding the sim2real accuracy comparison.

#### Config Count Math

The implementation enumerates all (TP, PP, deployment_replicas) triples satisfying `TP * PP * deployment_replicas <= 8`, then cross-products with per-tool parameter dimensions. Single-replica configs (deployment_replicas=1) omit routing/admission parameters. For llm-optimizer, the topology dimension is (TP, PP, data_parallel_size) with the same constraint.

**inference-sim (no PP):**
- 15 topology triples: 4 single-replica (TP in {1,2,4,8}, R=1) + 11 multi-replica
- Batching: 5 (max_num_seqs) x 3 (max_batched_tokens) x 2 (enable_chunked_prefill) with 3 thresholds when enabled = 5 x 3 x (1 + 3) = 60 combos (pruned to ~40 after removing threshold >= max_batched_tokens)
- Multi-replica (11 triples): routing/admission apply
  - Policy: 4 (scheduler) x 2 (admission) x 2 (preemption) x 6 (routing: 2 non-weighted + 4 weighted scorer configs) x 2 (block_size) = 192
  - 11 x 40 x 192 = 84,480
- Single-replica (4 triples): no routing/admission
  - Policy: 4 (scheduler) x 2 (preemption) x 2 (block_size) = 16
  - 4 x 40 x 16 = 2,560
- Total: 84,480 + 2,560 = 87,040
- After pruning invalid combos: **~25,000 configs**
- Wall-clock at ~1s each: ~6.9 hours (CPU-only, highly parallelizable)

**LLMServingSim:**
- 25 topology triples: 9 single-replica + 16 multi-replica
- Batching: 3 (max_batched_tokens) x (1 + 4) (chunked_prefill: disabled OR 4 thresholds when enabled) = 15 combos (pruned to ~12 after removing threshold >= max_batched_tokens)
- Additional dimensions: 2 (block-size) x 2 (prefix-caching) = 4
- Single-replica (9 triples): 9 x 5 (batch) x 12 (batching) x 4 = 2,160
- Multi-replica (16 triples): 16 x 5 (batch) x 12 (batching) x 4 x 3 (routing) = 11,520
- Total: 13,680
- After pruning invalid combos: **~4,700 configs**
- Wall-clock at ~18s each: ~23.5 hours (parallelizable across CPU cores)

**AIConfigurator:**
- 25 topology triples
- Internal sweep: ~80-100 batch sizes x ctx_tokens per triple = ~12,500 internal evaluations
- Output: Pareto-optimal configs only, with cluster throughput = single_replica_rate x deployment_replicas = **~250 output points**
- Wall-clock: analytical, ~2 min total

**Vidur:**
- 25 topology triples: 9 single-replica + 16 multi-replica
- Scheduler variants: 3 (vllm with max_tokens_in_batch) + 3 (sarathi with chunk_sizes) + 1 (orca) = 7
- Block size: 2 ({16, 32})
- Multi-replica routing: 3 ({round_robin, lor, random}) for 16 multi-replica triples, 1 for 9 single-replica triples
- Single-replica: 9 x 5 (batch) x 7 (scheduler) x 2 (block_size) = 630
- Multi-replica: 16 x 5 (batch) x 7 (scheduler) x 2 (block_size) x 3 (routing) = 3,360
- Total: 3,990
- After Vidur internal validity pruning: **~3,500 configs**
- Wall-clock: ~2.5 hours (native config_optimizer with Ray parallelization)

**llm-optimizer (deployment_replicas=1, sweeps vLLM data_parallel_size):**
- 25 valid (TP, PP, DP) triples satisfying `TP * PP * data_parallel_size <= 8` (same topology count as other PP-supporting tools, but DP replaces deployment replicas)
- Chunked prefill: 2 states (disabled, enabled). When enabled, max_batched_tokens serves as chunk cap.
- 25 (TP x PP x DP) x 5 (max_num_seqs) x 3 (max_batched_tokens) x 2 (chunked_prefill) x 2 (block_size) x 2 (prefix_caching) x 4 (concurrency) = 12,000
- After pruning: **~11,200 benchmark runs** (each requires real GPU time; GPU usage = TP * PP * DP GPUs per run)
- llm-optimizer's native grid search (`--server-args` with grid syntax, `--client-args`) handles the combinatorial expansion and execution loop. Runs are sequential (one server restart per config, no parallelization). The `--continue` flag resumes from JSONL checkpoints, matching completed configs by ID. The implementation plan should express the config space using llm-optimizer's native grid search syntax (paired `key1*key2=[(a,b)]`, list `key=[v1,v2]`, and `--constraints` for SLO filtering) rather than building a custom sweep harness. Use `--dry-run` to validate the config count before committing GPU time.

### Config Space Summary

| Tool | Total configs | Wall-clock estimate | GPU required |
|------|--------------|--------------------:|:------------:|
| inference-sim | ~25,000 | ~6.9 hours (CPU-only, ~1s each, highly parallelizable) | No |
| LLMServingSim | ~4,700 | ~23.5 hours (CPU-only, cycle-level, ~18s each; parallelizable) | No* |
| AIConfigurator | ~250 (output) | ~2 min (analytical; ~12,500 internal evaluations) | No |
| Vidur | ~3,500 | ~3.5 hours (native config_optimizer with Ray parallelization) | No |
| llm-optimizer | ~11,200 | ~375 hours (real benchmarks, ~2 min each; GPU usage varies with TP*PP*DP) | Yes |

*LLMServingSim requires 1 hour of H100 profiling (decided; see Decisions Made #6) but simulation itself is CPU-only.

## Tool Limitations

Each tool has known limitations that may affect prediction quality. These are not bugs to fix; they are characteristics of each tool's modeling approach that the experiment exposes through sim2real validation.

| Tool | Limitation | Expected impact on results |
|------|-----------|---------------------------|
| **inference-sim** | Analytical roofline model may underestimate contention effects at high batch sizes | Optimistic TTFT predictions under extreme load |
| **LLMServingSim** | Cycle-level simulation depends on profiled kernel timings; prediction quality is bounded by profiling coverage of batch/sequence/TP combinations. | Configs outside profiled ranges rely on interpolation, which may introduce prediction error at extremes |
| **AIConfigurator** | Single-point analytical predictions with no distribution awareness. Cannot express "P99 TTFT < X", only mean/point estimates. | May produce false positives: predicted TTFT meets SLO but tail latency violates it under real traffic variance |
| **Vidur** | Native search optimizes for scheduling delay (time in queue), not TTFT directly. We use scheduling-delay SLO to drive the binary search, then extract TTFT from cached simulation output in post-processing. The max QPS for scheduling delay may exceed the max QPS for TTFT (since TTFT = scheduling delay + prefill execution time). | Vidur's reported max QPS may be optimistic relative to our TTFT SLO. Post-processing corrects this, but if the true TTFT-constrained max QPS falls between binary search probes, precision is limited by probe granularity. |
| **llm-optimizer** | Requires real GPU hardware for every config evaluation. Estimate mode uses roofline (no simulation of queuing or batching dynamics). | In estimate mode: ignores contention effects. In benchmark mode: measurements are accurate but expensive. |

These limitations are the experiment's core finding: they explain WHY different tools produce different recommendations and WHY some tools' recommendations fail in production.

## Evaluation Protocol

### Per-Tool: Finding Max Throughput at SLO

For each static config, each tool determines the maximum sustainable throughput where mean TTFT < 300ms:

| Tool | Method |
|------|--------|
| inference-sim | Binary search over `--rate`: start at 10, double until TTFT > 300ms, then bisect to convergence (8 iterations) |
| LLMServingSim | Generate JSONL datasets at increasing arrival rates; find highest rate where mean TTFT < 300ms |
| AIConfigurator | Single call via `cli estimate` returns predicted TTFT and throughput directly (no search needed) |
| Vidur | Native `config_optimizer`: binary search over QPS with scheduling-delay SLO. Post-process cached results to find max QPS where TTFT (`prefill_e2e_time`) < 300ms. |
| llm-optimizer | Native grid search benchmarks each config; filter by `--constraints "ttft:mean<300ms"` |

### Output Schema

Each config evaluation produces a JSON record with four config layers: `vllm_args` (per-replica vLLM parameters), `routing_config` (multi-replica routing strategy deployed via llm-d), `tool_config` (tool-specific knobs that are not deployable), and `results`/`metadata`. Fields are nullable; tools only populate what they support.

The validation harness consumes both `vllm_args` and `routing_config` to deploy on llm-d. `vllm_args` configures each vLLM instance; `routing_config` configures the llm-d inference scheduler that routes requests across instances. `tool_config` captures prediction-only knobs (internal scheduler variants, preemption policies) that explain tool behavior but are not deployed.

```json
{
  "tool": "inference-sim",
  "workload": {
    "model": "meta-llama/Llama-3.1-8B",
    "hardware": "H100_SXM_80GB",
    "preset": "servegen_m-mid",
    "num_requests": 10000,
    "isl_mean": 512,
    "isl_max": 2048,
    "osl_mean": 256,
    "osl_max": 1024,
    "arrival_pattern": "poisson",
    "slo_ttft_mean_ms": 300,
    "seed": 42,
    "trace_file": "workloads/canonical_servegen_m-mid.yaml"
  },
  "vllm_args": {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "num_instances": 4,
    "data_parallel_size": 1,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 4096,
    "enable_chunked_prefill": true,
    "block_size": 16,
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "kv_cache_dtype": "auto",
    "enable_prefix_caching": false,
    "enforce_eager": false,
    "swap_space": 4
  },
  "routing_config": {
    "strategy": "weighted-scoring",
    "scorers": "prefix-cache:2,queue-depth:1,kv-utilization:1",
    "picker": "max-score"
  },
  "tool_config": {
    "scheduler": "priority-fcfs",
    "admission_policy": "tier-shed",
    "preemption_policy": "priority",
    "max_concurrency": null,
    "vidur_scheduler_type": null
  },
  "results": {
    "max_throughput_tok_s": 850,
    "max_throughput_qps": 12.5,
    "ttft_mean_ms": 245,
    "ttft_p50_ms": 220,
    "ttft_p99_ms": 380,
    "tpot_mean_ms": 18.5,
    "meets_slo": true,
    "cost_per_hour": 25.60,
    "cost_per_1k_tokens": 0.0084
  },
  "metadata": {
    "status": "ok",
    "tool_version": "v0.4.2",
    "wall_clock_seconds": 2.3,
    "num_rate_probes": 8,
    "config_hash": "a3f8c1d2",
    "timestamp": "2026-05-11T14:30:00Z"
  }
}
```

**workload fields** (inputs to the tool, shared across all configs for a given experiment run):

| Field | Description | Source |
|-------|-------------|--------|
| `model` | HuggingFace model identifier | Experiment design (Models section) |
| `hardware` | GPU type and spec | Fixed: H100 SXM 80GB |
| `preset` | Workload preset name | Fixed: servegen_m-mid |
| `num_requests` | Total requests in canonical trace | Fixed: 10,000 |
| `isl_mean` | Mean input sequence length (tokens) | From servegen m-mid distribution |
| `isl_max` | Max input sequence length (tokens) | From servegen m-mid distribution |
| `osl_mean` | Mean output sequence length (tokens) | From servegen m-mid distribution |
| `osl_max` | Max output sequence length (tokens) | From servegen m-mid distribution |
| `arrival_pattern` | Request arrival distribution | Fixed: poisson |
| `slo_ttft_mean_ms` | SLO threshold for mean TTFT | Fixed: 300ms |
| `seed` | Random seed for reproducibility | Fixed: 42 |
| `trace_file` | Path to canonical trace file (tool-native format) | Per-tool converted trace |

Notes on workload per tool:
- inference-sim: uses v2 WorkloadSpec YAML directly (`--workload-spec`)
- LLMServingSim: uses JSONL with `input_toks`, `output_toks`, `arrival_time` per request
- Vidur: uses CSV trace with `request_id`, `arrival_time`, `prefill_tokens`, `decode_tokens`
- llm-optimizer: uses sharegpt-format JSON with pre-tokenized lengths; `trace_file` points to the converted dataset
- AIConfigurator: analytical; uses `isl_mean`/`osl_mean` directly (`--isl 512 --osl 256`), no trace file needed (`trace_file` is null)

**vllm_args fields** (used directly by validation harness to deploy vLLM):

| Field | vLLM CLI flag | Source per tool |
|-------|--------------|-----------------|
| `tensor_parallel_size` | `--tensor-parallel-size` | All tools: from TP |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | PP-supporting tools: from PP. inference-sim: always 1 |
| `num_instances` | N/A (deployment topology) | Deployment replicas for replica-aware tools (inference-sim, LLMServingSim, AIConfigurator, Vidur). llm-optimizer: always 1 (uses vLLM-internal `data_parallel_size` instead; see `vllm_args.data_parallel_size` below) |
| `max_num_seqs` | `--max-num-seqs` | All tools: from max_batch_size / max_num_seqs / batch_size_cap |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | All tools: from max_batched_tokens / max_tokens_in_batch |
| `enable_chunked_prefill` | `--enable-chunked-prefill` | All tools: from chunked prefill config (see notes) |
| `block_size` | `--block-size` | inference-sim, LLMServingSim, Vidur: from block_size. Others: default 16 |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | Fixed: 0.9 |
| `dtype` | `--dtype` | Fixed: bfloat16 |
| `kv_cache_dtype` | `--kv-cache-dtype` | Fixed: auto |
| `data_parallel_size` | `--data-parallel-size` | llm-optimizer: from vLLM data_parallel_size sweep. Others: 1 (use deployment replicas instead) |
| `enable_prefix_caching` | `--enable-prefix-caching` | LLMServingSim, llm-optimizer: from prefix_caching. Others: false |
| `enforce_eager` | `--enforce-eager` | Fixed: false (CUDA graphs enabled) |
| `swap_space` | `--swap-space` | Fixed: 4 GB |

**routing_config fields** (deployed via llm-d inference scheduler for multi-replica configs):

| Field | Description |
|-------|-------------|
| `strategy` | High-level routing strategy: `round-robin`, `random`, `least-loaded`, `least-outstanding`, `weighted-scoring` |
| `scorers` | Comma-separated `scorer:weight` pairs (only when strategy is `weighted-scoring`). Null otherwise. |
| `picker` | Final selection method: `max-score` (highest-scored endpoint) or `random` (ignores scores). Null for `round-robin`/`random` strategies. |

For single-replica configs (num_instances=1 and data_parallel_size=1), `routing_config` is null (no external routing needed). For llm-optimizer configs with data_parallel_size > 1, `routing_config` is also null (vLLM handles internal DP routing).

**Per-tool routing translation:**

| Tool | Tool-native routing | routing_config |
|------|-------------------|----------------|
| **inference-sim** | `round-robin` | `strategy: round-robin` |
| | `least-loaded` | `strategy: least-loaded, scorers: queue-depth:1, picker: max-score` |
| | `weighted` with `--routing-scorers` | `strategy: weighted-scoring, scorers: <from flag>, picker: max-score` |
| **LLMServingSim** | `RR` | `strategy: round-robin` |
| | `RAND` | `strategy: random` |
| | `LOAD` (waiting*4 + running) | `strategy: least-loaded, scorers: queue-depth:4,running-requests:1, picker: max-score` |
| **Vidur** | `round_robin` | `strategy: round-robin` |
| | `lor` (least outstanding) | `strategy: least-outstanding, scorers: active-requests:1, picker: max-score` |
| | `random` | `strategy: random` |
| **AIConfigurator** | N/A (no routing model) | null (linear throughput scaling assumed) |
| **llm-optimizer** | N/A (no replica model) | null |

**llm-d deployment mapping** (routing_config to llm-d-inference-scheduler plugins):

| routing_config strategy | llm-d scorers | llm-d picker |
|------------------------|---------------|--------------|
| `round-robin` | (none) | `random-picker` |
| `random` | (none) | `random-picker` |
| `least-loaded` | `queue-scorer` | `max-score-picker` |
| `least-outstanding` | `active-request-scorer` | `max-score-picker` |
| `weighted-scoring` | mapped by scorer name (see below) | `max-score-picker` |

**Scorer name mapping** (routing_config scorer name to llm-d plugin name):

| routing_config scorer | llm-d plugin |
|----------------------|--------------|
| `prefix-cache` | `precise-prefix-cache-scorer` |
| `queue-depth` | `queue-scorer` |
| `kv-utilization` | `kv-cache-utilization-scorer` |
| `load-balance` | `load-aware-scorer` |
| `active-requests` | `active-request-scorer` |
| `running-requests` | `running-requests-size-scorer` |
| `no-hit-lru` | `no-hit-lru-scorer` |

Scorer weights from `routing_config` map directly to llm-d `schedulingProfiles[].plugins[].weight` values.

**tool_config fields** (prediction-only knobs, not deployed):

| Field | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|-------|:---:|:---:|:---:|:---:|:---:|
| scheduler | Y | - | - | - | - |
| admission_policy | Y | - | - | - | - |
| preemption_policy | Y | - | - | - | - |
| max_concurrency | - | - | - | - | Y |
| vidur_scheduler_type | - | - | - | Y | - |

Notes on vllm_args translation:
- `enable_chunked_prefill`: inference-sim translates `long_prefill_token_threshold > 0` to true. LLMServingSim has explicit boolean. Vidur: true when using sarathi scheduler. llm-optimizer: direct passthrough.
- `max_num_batched_tokens`: when chunked prefill is enabled in llm-optimizer, this value also serves as the effective chunk cap (vLLM's native behavior).
- Vidur's `max_tokens_in_batch` maps directly to `max_num_batched_tokens`. Sarathi's `chunk_size` does not have a direct vLLM equivalent; for validation, deploy with `enable_chunked_prefill=true` and `max_num_batched_tokens` set to the chunk_size value.
- `tool_config` fields capture knobs that affect the tool's prediction but cannot be deployed. These explain prediction differences (e.g., why inference-sim with `priority-fcfs` scheduler predicts differently than with `fcfs`) but are not consumed by the validation harness.
- `data_parallel_size`: llm-optimizer sweeps this as vLLM-internal DP. Other tools set it to 1 (they use deployment replicas for multi-GPU scaling instead). For validation, `deploy.py` passes `--data-parallel-size` to each vLLM pod; for llm-optimizer configs with DP > 1, there is a single pod with multiple internal DP groups.
- For validation, `deploy.py` consumes `vllm_args` (per-instance config) and `routing_config` (llm-d scheduler config). `tool_config` is recorded for analysis only.

### Pareto Front Construction

From each tool's results:
1. Filter to configs where `meets_slo == true`
2. Plot (cost_per_hour, max_throughput_tok_s)
3. Compute Pareto front (non-dominated points)
4. Rank by cost ascending; select top-3 cheapest configs that also exceed a minimum throughput threshold (e.g., 100 tok/s)

## Validation Protocol (Charts 1-2)

### Deployment

For each tool's top-3 recommended configs (15 total, some may overlap):
1. Deploy on llm-d with the recommended configuration:
   - `vllm_args`: configure each vLLM instance (TP, PP, data_parallel_size, max_num_seqs, etc.)
   - `num_instances`: deploy N deployment replica pods (for llm-optimizer configs, num_instances=1 with data_parallel_size > 1 inside the pod)
   - `routing_config`: configure llm-d-inference-scheduler with the corresponding scorer/picker plugins and weights (null for llm-optimizer's DP configs)
   - Use the same model (meta-llama/Llama-3.1-8B)
2. Send servegen m-mid workload at the tool's predicted max throughput rate
3. Measure for 5 minutes after 1-minute warmup
4. Record actual TTFT (mean, P50, P99) and throughput

### Drift Measurement

For each validated config:
- `drift_ttft = actual_ttft_mean - predicted_ttft_mean`
- `drift_throughput = actual_throughput - predicted_throughput`
- `slo_violation = actual_ttft_mean > 300ms`

Chart 1 shows: scatter plot of (TTFT, throughput) with drift arrows from predicted to actual positions.

### Runtime Measurement (Chart 2)

For each tool, record:
- Wall-clock time to complete config search
- Peak memory usage
- GPU-hours consumed (0 for CPU-only tools)
- Whether real GPU hardware was required

## BLIS-Only Analysis (Charts 3-5)

### Chart 3: SLO Tiering

**Model:** Qwen3-14B (BLIS-only, no profiling needed)

Run config search on the BLIS config space twice:
1. **Baseline:** Uniform SLO (mean TTFT < 300ms for all)
   - `--scheduler fcfs --admission-policy always-admit`
2. **Tiered:** Premium at 300ms, standard relaxed to 400ms
   - `--scheduler priority-fcfs --admission-policy tier-shed`

Compare Pareto fronts. Select best config from each. Deploy both on llm-d at their respective predicted max throughput. Measure actual gain.

### Chart 4: Scaling Curve

**Model:** Qwen3-14B (BLIS-only)

For target throughput levels {200, 400, 600, 800, 1000, 1200, 1400, 1600} tok/s:
- BLIS: find minimum GPUs (1, 2, 4, 8) that meets SLO at target throughput
- Other tools: run same sweep (using their results from Charts 1-2, extrapolated to Qwen3-14B if possible, or re-run with Llama-3.1-8B data)

Plot step function: target_throughput (x) vs. cost_per_hour (y).

Validate BLIS's claimed capacity at key inflection points on real hardware.

### Chart 5: Model Selection

**Models:** meta-llama/Llama-3.1-8B, CodeLlama-34B-Instruct, Llama-3.1-70B (aligned with accuracy experiments; spans TP=1 to TP=4)

Setup: 8xH100 node, servegen m-mid workload, SLO: minimum 300 tok/s throughput.
Each model uses its minimum TP: 8B at TP=1, 34B at TP=2, 70B at TP=4.

Each tool predicts max throughput for each model at its minimum TP:
- inference-sim: analytical prediction for each model
- LLMServingSim: simulation with profiling data
- AIConfigurator: `cli estimate` for each model
- Vidur: simulation with H100 profiling data
- llm-optimizer: estimate mode (roofline) for each model

Validate on real hardware: deploy each model on H100 vLLM at minimum TP, measure actual throughput.

## Orchestration Architecture

### Directory Structure

```
experiments/
  config/
    shared_base.yaml          # 40 base configs
    blis_extensions.yaml      # BLIS policy variants
    llmservingsim_ext.yaml    # LLMServingSim knobs
    vidur_config.yml          # Vidur config_optimizer YAML
    llm_optimizer_grid.yaml   # llm-optimizer grid spec
  runners/
    run_blis.py               # Invoke inference-sim with binary rate search
    run_llmservingsim.py      # Generate JSONL, invoke simulator
    run_aiconfigurator.py     # Call AIConfigurator Python API
    run_vidur.py              # Invoke Vidur config_optimizer
    run_llm_optimizer.py      # Invoke llm-optimizer grid search
  validation/
    deploy.py                 # Deploy vLLM config on llm-d cluster
    benchmark.py              # Run workload against deployed instance
    measure.py                # Collect and compute drift metrics
  analysis/
    pareto.py                 # Compute Pareto fronts from results
    charts.py                 # Generate paper figures (matplotlib)
    drift.py                  # Compute and visualize drift arrows
  results/
    raw/                      # Per-tool raw output
    processed/                # Normalized results in common schema
    validated/                # Real deployment measurements
    figures/                  # Generated charts
```

### Execution Order

1. **Setup:** Pull Git LFS for AIConfigurator, verify submodules, build inference-sim
2. **Phase 1 (estimation):** Run all 5 tool sweeps (parallelizable across tools)
3. **Phase 2 (analysis):** Compute Pareto fronts, select top-3 per tool
4. **Phase 3 (validation):** Deploy selected configs on llm-d, benchmark
5. **Phase 4 (BLIS-only):** Run Charts 3-5 experiments, validate key configs
6. **Phase 5 (figures):** Generate all paper charts

### Reproducibility

- All configs stored in YAML (version-controlled)
- Random seed fixed for inference-sim (`--seed 42`)
- Results stored as JSON with full provenance (tool version, timestamp, config hash)
- Validation workloads generated deterministically from fixed seed

**Seed alignment across tools:** Only inference-sim supports explicit seeding (`--seed 42`). LLMServingSim, Vidur, and llm-optimizer do not expose seed flags. Workload arrival times are fully deterministic (generated once from the canonical trace and converted to each tool's format), so the primary source of non-determinism is internal simulation state (e.g., scheduler tie-breaking, cache eviction order). For llm-optimizer, results are from real benchmarks, so run-to-run variance is inherent. Document per-tool variance by running 3 repeats on a representative config subset and reporting the coefficient of variation.

### Error Handling

Each runner script must handle tool failures gracefully to avoid losing progress on long sweeps:

| Failure mode | Detection | Response |
|---|---|---|
| Server startup timeout (llm-optimizer) | `ServerNotReadyError` after 5-min health check | Log error, record config as `status: "server_start_failed"`, continue to next config |
| OOM during inference | Process killed (exit code 137) or CUDA OOM exception | Log error, record config as `status: "oom"`, continue |
| Simulation crash (LLMServingSim, Vidur) | Non-zero exit code or missing output files | Log error, record config as `status: "crashed"`, continue |
| Binary search non-convergence (inference-sim) | Max iterations reached without TTFT crossing SLO threshold | Record last valid probe as result with `status: "unconverged"`, note probe granularity |
| Benchmark timeout | Wall-clock exceeds per-config limit | Kill process, record as `status: "timeout"`, continue |

**Per-config timeouts:**
- inference-sim: 30 seconds (typically ~1s; 30s catches pathological configs)
- LLMServingSim: 5 minutes (typically ~18s; long configs may hit degenerate scheduling)
- AIConfigurator: 30 seconds (analytical, should be instant)
- Vidur: 10 minutes (native search with binary search iterations)
- llm-optimizer: 10 minutes (real benchmark at ~2 min; allows for slow server startup)

**Output schema for failed configs:** Add a `status` field to the output JSON. Values: `"ok"` (normal completion), `"oom"`, `"timeout"`, `"crashed"`, `"server_start_failed"`, `"unconverged"`. Failed configs are excluded from Pareto analysis but included in the raw results for debugging and completeness. The `results` block is null for failed configs; `metadata.wall_clock_seconds` still records elapsed time.

**Checkpointing:** All runners write results incrementally (append to JSONL after each config). For llm-optimizer, use the native `--continue` flag. For other tools, the runner script checks for existing result files on startup and skips completed config hashes.

## Decisions Made

1. **Model choice:** meta-llama/Llama-3.1-8B for comparative charts (aligned with BLIS accuracy experiments #1257), Qwen3-14B for BLIS-only charts. Why: shares infrastructure and validation data with accuracy experiments; Vidur supports Llama-3-8B with H100 profiling.
2. **Two-phase comparison:** Phase A (shared config space with independent replica sweep) isolates prediction accuracy; Phase B (tool-native extensions) demonstrates expressiveness. Why: separates "who predicts best on identical configs" from "who finds better configs by exploring more dimensions."
3. **Vidur SLO metric:** Use Vidur as-is with scheduling-delay SLO. The validation step reveals whether this proxy metric correlates with actual TTFT. This is a finding, not a bug. Why: modifying Vidur's internals would invalidate its published results; better to document the gap.
4. **Validation platform:** llm-d on H100 (vLLM serving engine, llm-d-inference-scheduler for routing). Why: llm-d provides the composable routing layer needed to validate multi-replica routing predictions; vLLM is the serving engine that all estimators model internally.
5. **Cost normalization:** $3.20/hr per GPU applied consistently in post-processing for all tools. Why: removes cloud pricing variance as a confound; makes cost comparisons purely about GPU efficiency.
6. **LLMServingSim H100 profiling:** Run the ~1 hour H100 profiling to generate hardware-matched kernel timing data. Why: strongest sim2real signal requires hardware-matched profiles; 1 hour is cheap relative to the 18+ hours of llm-optimizer benchmarking that follows.

## Open Questions

1. **Minimum throughput threshold for Pareto selection:** Need a floor to avoid recommending configs that meet TTFT SLO but serve trivial throughput. Suggest: minimum 200 tok/s (reasonable for a chatbot deployment).
2. **Chart 4 (scaling curve):** Make BLIS-only (no cross-tool comparison on Qwen3-14B), since other tools lack profiling data for it. Frame as: "BLIS enables scaling analysis for models other tools cannot evaluate."
3. **llm-optimizer scope:** With the addition of the `data_parallel_size` sweep and TP=8, the config count is ~11,200 (~375 GPU-hours at ~2 min each). Need to prune aggressively: reduce concurrency levels, limit block_size/prefix_caching combinations, or restrict DP to only TP=1 configs where DP > 1 is most impactful. Use `--continue` for checkpointing.
