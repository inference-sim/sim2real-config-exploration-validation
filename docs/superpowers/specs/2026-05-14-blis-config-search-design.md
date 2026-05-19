# BLIS Config Search Design

## Overview

Design for the Python-based config exploration search capability for BLIS (inference-sim). BLIS is a single-point simulator: each `blis run` invocation evaluates one config at one request rate. This module wraps `blis run` to sweep across a pruned config space, find the maximum sustainable QPS per config, and identify Pareto-optimal configurations.

The search runs locally on a Mac (8 cores) in a reasonable timeframe (under 12 hours total), with no cluster dependency.

## Goals

1. Evaluate all valid BLIS configs to find max QPS meeting the TTFT SLO (mean < 300ms)
2. Produce `ConfigResult` records in the existing output schema (compatible with analysis pipeline)
3. Identify Pareto-optimal configs (best throughput at each cost level)
4. Run on a single Mac (8 cores) in under 12 hours total
5. Support resume (checkpoint after each config, skip completed on restart)

## Config Space Pruning

### Problem

The raw Cartesian product from `generate_blis_configs()` produces 97,920 configs. The experiment spec targets ~25,000 after pruning invalid and redundant combinations.

### Pruning Rules

**Rule 1: Batching pruning (60 raw to 36 combos)**

Two sub-rules applied in sequence:

1a. **Threshold validity** (already in `_batching_combos()`): `chunked_prefill_threshold` must be less than `max_batched_tokens` (otherwise the threshold never triggers). This reduces the raw 60 combos (5 seqs x 3 tokens x 4 threshold values including disabled) to 45.

1b. **Drop `max_num_seqs=512`**: At ISL mean 512 tokens, 512 concurrent sequences would need ~262K tokens of KV cache at TP=1, exceeding practical GPU memory under load. The remaining values {32, 64, 128, 256} cover the useful range. This reduces 45 to 36.

- Raw: 5 (max_num_seqs) x 3 (max_batched_tokens) x 4 (threshold options) = 60
- After threshold pruning: 45 (variable thresholds per token value)
- After dropping seqs=512: 4 x 9 = 36

**Rule 2: Policy pruning for multi-replica configs (192 to ~60 combos)**

The full cross-product `4 schedulers x 2 admission x 2 preemption x 6 routing x 2 block_size = 192` includes many redundant combinations:

1. `admission=tier-shed` is a no-op without priority-aware scheduling. Tier-shed sheds requests based on SLO priority tiers; `fcfs` and `sjf` schedulers ignore priority, making tier-shed equivalent to `always-admit`. Only pair `tier-shed` with `priority-fcfs` and `reverse-priority`.

2. `preemption=priority` is a no-op without priority-aware scheduling. Priority preemption evicts the lowest-priority request; `fcfs` and `sjf` schedulers have no priority signal, making priority preemption equivalent to `fcfs` preemption. Only pair `preemption=priority` with `priority-fcfs` and `reverse-priority`.

Pruned multi-replica policy enumeration:

| Scheduler | Admission | Preemption | Routing | Block | Count |
|-----------|-----------|------------|---------|-------|-------|
| fcfs | always-admit | fcfs | 6 routing | 2 block | 12 |
| sjf | always-admit | fcfs | 6 routing | 2 block | 12 |
| priority-fcfs | always-admit | fcfs | 6 routing | 2 block | 12 |
| priority-fcfs | tier-shed | priority | 6 routing | 2 block | 12 |
| reverse-priority | always-admit | fcfs | 6 routing | 2 block | 12 |
| reverse-priority | tier-shed | priority | 6 routing | 2 block | 12 |

Total: 72 multi-replica policy combos (down from 192).

Note: this is slightly more than the original ~60 estimate because we keep both block sizes. If further reduction is needed, fixing `block_size=16` would halve this to 36.

**Rule 3: Policy pruning for single-replica configs (16 to 12 combos)**

Same logic: only pair `preemption=priority` with priority-aware schedulers.

| Scheduler | Preemption | Block | Count |
|-----------|------------|-------|-------|
| fcfs | fcfs | 2 | 2 |
| sjf | fcfs | 2 | 2 |
| priority-fcfs | fcfs | 2 | 2 |
| priority-fcfs | priority | 2 | 2 |
| reverse-priority | fcfs | 2 | 2 |
| reverse-priority | priority | 2 | 2 |

Total: 12 single-replica policy combos (down from 16).

### Pruned Config Count

| Segment | Topologies | Batching | Policy | Total |
|---------|-----------|----------|--------|-------|
| Multi-replica | 11 | 36 | 72 | 28,512 |
| Single-replica | 4 | 36 | 12 | 1,728 |
| **Total** | | | | **30,240** |

~30,000 configs. Close to the spec's ~25,000 target. To get closer to 25,000, we could fix `block_size=16` (standard vLLM default), which gives 11 x 36 x 36 + 4 x 36 x 6 = 15,120. That may be too aggressive. The 30,240 count is a reasonable compromise.

## Rate Search Algorithm

### Per-Config Search: Exponential Ramp + Bisection

For each config, find the maximum QPS where mean TTFT < 300ms. TTFT is monotonically increasing in rate (more load = more queuing = higher latency).

```
1. Exponential ramp: start at 10 QPS, double until TTFT > SLO or timeout
   Probes: ~4-6 (log2 of rate range)
2. Bisection: narrow the boundary with 5 binary search steps
   Probes: 5
3. Total: ~9-11 probes per config
```

### Multi-Fidelity Approach

Empirical measurement shows `blis run` execution time scales linearly with `num_requests`:

| num_requests | Time per probe | Accuracy vs 10k |
|-------------|----------------|-----------------|
| 100 | ~0.04s | Unreliable (no steady state) |
| 500 | ~0.4s | ~10-30% error, ranking preserved |
| 1000 | ~1.0s | ~2-9% error, ranking preserved |
| 10000 | ~27s | Ground truth |

Key finding: 1000-request probes preserve the relative ranking of configs (which is better/worse) while being 27x faster. 500-request probes are 67x faster but break down for high-throughput multi-replica configs (simulator doesn't reach steady state).

**Two-phase strategy:**

**Phase 1: Screening sweep (default: 500 requests)**
- Evaluate ALL ~30,000 configs with binary rate search at reduced fidelity
- At 500 requests (~0.4s/probe): ~10 probes x 0.4s = ~4s per config
- With 6 parallel workers: 30,000 x 4 / 6 = 20,000s = **~5.5 hours**
- At 1000 requests (~1.0s/probe): ~10s per config, 50,000s / 6 = ~14 hours (exceeds overnight target)
- Output: approximate max QPS for every config, sufficient for ranking

**Phase 2: Confirmation sweep (10,000 requests)**
- Select top candidates: Pareto front per topology + top 20 configs per topology
- Estimated ~300-500 configs
- Full binary rate search at 10,000 requests: ~10 probes x 27s = ~270s per config
- With 6 parallel workers: 500 x 270 / 6 = 22,500s = **~6.3 hours**

**Total runtime: ~12 hours** with 500-request screening (default). Fits in an overnight run on a Mac. Using 1000-request screening increases Phase 1 to ~14 hours (~20 hours total), which exceeds the overnight target but provides better accuracy for the screening phase.

### Configurable Parameters

All search parameters are configurable via CLI flags or config file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-requests-screen` | 500 | Requests per probe in Phase 1 |
| `--num-requests-confirm` | 10000 | Requests per probe in Phase 2 |
| `--slo-ttft-ms` | 300 | Mean TTFT SLO threshold |
| `--max-rate` | 10000 | Upper bound for rate search |
| `--bisect-iters` | 5 | Binary search iterations |
| `--workers` | 6 | Parallel worker count |
| `--top-k-per-topo` | 20 | Candidates per topology for Phase 2 |
| `--output` | results/raw/blis.jsonl | Output path |

## Parallelism

Use `concurrent.futures.ProcessPoolExecutor` with configurable worker count (default 6, leaving 2 cores free on an 8-core Mac). Each worker runs one `blis run` subprocess at a time.

Work distribution: partition configs into shards (round-robin across workers by config index, matching the K8s Indexed Job pattern for consistency). Each worker processes its shard sequentially.

No shared state between workers. Each worker writes to its own shard file; a final aggregation step concatenates them.

## Resume Support

Each shard file is append-only JSONL. On startup, each worker loads completed config hashes from its shard file via `BaseRunner.load_completed_hashes()` and skips already-evaluated configs. This allows resuming after interruption without re-evaluating completed work.

Phase 2 similarly checks for existing results before re-evaluating.

## OOM Mitigation

The previous K8s deployment experienced OOM after ~60 configs per pod. The OOM was in the Python orchestrator process, not in the blis binary (which is a fresh Go process per invocation).

Mitigations:
1. Each worker subprocess is a fresh Python process (via ProcessPoolExecutor), not a long-lived loop in a single process
2. Results are written to disk immediately after each config (not accumulated in memory)
3. The blis subprocess is a fresh OS process per invocation (no cumulative memory in Go)

## Output Format

Each config evaluation produces a `ConfigResult` JSON record matching the existing schema (`experiments/schema/output.py`). Phase 1 results include `metadata.status = "screened"` and Phase 2 results include `metadata.status = "ok"`.

The output JSONL file is directly consumable by the existing `experiments/analysis/pareto.py` and `experiments/analysis/chart1.py` analysis pipeline.

## File Structure

```
experiments/
  config/
    blis_configs.py          # MODIFIED: add generate_pruned_blis_configs()
    pruning.py               # Existing pruning utilities
  runners/
    run_blis.py              # MODIFIED: add multi-fidelity rate search methods
    blis_search.py           # NEW: orchestrator (CLI entrypoint, parallel dispatch)
```

### `blis_configs.py` changes

Add `generate_pruned_blis_configs()` that applies Rule 1 (batching) and Rule 2/3 (policy) pruning. Keep `generate_blis_configs()` unchanged for backward compatibility.

### `run_blis.py` changes

Add `BLISRunner.find_max_rate(config, num_requests, bisect_iters)` as a lightweight rate search method that returns `(max_rate, metrics)` without constructing a full `ConfigResult`. This is used by both Phase 1 screening and Phase 2 confirmation.

### `blis_search.py` (new)

CLI entrypoint for the two-phase config search:

```
python -m experiments.runners.blis_search [--phase1-only] [--phase2-only] [--workers 6] ...
```

Orchestrates:
1. Generate pruned config list
2. Phase 1: parallel screening sweep
3. Identify Phase 2 candidates (Pareto front + top-k per topology)
4. Phase 2: parallel confirmation sweep
5. Aggregate results into final output file

## Validation

Before running the full sweep, validate the search on a small sample:

```bash
# Smoke test: 10 configs, verify output format
python -m experiments.runners.blis_search --max-configs 10 --workers 2

# Fidelity check: compare 1000-req vs 10k-req rankings on 50 configs
python -m experiments.runners.blis_search --fidelity-check --max-configs 50
```

## Timing Estimates

| Scenario | Phase 1 | Phase 2 | Total |
|----------|---------|---------|-------|
| **Mac (6 workers, 500 req screen)** | **~5.5h** | **~6h** | **~11.5h** |
| Mac (6 workers, 1000 req screen) | ~14h | ~6h | ~20h |
| K8s (512 workers, 500 req screen) | ~4 min | ~4.5 min | ~9 min |
| K8s (512 workers, 10k req full, no Phase 2) | ~2.5h | N/A | ~2.5h |

Default (bolded) is the recommended local configuration.

## Open Questions

1. Should we fix `block_size=16` to reduce from ~30,000 to ~15,000 configs? This halves runtime but loses visibility into block size effects.
2. Is 500-request screening fidelity sufficient? The ranking is preserved for most configs, but high-throughput multi-replica configs may be mismeasured. The Phase 2 confirmation corrects this for top candidates.
3. Should Phase 2 also re-evaluate configs that were "unconverged" (timed out) in Phase 1? These might succeed at 10k requests with a longer timeout.
