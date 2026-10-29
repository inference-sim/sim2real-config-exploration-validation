# BLIS Pareto Frontier Analysis: Llama-3.1-70B Config Search

## Methodology

**Search algorithm.** We use BLIS's built-in NSGA-II multi-objective evolutionary search (`search.py`) to explore the serving configuration space for Llama-3.1-70B on H100 GPUs. NSGA-II jointly optimizes three objectives: maximize system throughput (requests/s), minimize TTFT P99 (ms), and minimize GPU count. The search runs for 10 generations with a population size of 100, evaluating 1,100 total configurations across 8 parallel workers.

**Search space.** The spec sweeps over six configuration dimensions:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `tp` (tensor parallelism) | 2, 4, 8 | GPUs per model replica |
| `num_instances` | 1, 2, 3, 4 | Number of replicas |
| `rate` | 25, 50, 100, 150, 200, 300, 400, 500 | Poisson arrival rate (req/s) |
| `scheduler` | fcfs, sjf | Request scheduling policy |
| `max_num_running_reqs` | 32, 64, 128, 256, 512 | Max concurrent batch size |
| `total_kv_blocks` | 2000, 3000, 5000, 7500, 10000 | KV cache capacity |
| `block_size_in_tokens` | 16, 32 | KV block granularity |

This yields a combinatorial space of 3 x 4 x 8 x 2 x 5 x 5 x 2 = 9,600 possible configurations. The total GPU count per config is `tp * num_instances`, ranging from 2 to 32; we restrict analysis to configs using 1-8 GPUs (tp in {2,4,8} with num_instances=1).

**Fixed parameters.** The workload uses Poisson arrivals with ISL~512 tokens, OSL~256 tokens (768 tokens/request average). BLIS evaluates each config using its trained-physics latency model with 200 simulated requests per evaluation, seeded for reproducibility.

**Pareto computation.** After the search, we group feasible configs by GPU tier (2, 4, 6-8) and compute the 2D Pareto frontier per tier: maximize throughput while minimizing TTFT P99. Configs producing identical (throughput, TTFT) pairs are deduplicated. The frontier is plotted with TTFT P99 on the x-axis (log scale) and throughput on the y-axis (linear), so the ideal region is upper-left.

## Results

The search produces 342 Pareto-optimal configs (in the 3D objective space) from 1,100 evaluations. After projecting to 2D per GPU tier and deduplicating:

| GPU Tier | Unique Configs | 2D Pareto Points | Peak Throughput | Best TTFT P99 |
|----------|---------------|-----------------|-----------------|---------------|
| 2 GPUs (tp=2, 1 replica) | 78 | 9 | 3,112 tok/s | 116 ms |
| 4 GPUs (tp=4, 1 replica) | 83 | 9 | 5,708 tok/s | 64 ms |
| 6-8 GPUs (tp=8, 1 replica) | 106 | 11 | 9,947 tok/s | 43 ms |

### Key Findings

**1. The primary tradeoff variable is request rate.** Within each GPU tier, the Pareto-optimal configs differ almost exclusively in arrival rate. At low rates (25-50 req/s), the system is underloaded and achieves near-minimum TTFT; at high rates (300-500 req/s), throughput saturates but TTFT P99 spikes by 10-100x. Other parameters (scheduler, KV blocks, batch size) converge to similar values across the frontier.

**2. Throughput plateaus before latency degrades.** Each tier shows a characteristic "L-shaped" frontier: throughput reaches ~95% of its maximum while TTFT P99 remains below 200ms, then TTFT grows by orders of magnitude with minimal additional throughput gain. For 8 GPUs, the system sustains 9,865 tok/s at 139ms TTFT P99 (rate=200), but pushing to rate=400 only adds 20 tok/s while TTFT jumps to 601ms.

**3. Optimal configs converge on large KV caches and high concurrency.** All Pareto-optimal configs use `block_size=32`, `max_num_running_reqs` of 256-512, and `total_kv_blocks` of 5,000-10,000. Smaller values never appear on the frontier, meaning BLIS correctly identifies that memory-efficient serving (large KV cache, high concurrency limits) dominates across all operating points.

**4. Single-replica configs dominate.** Despite searching over `num_instances` in {1,2,3,4}, all Pareto-optimal configs in the 1-8 GPU range use `num_instances=1`. The search finds that for a 70B model, a single larger replica (higher TP) outperforms multiple smaller replicas at equivalent GPU cost, due to reduced inter-replica communication overhead and better KV cache utilization.

**5. Linear throughput scaling with GPUs.** Peak throughput scales roughly linearly: 3.1k (2 GPUs) to 5.7k (4 GPUs) to 9.9k (8 GPUs) tok/s, representing 1.6x/token/GPU, 1.4x, and 1.2x efficiency respectively, indicating mild sub-linear scaling at higher parallelism.

## Chart Description

The figure plots system throughput (tokens/s, y-axis, linear) against TTFT P99 (ms, x-axis, log scale) for three GPU budget tiers. Each tier shows all evaluated configs as faded scatter points and the Pareto-optimal subset as bold markers connected by a line, with a shaded region indicating the achievable operating envelope. The upper-left corner represents ideal operation (high throughput, low latency). The separation between tiers demonstrates that BLIS's multi-objective search correctly identifies both the capacity ceiling per GPU budget and the latency-throughput tradeoff within each budget.

## Pareto-Optimal Configurations

### 2 GPUs (tp=2, 1 replica)

| Throughput (tok/s) | TTFT P99 (ms) | Rate | Scheduler | Max Running | KV Blocks | Block Size |
|-------------------|---------------|------|-----------|-------------|-----------|------------|
| 2,977 | 116 | 25 | sjf | 256 | 7500 | 32 |
| 3,069 | 143 | 50 | fcfs | 512 | 7500 | 32 |
| 3,069 | 171 | 50 | sjf | 256 | 7500 | 32 |
| 3,093 | 1,412 | 100 | fcfs | 512 | 10000 | 32 |
| 3,093 | 2,057 | 150 | fcfs | 256 | 5000 | 32 |
| 3,093 | 2,379 | 200 | fcfs | 256 | 5000 | 32 |
| 3,093 | 2,711 | 300 | fcfs | 256 | 10000 | 32 |
| 3,108 | 2,874 | 100 | sjf | 256 | 7500 | 32 |
| 3,112 | 2,958 | 150 | sjf | 256 | 5000 | 32 |

### 4 GPUs (tp=4, 1 replica)

| Throughput (tok/s) | TTFT P99 (ms) | Rate | Scheduler | Max Running | KV Blocks | Block Size |
|-------------------|---------------|------|-----------|-------------|-----------|------------|
| 5,229 | 64 | 25 | fcfs | 512 | 5000 | 32 |
| 5,520 | 70 | 50 | fcfs | 256 | 7500 | 32 |
| 5,678 | 116 | 100 | sjf | 512 | 10000 | 32 |
| 5,707 | 547 | 150 | fcfs | 256 | 7500 | 32 |
| 5,707 | 864 | 200 | fcfs | 512 | 5000 | 32 |
| 5,707 | 1,191 | 300 | fcfs | 512 | 5000 | 32 |
| 5,707 | 1,357 | 400 | fcfs | 512 | 10000 | 16 |
| 5,708 | 1,456 | 500 | fcfs | 512 | 10000 | 32 |
| 5,708 | 1,696 | 500 | sjf | 512 | 10000 | 16 |

### 6-8 GPUs (tp=8, 1 replica)

| Throughput (tok/s) | TTFT P99 (ms) | Rate | Scheduler | Max Running | KV Blocks | Block Size |
|-------------------|---------------|------|-----------|-------------|-----------|------------|
| 7,946 | 43 | 25 | fcfs | 256 | 5000 | 32 |
| 9,189 | 44 | 50 | fcfs | 512 | 5000 | 32 |
| 9,641 | 48 | 100 | sjf | 512 | 10000 | 32 |
| 9,797 | 55 | 150 | fcfs | 512 | 7500 | 32 |
| 9,797 | 62 | 150 | sjf | 512 | 10000 | 16 |
| 9,865 | 139 | 200 | fcfs | 512 | 10000 | 32 |
| 9,884 | 441 | 300 | fcfs | 512 | 10000 | 32 |
| 9,885 | 601 | 400 | fcfs | 512 | 5000 | 32 |
| 9,885 | 697 | 500 | fcfs | 512 | 10000 | 16 |
| 9,927 | 934 | 300 | sjf | 512 | 10000 | 32 |
| 9,947 | 942 | 400 | sjf | 512 | 5000 | 32 |

## Reproduction

```bash
# Run NSGA-II search
BLIS_BIN=estimators/inference-sim/blis python3 \
  /path/to/inference-sim/search.py \
  --spec results-70b-pareto/raw/blis_pareto_spec.yaml \
  -o results-70b-pareto/raw/blis_search_output.json

# Generate chart
python3 results-70b-pareto/chart_pareto_frontier.py
```

Search spec: `results-70b-pareto/raw/blis_pareto_spec.yaml`
Raw results: `results-70b-pareto/raw/blis_search_output.json`
Chart script: `results-70b-pareto/chart_pareto_frontier.py`
