# Results: Llama-2-70b Config Exploration (500ms SLO, staged constant)

Experiment: sim2real config exploration validation for the BLIS paper.
Model: meta-llama/Llama-2-70b-hf on H100 SXM 80GB GPUs.
Workload: shared_prefix_codegen (ISL mean=566/max=1024, OSL mean=247/max=512, staged constant arrival, 10000 requests).
SLO: 500ms mean TTFT.

## Search Summary

| Tool | Configs Searched | Evaluated | Feasible | Pareto Points |
|------|-----------------|-----------|----------|---------------|
| BLIS (inference-sim) | 840 | 344 | 344 | 2 distinct (298 equivalent configs) |
| AIConfigurator | 13 | 13 | 13 | n/a |
| llm-optimizer | 13 | 13 | 13 | n/a |

## BLIS Pareto Analysis

BLIS's NSGA-II search optimizes three objectives simultaneously: minimize GPU count, maximize throughput, and minimize mean TTFT. Of 840 configs searched, 496 were infeasible (all TP=2 configs, plus TP=4x2 and some TP=4x1/TP=8x1 configs that exceed capacity).

The 344 feasible configs collapse to only **2 distinct Pareto-optimal operating points** in objective space:

| Operating Point | GPUs | Throughput | Mean TTFT | Configs |
|-----------------|------|------------|-----------|---------|
| TP=4x1 | 4 | 8.0 qps | 46 ms | 146 |
| TP=8x1 | 8 | 9.0 qps | 33 ms | 152 |

298 configs are flagged as Pareto-optimal because they map to these two points with identical predicted performance. The remaining 46 feasible configs (TP=4x2 = 8 GPUs, 8.0 qps, 45-46ms TTFT) are dominated by TP=8x1 (same GPU count, better throughput and TTFT).

### Why so few distinct points?

The simulator (inference-sim) is insensitive to scheduling knobs at this workload and offered rate. Parameters that vary across the 298 Pareto configs but produce identical predictions:

- `max_num_seqs`: 32, 64, 128, 256, 512
- `enable_chunked_prefill`: true/false
- `block_size`: 16, 32
- `max_num_batched_tokens`: 2048, 4096, 8192

Only GPU topology (TP x replicas) meaningfully differentiates performance in the simulator's model at this load level. This is a known limitation of analytical/trained-physics simulators at moderate utilization: when the system is far from saturation, scheduling policy has negligible impact on mean latency.

### Effect of tightening TTFT SLO

Tightening the SLO does not reduce the Pareto set because all feasible configs already predict TTFT between 33-46ms (well below any reasonable SLO threshold):

| SLO | Feasible | Pareto | % |
|-----|----------|--------|---|
| 500ms | 344 | 298 | 86% |
| 100ms | 344 | 298 | 86% |
| 50ms | 344 | 298 | 86% |

### Correct paper framing

Instead of reporting "298 Pareto-optimal configs" (which inflates the apparent exploration value), report:

> "840 configs evaluated across 6 topology/scheduling dimensions. 344 feasible under 500ms TTFT SLO. The 3-objective Pareto front reduces to 2 distinct operating points: TP=4x1 (4 GPUs, 8.0 qps, 46ms) and TP=8x1 (8 GPUs, 9.0 qps, 33ms). 298 configs with varying scheduling parameters achieve identical predicted performance, confirming that GPU topology dominates at moderate utilization."

## Infeasibility Breakdown

| Topology | Searched | Infeasible | Reason |
|----------|----------|------------|--------|
| TP=2x1 (2 GPUs) | 98 | 98 (100%) | Cannot sustain offered rate at TP=2 |
| TP=2x2 (4 GPUs) | 86 | 86 (100%) | Cannot sustain offered rate at TP=2 |
| TP=2x4 (8 GPUs) | 99 | 99 (100%) | Cannot sustain offered rate at TP=2 |
| TP=4x1 (4 GPUs) | 210 | 64 (30%) | Some configs exceed capacity |
| TP=4x2 (8 GPUs) | 142 | 96 (68%) | Some configs exceed capacity |
| TP=8x1 (8 GPUs) | 205 | 53 (26%) | Some configs exceed capacity |

Key finding: TP=2 is completely infeasible for Llama-2-70b at this workload regardless of replicas or scheduling parameters. The minimum viable topology is TP=4x1 (4 GPUs).
