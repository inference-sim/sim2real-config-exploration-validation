# Vidur - LLM Inference System Simulator

## Overview

Vidur is a high-fidelity simulator for LLM inference systems from Microsoft Research. It enables capacity planning, performance analysis, and testing of scheduling algorithms without requiring GPU access beyond an initial profiling phase.

## Installation

### Option 1: venv (Recommended)
```bash
cd estimators/vidur
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: mamba
```bash
cd estimators/vidur
mamba env create -p ./env -f ./environment.yml
mamba env update -f environment-dev.yml
```

### Option 3: conda
```bash
cd estimators/vidur
conda env create -p ./env -f ./environment.yml
conda env update -f environment-dev.yml
```

### Dependencies
- numpy, pandas, scikit-learn
- wandb (optional, can be disabled)
- plotly_express, matplotlib, seaborn
- kaleido, ddsketch, fasteners

## Usage

### Basic Command
```bash
python -m vidur.main
```

### Key Configuration Parameters

**Model and Hardware:**
- `--replica_config_model_name`: Model to simulate (e.g., `meta-llama/Meta-Llama-3-8B`, `meta-llama/Llama-2-7b-hf`)
- `--replica_config_device`: Device type (`a100`, `h100`)
- `--replica_config_tensor_parallel_size`: Tensor parallelism degree (1, 2, 4, 8)
- `--replica_config_num_pipeline_stages`: Pipeline parallel stages
- `--cluster_config_num_replicas`: Number of independent serving replicas. Each replica is a fully independent Kubernetes-style pod with its own `ReplicaScheduler`, KV cache (memory planner), and request queue (`cluster.py:27-29` creates separate `Replica` objects, `base_global_scheduler.py:26-37` creates a separate scheduler per replica). Requests are dispatched across replicas by `--global_scheduler_config_type` (round_robin, lor, random). Not equivalent to vLLM's internal `--data-parallel-size`.

**Request Generation:**
- `--request_generator_config_type`: `synthetic` or `trace`
- `--synthetic_request_generator_config_num_requests`: Number of requests to generate
- `--length_generator_config_type`: `trace`, `zipf`, or `fixed`
- `--trace_request_length_generator_config_trace_file`: Path to trace file
- `--trace_request_length_generator_config_max_tokens`: Maximum context length

**Request Arrival Pattern:**
- `--interval_generator_config_type`: `poisson`, `gamma`, `trace`, or `static`
- `--poisson_request_interval_generator_config_qps`: Queries per second for Poisson arrivals

**Scheduling:**
- `--replica_scheduler_config_type`: Scheduler type (`sarathi`, `vllm`, `orca`, `lightllm`, `faster_transformer`)
- `--sarathi_scheduler_config_batch_size_cap`: Maximum batch size
- `--sarathi_scheduler_config_chunk_size`: Chunk size for chunked prefill
- `--vllm_scheduler_config_max_tokens_in_batch`: Token limit per batch

**Execution Time Prediction:**
- `--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size`: Max prefill chunk size
- `--random_forrest_execution_time_predictor_config_prediction_max_batch_size`: Max batch size for prediction
- `--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request`: Max tokens per request

**Metrics and Output:**
- `--metrics_config_output_dir`: Output directory (default: `simulator_output/`)
- `--metrics_config_wandb_project`: W&B project name (optional)
- `--metrics_config_wandb_group`: W&B group name (optional)

### Full Example
```bash
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 512 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 16384 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 6.45 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_batch_size_cap 512 \
  --sarathi_scheduler_config_chunk_size 512 \
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
  --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
  --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
```

### Get Help
```bash
python -m vidur.main -h
```

## Output

The simulator generates:
- Metrics logged to W&B (if configured) and stored in `simulator_output/<timestamp>/`
- Chrome trace files viewable at `chrome://tracing/` or `edge://tracing/`
- Performance metrics: TTFT, TPOT, request E2E time, batch size distributions, memory usage, MFU

Key metrics include:
- Time-to-first-token (TTFT) - latency to first output
- Time-per-output-token (TPOT) - inter-token delay
- Request end-to-end latency
- Scheduling delays and preemption times
- Batch size and token distributions
- Per-replica memory usage and utilization
- Model FLOPS Utilization (MFU)

See `docs/metrics.md` for complete metric descriptions.

## Configuration Space Search

Vidur includes native support for automated configuration space exploration to find optimal deployment configurations that meet SLO requirements.

### Config Explorer

The config explorer (`vidur.config_optimizer.config_explorer.main`) performs automated capacity search using binary search to find the maximum sustainable QPS under specified SLO constraints.

**Key features:**
- Binary search over QPS to find capacity limits for each configuration
- SLO-aware search with configurable latency targets (scheduling delay quantiles)
- Parallel execution using Ray for sweeping multiple configurations
- Caching support for faster iteration
- Generates Pareto curves for cost-capacity tradeoffs

**Usage:**
```bash
python -m vidur.config_optimizer.config_explorer.main \
  --config-path <path-to-config.yml> \
  --output-dir <output-directory> \
  --cache-dir ./cache \
  --scheduling-delay-slo-value 5.0 \
  --scheduling-delay-slo-quantile 0.99 \
  --max-iterations 20 \
  --min-search-granularity 2.5 \
  --time-limit 30 \
  --num-threads 8
```

**Key arguments:**
- `--config-path`: YAML file specifying configuration space (models, devices, schedulers, batch sizes, TP/PP dimensions)
- `--scheduling-delay-slo-value`: Target scheduling delay threshold (seconds)
- `--scheduling-delay-slo-quantile`: Percentile for SLO (e.g., 0.99 for P99)
- `--max-iterations`: Maximum binary search iterations per configuration
- `--min-search-granularity`: Minimum search granularity as percentage of QPS
- `--time-limit`: Time limit per simulation run (minutes)

**Configuration space YAML format:**
```yaml
clusters:
  - device: a100
    num_gpus: 16
    gpus_per_node: 4

schedulers:
  - scheduler: vllm
  - scheduler: sarathi
    chunk_size: 512

traces:
  - name: chat
    trace_file: "./data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv"
    max_seq_len: 4096
    num_requests: 16000
    start_qps: 32

batch_sizes: [32, 64, 128]
tp_dimensions: [1, 2, 4, 8]
pp_dimensions: [1, 2, 4]

models:
  - name: llama-2-7b-hf
    identifier: meta-llama/Llama-2-7b-hf
```

The explorer generates all valid combinations from the Cartesian product of the specified dimensions.

### Binary Search Algorithm (Internals)

Source: `vidur/config_optimizer/config_explorer/capacity_search.py`

**Algorithm:**
1. Initialize bounds: `left = 0`, `right = start_qps * 2`
2. For each iteration (max 20 by default):
   - Probe midpoint QPS: run full simulation via `vidur.main`
   - Read result metric from `{run_dir}/*/plots/request_scheduling_delay.csv`
   - Compute: `metric_value = df["request_scheduling_delay"].quantile(slo_quantile)`
   - Pass/fail: `metric_value <= slo_value`
   - Adaptive step sizing: if well under SLO (< slo/8), expand right bound 4x; if far over (> 1000ms), halve left bound
3. Converge when: `abs(left - right) < min_search_granularity * qps / 100` (default 2.5%)

**SLO constraint is hardcoded to scheduling delay.** The `_is_under_sla()` method (lines 69-85) reads only `request_scheduling_delay.csv`. There is no CLI option to switch to TTFT or other metrics.

### Available Output Metrics (Per Simulation Run)

Each simulation produces CSV files in `{output_dir}/{timestamp}/plots/`:

| File | Metric | Description |
|------|--------|-------------|
| `prefill_e2e_time.csv` | **TTFT** | Time from request arrival to prefill completion |
| `request_scheduling_delay.csv` | Scheduling delay | Time waiting in queue before first execution |
| `request_e2e_time.csv` | E2E latency | Total time from arrival to last token |
| `request_e2e_time_normalized.csv` | Normalized E2E | E2E time per decode token |
| `request_execution_time.csv` | Execution time | Actual compute time (excludes queueing) |
| `batch_execution_time.csv` | TBT (approx) | Per-batch execution time (proxy for inter-token latency) |
| `request_preemption_time.csv` | Preemption time | Time spent preempted |

Additionally, `request_metrics.csv` contains per-request detail with all metrics above as columns.

The `stats_extractor.py` (in `config_optimizer/analyzer/`) extracts summary statistics (mean, P90, P95, P99) for `ttft` (from `prefill_e2e_time`), `tbt`, and `scheduling_delay`.

### Using TTFT as SLO Constraint (Requires Code Patch)

The native config_optimizer only supports scheduling delay as the SLO metric. To use TTFT (or other metrics), a small patch to `capacity_search.py` is required:

1. **Change `_get_result_file()`** (lines 60-67): read `prefill_e2e_time.csv` instead of `request_scheduling_delay.csv`
2. **Change `_is_under_sla()`** (lines 69-85): compute quantile over `prefill_e2e_time` column
3. **Adjust adaptive step sizing** (lines 154-171): thresholds are tuned for scheduling delay in seconds; TTFT values will differ in magnitude

The dashboard (`analyzer/dashboard/best_config_page.py`, lines 185-198) already supports TTFT and TBT SLO filtering for post-hoc analysis of search results, but this does NOT feed back into the binary search.

### Analysis Tools

**Pareto Curve Generation:**
```bash
python -m vidur.config_optimizer.analyzer.generate_pareto_curves \
  --results-dir <search-results-dir>
```

Generates Pareto frontier plots showing capacity-per-dollar vs latency metrics for different configurations.

**Interactive Dashboard:**
```bash
streamlit run vidur/config_optimizer/analyzer/dashboard/main.py
```

Provides interactive analysis with:
- Best configuration selection under SLO constraints (supports TTFT and TBT filtering here)
- Configuration comparison across metrics
- Cost analysis and capacity planning
- Search convergence analysis
- Pareto curve visualization

### Output

The config explorer produces:
- Per-configuration max QPS under SLO
- Simulation results for each tested QPS point (cached for reuse)
- JSON files with search results and configuration metadata
- Data suitable for Pareto analysis and dashboard visualization

## Supported Models

- Llama-3-8B, Llama-3-70B (16k context)
- Llama-2-7B, Llama-2-70B (4k context)
- CodeLlama-34B (4k context)
- InternLM-20B, Qwen-72B (4k context)

Supported devices: A100 80GB DGX, H100 DGX, 4xA100 80GB Pairwise NVLink, 8xA40 Pairwise NVLink

## References

- Paper: [Vidur: A Large-Scale Simulation Framework For LLM Inference](https://arxiv.org/abs/2405.05465) (MLSys'24)
- Profiling guide: `docs/profiling.md`
- Metrics reference: `docs/metrics.md`
