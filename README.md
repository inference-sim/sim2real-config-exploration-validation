# sim2real-config-exploration-validation

Experiment implementation for the Config Exploration section of the BLIS paper. Five LLM serving estimators search a shared config space, and their recommendations are validated against real llm-d deployments on H100 GPUs to measure sim2real drift and SLO compliance.

**Part 1 (Comparative):** All estimators search the same config space. Recommendations are deployed on llm-d to measure sim2real drift and SLO compliance.

**Part 2 (BLIS-only what-if analysis):** Explores SLO tiering, scaling curves, and cross-model selection. All validated on llm-d.

Paper plan: https://github.com/inference-sim/inference-sim/discussions/1237

## Requirements

- Python >= 3.10
- Go >= 1.21 (for building inference-sim)

## Setup

```bash
git clone https://github.com/inference-sim/sim2real-config-exploration-validation.git
cd sim2real-config-exploration-validation
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Defining Your Experiment

The experiment is defined by a YAML config file. The default is `experiments/default_config.yaml`:

```yaml
model:
  name: meta-llama/Llama-3.1-8B       # HuggingFace model ID
  hardware: H100_SXM_80GB             # GPU type
  max_gpus: 8                         # GPU budget (TP * PP * replicas <= this)
  gpu_cost_per_hour: 3.20             # $/GPU-hour for cost calculations

workload:
  preset: servegen                    # Workload family (blis preset name)
  variant: m-mid                      # Workload variant
  num_requests: 10000                 # Total requests in the trace
  isl_mean: 512                       # Mean input sequence length
  osl_mean: 256                       # Mean output sequence length
  arrival_pattern: poisson            # Arrival process
  rate: 10.0                          # Requests/sec for trace generation
  seed: 42

slo:
  ttft_mean_ms: 300                   # SLO threshold for mean TTFT

analysis:
  top_k: 3                           # Number of top configs to select per tool
  min_throughput_tok_s: 200           # Minimum throughput for selection

tools:                                # Which estimators to run
  - blis
  - llmservingsim
  - aiconfigurator
  - vidur
  - llm-optimizer
```

To customize, copy it and pass your version:

```bash
cp experiments/default_config.yaml my_experiment.yaml
# Edit my_experiment.yaml (change model, SLO, etc.)
python -m experiments.run_all --config my_experiment.yaml setup
```

For example, to run Llama 3 70B with a tighter SLO:

```yaml
model:
  name: meta-llama/Llama-3-70B
  hardware: H100_SXM_80GB
  max_gpus: 8
  gpu_cost_per_hour: 3.20
slo:
  ttft_mean_ms: 200
```

## Running Locally

The experiment runs in four phases: **setup**, **prune**, **sweep**, **analyze**. All phases share the same `--config` and `--results-dir` flags.

```bash
# All commands accept these global flags:
#   --config PATH       Experiment YAML (default: experiments/default_config.yaml)
#   --results-dir DIR   Output directory (default: results/)
```

### 1. Setup: generate workloads

Generates the canonical workload trace and converts it to per-tool formats. Requires a built `blis` binary (see [Building inference-sim](#building-inference-sim)).

```bash
python -m experiments.run_all setup
```

This creates `results/workloads/` with the canonical trace and tool-specific conversions. The workload parameters (preset, variant, num_requests, rate, seed) come from the config file.

### 2. Prune: topology pruning pre-pass

Evaluates each topology at a reference config to eliminate those that cannot meet the SLO threshold. Runs per tool. The SLO threshold and GPU budget come from the config file.

```bash
python -m experiments.run_all prune --tool blis
python -m experiments.run_all prune --tool llmservingsim
python -m experiments.run_all prune --tool aiconfigurator
python -m experiments.run_all prune --tool vidur
python -m experiments.run_all prune --tool llm-optimizer
```

You can override config values from the command line:

```bash
python -m experiments.run_all prune --tool blis --max-gpus 4 --slo-ttft-ms 200
```

Pruned topology lists are saved to `results/pruned/`.

### 3. Sweep: full config sweep

Runs the full parameter sweep for a tool. Each tool's runner writes JSONL output to `results/raw/<tool>.jsonl` with checkpoint/resume support (safe to interrupt and restart).

```bash
# inference-sim (CPU only, uses binary rate search)
python -m experiments.run_all sweep --tool blis

# llm-optimizer (requires GPUs; prints the CLI command to run on a GPU node)
python -m experiments.run_all sweep --tool llm-optimizer
```

Other tool runners (llmservingsim, aiconfigurator, vidur) have argument builders implemented but their `evaluate_config` methods are stubs that raise `NotImplementedError`. See [Tool runner status](#tool-runner-status) for details.

### 4. Analyze: Pareto fronts and Chart 1

Computes Pareto fronts per tool, selects top-k configs, and generates Chart 1 (scatter plot with TTFT vs. throughput, per-tool Pareto frontiers, and drift arrows if validation data exists). The `top_k` and `min_throughput` values come from the config file.

```bash
python -m experiments.run_all analyze
```

Outputs:
- `results/processed/top3_selection.json` (or `topN` based on config)
- `results/figures/chart1.pdf` and `results/figures/chart1.png`

### Full local workflow (copy-paste)

```bash
# 0. Build inference-sim
cd estimators/inference-sim && go build -o blis main.go && cd ../..

# 1. Generate workloads
python -m experiments.run_all setup

# 2. Prune topologies (run for each tool you want)
python -m experiments.run_all prune --tool blis

# 3. Sweep configs
python -m experiments.run_all sweep --tool blis

# 4. Generate analysis and Chart 1
python -m experiments.run_all analyze
```

Or with a custom config:

```bash
python -m experiments.run_all --config my_experiment.yaml setup
python -m experiments.run_all --config my_experiment.yaml prune --tool blis
python -m experiments.run_all --config my_experiment.yaml sweep --tool blis
python -m experiments.run_all --config my_experiment.yaml analyze
```

## Running on Kubernetes

All K8s jobs are in `k8s/` and use namespace `jchen`, PVC `data-pvc` mounted at `/data`, and secret `hf-secret` with key `HF_TOKEN`. Results go to `/data/config-exploration/results/`.

### Prerequisites

```bash
# Create HF token secret (if not already present)
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=<your-token> -n jchen
```

### Execution order

```bash
# 1. Setup: build blis binary, generate workloads
kubectl apply -f k8s/setup-job.yaml -n jchen

# 2. Sweeps (run after setup completes; can run in parallel)
kubectl apply -f k8s/blis-sweep.yaml -n jchen           # CPU, 12h limit
kubectl apply -f k8s/llmservingsim-sweep.yaml -n jchen   # CPU, 24h limit
kubectl apply -f k8s/aiconfigurator-sweep.yaml -n jchen  # CPU, 30min limit
kubectl apply -f k8s/vidur-sweep.yaml -n jchen           # CPU, 6h limit
kubectl apply -f k8s/llm-optimizer-sweep.yaml -n jchen   # 8x H100, 15d limit

# 3. Analysis (run after all sweeps complete)
kubectl apply -f k8s/analysis-job.yaml -n jchen          # CPU, 30min limit
```

### Monitoring

```bash
kubectl get jobs -n jchen -l experiment=config-exploration
kubectl logs -f job/<job-name> -n jchen
```

### Cleanup

```bash
kubectl delete -f k8s/ -n jchen
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Building inference-sim

```bash
cd estimators/inference-sim
go build -o blis main.go
```

The binary is used by the setup phase to generate canonical workload traces and by the blis sweep runner for simulation.

## Config Space

Each tool searches over (TP, PP/DP, replicas) topology triples with a GPU budget constraint of `TP * PP * replicas <= 8`, crossed with tool-specific batching, scheduling, and routing parameters.

| Tool | Topologies | Total configs | Notes |
|------|-----------|---------------|-------|
| inference-sim | 15 (no PP) | ~98k | Schedulers, admission, preemption, routing scorers |
| LLMServingSim | 25 (with PP) | ~14k | Chunked prefill, prefix caching, routing |
| AIConfigurator | 25 (with PP) | 25 input triples | Internal sweep produces ~250 output points |
| Vidur | 25 (with PP) | ~4k | Three scheduler types (vllm, sarathi, orca) |
| llm-optimizer | 25 (DP instead of replicas) | ~12k | Native grid search with `--continue` checkpointing |

The topology pruning pre-pass evaluates each topology at a reference config and eliminates those with TTFT > 1.5x the SLO threshold, reducing the config space by 50-70% before the full sweep.

## Results Directory Layout

```
results/
  workloads/         # Canonical trace + per-tool format conversions
  pruned/            # Per-tool pruned topology JSON
  raw/               # Per-tool JSONL sweep output (checkpointed)
  processed/         # Top-3 selection, merged results
  validated/         # Real deployment measurements (future)
  figures/           # chart1.pdf, chart1.png
```

## Tool Runner Status

| Tool | Config generator | Arg builder | Runner | Notes |
|------|-----------------|-------------|--------|-------|
| inference-sim | Done | `build_blis_args()` | Full binary rate search | Requires `blis` binary |
| LLMServingSim | Done | `build_llmservingsim_args()` | Stub | Requires profiling data from `k8s/llmservingsim-profiler.yaml` |
| AIConfigurator | Done | `build_aiconfigurator_args()` | Stub | `pip install aiconfigurator==0.8.0` |
| Vidur | Done | `build_vidur_config_yaml()` | Stub | Requires Vidur + Ray installation |
| llm-optimizer | Done | `build_llm_optimizer_cmd()` | Stub (prints CLI) | Requires GPUs + running vLLM/SGLang server |

## Estimators

| Estimator | Path | Version | Repo |
|-----------|------|---------|------|
| LLMServingSim | `estimators/LLMServingSim` | `baf0feb` | https://github.com/casys-kaist/LLMServingSim |
| AIConfigurator | `estimators/aiconfigurator` | `v0.8.0` | https://github.com/ai-dynamo/aiconfigurator |
| Vidur | `estimators/vidur` | `8383d29` | https://github.com/microsoft/vidur |
| llm-optimizer | `estimators/llm-optimizer` | `bb82d22` | https://github.com/bentoml/llm-optimizer |
| inference-sim | `estimators/inference-sim` | `2da40b9` | https://github.com/inference-sim/inference-sim |
