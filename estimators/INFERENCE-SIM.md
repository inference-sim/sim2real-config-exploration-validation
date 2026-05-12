# inference-sim (BLIS)

## Overview

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation - all driven by trained performance coefficients, analytical roofline estimates, or physics-informed cross-model prediction. The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

## Build and Install

**Requirements:** Go >= 1.21

**Build:**
```bash
cd estimators/inference-sim
go build -o blis main.go
```

**Optional environment setup:**
```bash
export HF_TOKEN=your_token_here  # For gated models and avoiding rate limits
```

**Note:** On first run, BLIS auto-fetches the model's `config.json` from HuggingFace. Subsequent runs use the cached config in `model_configs/`.

## Commands

### `blis run`
Run the inference simulation with workload generation.

**Key flags:**
- `--model` - LLM name (e.g., `qwen/qwen3-14b`)
- `--hardware` - GPU type (e.g., `H100`)
- `--tp` - Tensor parallelism degree
- `--latency-model` - Latency estimation backend: `roofline` (default, analytical) or `trained-physics` (physics-informed, recommended)
- `--num-instances` - Number of independent serving replicas in the cluster (default: 1). Each instance is a fully independent Kubernetes-style pod with its own KV cache, scheduler, and request queue (`cluster.go:274-341` creates separate `InstanceSimulator` objects). Requests are dispatched across instances by the cluster-level router (`--routing-policy`). This models llm-d's architecture where each vLLM pod runs independently and the llm-d inference scheduler routes requests across pods. Not equivalent to vLLM's internal `--data-parallel-size`.
- `--rate` - Requests per second (default: 1)
- `--num-requests` - Total requests to generate (default: 100)
- `--workload` - Workload preset: `chatbot`, `summarization`, `contentgen`, `multidoc`, `distribution` (default)
- `--workload-spec` - Path to YAML workload specification file
- `--max-num-running-reqs` - Maximum requests in running batch (default: 256; equivalent to vLLM's `max_num_seqs`)
- `--max-num-scheduled-tokens` - Maximum total new tokens per step (default: 2048; equivalent to vLLM's `max_num_batched_tokens`)
- `--long-prefill-token-threshold` - Chunked prefill threshold (default: 0 = disabled; >0 = chunk size in tokens)
- `--block-size-in-tokens` - KV cache block size in tokens (default: 16)
- `--routing-policy` - Routing policy: `round-robin`, `least-loaded`, `weighted`, `always-busiest`
- `--routing-scorers` - Scorer weights for weighted routing (e.g., `precise-prefix-cache:2,queue-depth:1,kv-utilization:1`)
- `--admission-policy` - Admission policy: `always-admit` (default), `token-bucket`, `tier-shed`, `gaie-legacy`, `reject-all`
- `--scheduler` - Instance scheduler: `fcfs`, `priority-fcfs`, `sjf`, `reverse-priority`
- `--preemption-policy` - Preemption policy: `fcfs` (tail-of-batch), `priority` (least-urgent SLO tier)
- `--policy-config` - Path to YAML policy configuration file
- `--trace-output` - Export workload as TraceV2 files (`<prefix>.yaml` + `<prefix>.csv`)
- `--metrics-path` - Write aggregate metrics JSON

### `blis observe`
Dispatch workload requests to a real inference server and record timing into TraceV2 files.

**Key flags:**
- `--server-url` - Inference server URL (required)
- `--model` - Model name for API requests (required)
- `--trace-header` - Output path for TraceV2 header YAML (required)
- `--trace-data` - Output path for TraceV2 data CSV (required)
- `--workload-spec` - Path to YAML workload specification
- `--workload` - Workload preset name (requires `--rate`)
- `--rate` - Requests per second
- `--concurrency` - Number of concurrent virtual users (closed-loop)
- `--num-requests` - Maximum requests to generate
- `--api-format` - API format: `completions` (default) or `chat` (for `/v1/chat/completions`)
- `--rtt-ms` - Measured network round-trip time in milliseconds
- `--record-itl` - Record per-chunk timestamps for inter-token latency calibration
- `--itl-output` - Output path for ITL CSV file
- `--max-concurrency` - Maximum simultaneous in-flight requests (default: 256)
- `--min-tokens` - Set `min_tokens` to force server to generate at least N tokens before EOS
- `--unconstrained-output` - Do not set `max_tokens` (let server decide output length)

### `blis replay`
Replay a TraceV2 file through the discrete-event simulator.

**Key flags:**
- `--trace-header` - Path to TraceV2 header YAML file (required)
- `--trace-data` - Path to TraceV2 data CSV file (required)
- `--model` - LLM name (required)
- `--results-path` - File to write per-request SimResult JSON for calibration
- `--session-mode` - Session replay mode: `fixed` (default, pre-baked arrivals) or `closed-loop` (load-adaptive follow-ups)
- `--think-time-ms` - Override think time between session rounds (closed-loop mode)
- `--trace-output` - Export replay results as TraceV2 files (header mode: "replayed")
- All simulation configuration flags from `blis run` apply

### `blis calibrate`
Compare real observed latencies against simulator predictions.

**Key flags:**
- `--trace-header` - Path to TraceV2 header YAML file (required)
- `--trace-data` - Path to TraceV2 data CSV file (required)
- `--sim-results` - Path to SimResult JSON file from `blis replay --results-path` (required)
- `--report` - Path to write calibration report JSON (required)
- `--itl-data` - Optional path to ITL CSV file (from `blis observe --record-itl`)
- `--warmup-requests` - Number of initial requests to exclude (default: from trace header)
- `--network-rtt-us` - Network RTT in microseconds added to sim-side latencies

### `blis convert`
Convert external workload formats to v2 WorkloadSpec YAML. Output to stdout.

**Subcommands:**
- `preset` - Convert a named workload preset to v2 spec
  - `--name` - Preset name (e.g., `chatbot`)
  - `--rate` - Requests per second
  - `--num-requests` - Number of requests
- `servegen` - Convert ServeGen data directory to v2 spec
  - `--path` - Path to ServeGen data directory
  - `--time` - Single period for testing (e.g., `midnight`)
- `inference-perf` - Convert inference-perf YAML spec to v2 spec
  - `--spec` - Path to inference-perf spec YAML

### `blis compose`
Merge multiple v2 WorkloadSpec YAML files into one. Output to stdout.

**Key flags:**
- `--from` - Path to v2 WorkloadSpec YAML file (can be repeated)

## Example Invocations

**Basic simulation:**
```bash
./blis run --model qwen/qwen3-14b
```

**Multi-instance cluster with weighted routing:**
```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --routing-policy weighted \
  --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1" \
  --rate 100 --num-requests 500
```

**Using trained-physics latency model:**
```bash
./blis run --model qwen/qwen3-14b --latency-model trained-physics
```

**Observe-replay-calibrate pipeline:**
```bash
# 1. Observe real server
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload chatbot --rate 10 --num-requests 100 \
  --trace-header trace.yaml --trace-data trace.csv

# 2. Replay through simulator
./blis replay --trace-header trace.yaml --trace-data trace.csv \
  --model qwen/qwen3-14b --results-path results.json

# 3. Compare and calibrate
./blis calibrate --trace-header trace.yaml --trace-data trace.csv \
  --sim-results results.json --report calibration.json
```

**Workload conversion:**
```bash
# Convert preset to v2 spec
./blis convert preset --name chatbot --rate 10 --num-requests 100 > workload.yaml

# Merge multiple specs
./blis compose --from spec1.yaml --from spec2.yaml > merged.yaml
```

**Custom workload specification:**
```bash
./blis run --model qwen/qwen3-14b --workload-spec examples/servegen-language.yaml
```

## Output Metrics

Simulation results include:
- `ttft_mean_ms`, `ttft_p99_ms` - Time to First Token (prefill latency)
- `e2e_mean_ms`, `e2e_p99_ms` - End-to-End latency (total request time)
- `itl_mean_ms`, `itl_p99_ms` - Inter-Token Latency (time between output tokens)
- `responses_per_sec` - Completed requests per second
- `tokens_per_sec` - Output tokens generated per second
- `completed_requests` - Number of requests finished within simulation window
- `preemption_count` - Number of request evictions (0 = healthy)

## Prefix Caching Support

inference-sim **models prefix caching** extensively:

- **Router-side cache tracking**: `PrefixCacheIndex` maintains an approximate view of which block hashes each instance has cached, using hierarchical block hashing (each block's hash chains the previous) and LRU eviction per instance
- **Request metadata**: Requests carry `PrefixGroup` (shared prefix group name) and `PrefixLength` (shared prefix token count) fields
- **Routing integration**: Multiple routing scorers leverage prefix cache state:
  - `precise-prefix-cache`: Queries actual instance KV cache state with min-max normalization (llm-d production parity)
  - `prefix-affinity`: Routes requests with shared prefixes to instances that have seen similar prefixes
  - `no-hit-lru`: Distributes cold requests to least-recently-used endpoints
- **Default routing profile**: `precise-prefix-cache:2,queue-depth:1,kv-utilization:1` (llm-d parity)

Implementation files:
- `sim/prefix_cache_index.go` - Main prefix cache index implementation
- `sim/request.go` - Request struct with PrefixGroup and PrefixLength fields
- `sim/routing_scorers.go` - Routing scorer integration
- `sim/routing_precise_prefix_scorer.go` - Precise prefix cache scorer
- `sim/routing_prefix_scorer.go` - Prefix affinity scorer

## Parallelism Support

- **Tensor Parallelism (TP)**: Supported via `--tp` flag. Used for latency estimation and KV cache capacity calculations.
- **Pipeline Parallelism (PP)**: **NOT supported**. There is no PP flag or pipeline parallel logic in the codebase. Only TP is modeled.

## Config Search / Exploration

**Native Support:** No

inference-sim (BLIS) does not provide built-in configuration space search or optimization capabilities. There is no native command, flag, or feature for automatically sweeping or optimizing over configurations to find the best deployment that meets SLO compliance targets.

### What is Available

BLIS is a single-point simulator - each invocation of `blis run` evaluates one specific configuration (model, hardware, TP, number of instances, routing policy, etc.) and returns performance metrics. Users must manually:

1. Run multiple simulations with different configurations (e.g., different `--num-instances` values)
2. Compare output metrics (`ttft_p99_ms`, `e2e_mean_ms`, `responses_per_sec`) across runs
3. Identify which configuration meets their SLO requirements

**Fitness evaluation** (`--fitness-weights`) provides a weighted score for comparing configurations, but does not automate the search process. Users must still manually invoke the simulator for each configuration candidate.

### Manual Exploration Approach

The tutorial at `docs/getting-started/tutorial.md` demonstrates manual capacity planning:
- Measure single-instance capacity under load
- Run simulations at different instance counts
- Compare metrics to find the minimum deployment meeting SLO targets
- This requires multiple manual invocations with different `--num-instances` values

### Using BLIS for Config Exploration

To perform configuration space search with BLIS, users need to:
1. Write a wrapper script that invokes `blis run` with different parameter combinations
2. Parse JSON output metrics from each run
3. Implement search logic (grid search, random search, optimization algorithm) externally
4. Use `--seed` flag for deterministic results during comparisons

## Documentation

Full documentation at `estimators/inference-sim/docs/`:
- [Getting Started](docs/getting-started/index.md) - Installation, tutorials, capacity planning
- [User Guide](docs/guide/index.md) - Routing policies, KV cache, workloads, cluster simulation
- [Concepts](docs/concepts/index.md) - Architecture, core engine, roofline estimation
- [Reference](docs/reference/index.md) - CLI flags, supported models, workload spec schema
- [CLAUDE.md](CLAUDE.md) - Codebase architecture and development guidelines
