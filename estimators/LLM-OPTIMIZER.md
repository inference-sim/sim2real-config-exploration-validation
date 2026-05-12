# LLM Optimizer

## Overview

llm-optimizer is a Python tool from BentoML for benchmarking and optimizing inference performance of open-source LLMs. It supports benchmarking across inference frameworks like SGLang and vLLM, applies SLO constraints, estimates performance theoretically, and provides interactive visualizations.

## Installation

Install from the repository:

```bash
pip install -e .
```

For development with linting and type checking:

```bash
pip install -e .[dev]
```

### Dependencies

Core dependencies (from pyproject.toml):
- aiohttp, requests, openai - HTTP client libraries
- psutil - System resource monitoring
- click - CLI framework
- nvidia-ml-py3 - GPU detection and monitoring
- numpy, scipy - Numerical computation
- huggingface_hub - Model metadata and access
- pydantic - Data validation

Python 3.9+ required.

### Environment Setup

For gated models, export your Hugging Face token:

```bash
export HF_TOKEN=<your token>
```

## Commands

### Main Command: `llm-optimizer`

Runs benchmarks across multiple configurations.

#### Key Flags

**Server Configuration:**
- `--server-cmd` - Custom server startup command
- `--model` - HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
- `--framework` - Framework to use: `sglang`, `vllm`, `max`
- `--server-args` - Server arguments with grid search syntax (multiple allowed)
- `--gpus` - Number of GPUs (auto-detected if not specified)
- `--host` - Server host (default: 127.0.0.1)
- `--port` - Server port (default: framework-dependent)

**Client Configuration:**
- `--client-args` - Client benchmark arguments with grid search syntax (multiple allowed)

**Execution Control:**
- `--dry-run` - Preview configurations without running
- `--continue` / `-c` - Resume from existing results, skip completed configs
- `--rest` - Rest time in seconds between runs (default: 10)
- `--mute-server` - Suppress server process stdout
- `--ready-endpoint` - Health check endpoint (default: /health)

**Output:**
- `--output-dir` - Directory for output files (default: results)
- `--output-json` - Path for single JSON file with all results
- `--constraints` - SLO constraints (e.g., 'ttft<300ms;itl:p95<50ms')

#### Argument Grid Search Syntax

Server and client arguments support grid search notation:
- List values: `key=[val1,val2,val3]`
- Range: `key=range(start,end,step)`
- Paired parameters: `key1*key2=[(a,b),(c,d)]`
- Multiple parameters: separate with semicolons

### Subcommand: `estimate`

Predicts latency, throughput, and concurrency limits without running full benchmarks.

#### Key Flags

**Required (unless --interactive):**
- `--model` - HuggingFace model ID
- `--input-len` - Input sequence length in tokens
- `--output-len` - Output sequence length in tokens

**Optional:**
- `--gpu` - GPU model (auto-detected if not specified; case-insensitive)
- `--num-gpus` - Number of GPUs (auto-detected if not specified)
- `--precision` - Model precision: `fp16`, `bf16`, `fp8` (auto-inferred from model if not specified)
- `--framework` - Framework: `sglang`, `vllm`, `both` (default: both)
- `--constraints` - SLO constraints (e.g., 'ttft:mean<300ms;itl:p95<50ms')
- `--target` - Optimization target: `throughput`, `latency` (default: throughput)
- `--dataset` - Dataset: `random`, `sharegpt` (default: random)
- `--interactive` - Run in interactive guided mode

#### Supported GPUs

H100, H200, A100, L20, L40, B100, B200

### Subcommand: `visualize`

Generates interactive HTML dashboards from benchmark results.

#### Key Flags

- `--data-file` - Path to JSON results file (required; comma-separated for multiple files)
- `--config` - Path to visualization config file (optional)
- `-o` / `--output` - Output HTML file path (default: pareto_llm_dashboard.html)
- `--serve` - Start HTTP server after generating HTML
- `--port` - Server port when using --serve (default: 8080)

## Example Invocations

### Benchmarking

SGLang with multiple TP/DP combinations:

```bash
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,4),(2,2),(4,1)];chunked_prefill_size=[2048,4096,8192]" \
  --client-args "max_concurrency=[50,100,200];num_prompts=1000" \
  --output-json sglang_results.json
```

vLLM with batch size tuning:

```bash
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size*data_parallel_size=[(1,2),(2,1)];max_num_batched_tokens=[4096,8192,16384]" \
  --client-args "max_concurrency=[32,64,128];num_prompts=1000;dataset_name=sharegpt" \
  --output-json vllm_results.json
```

Latency-optimized with constraints:

```bash
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size*data_parallel_size=[(1,2),(2,1)];max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[8,16,32];num_prompts=500" \
  --constraints "ttft<200ms;itl:p99<10ms" \
  --output-json latency_optimized.json
```

Custom server command:

```bash
llm-optimizer \
  --server-cmd "python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000" \
  --client-args "max_concurrency=[25,50,100];num_prompts=1000" \
  --host 0.0.0.0 \
  --port 30000
```

### Performance Estimation

Basic estimation:

```bash
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu A100 \
  --input-len 1024 \
  --output-len 512
```

With constraints and multi-GPU:

```bash
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512 \
  --gpu H100 \
  --num-gpus 4 \
  --constraints "ttft:mean<300ms;itl:p95<50ms"
```

Interactive mode:

```bash
llm-optimizer estimate --interactive
```

### Visualization

Single file:

```bash
llm-optimizer visualize --data-file results.json --port 8080
```

Multiple files with server:

```bash
llm-optimizer visualize --data-file "sglang_results.json,vllm_results.json" --serve --port 8080
```

## Key Parameters

### Framework-Specific Server Parameters

**SGLang:**
- `tp_size*dp_size` - Tensor/Data parallelism combinations
- `chunked_prefill_size` - Prefill chunk size for throughput
- `schedule_conservativeness` - Request scheduling aggressiveness
- `schedule_policy` - Scheduling policy (fcfs, priority)

**vLLM:**
- `tensor_parallel_size` - Tensor parallelism degree
- `pipeline_parallel_size` - Pipeline parallelism degree
- `data_parallel_size` - vLLM-internal data parallelism degree. This spawns multiple worker groups **within a single vLLM process/deployment** (`predefined/__init__.py:21` launches one `vllm serve` command; `server_utils.py:43-53` starts a single `Popen`). vLLM handles routing between DP groups internally. This is NOT equivalent to Kubernetes-style independent replicas (separate pods with separate KV caches routed by an external scheduler like llm-d). Other estimators (inference-sim, LLMServingSim, Vidur, AIConfigurator) model Kubernetes replicas; llm-optimizer models vLLM-internal DP.
- `max_num_batched_tokens` - Maximum batch size in tokens
- `max_num_seqs` - Maximum concurrent sequences

### Client Parameters

- `max_concurrency` - Maximum concurrent requests
- `num_prompts` - Total number of requests to send
- `dataset_name` - Dataset: `sharegpt`, `random`
- `random_input_len` / `random_output_len` - Random sequence lengths

### Constraint Syntax

Supports mean, median, p95, p99 statistics:

```bash
# Time to first token
--constraints "ttft<300ms"           # Mean TTFT under 300ms
--constraints "ttft:median<200ms"    # Median TTFT under 200ms
--constraints "ttft:p95<500ms"       # 95th percentile under 500ms

# Inter-token latency
--constraints "itl:mean<20ms"        # Mean ITL under 20ms
--constraints "itl:p99<50ms"         # 99th percentile under 50ms

# End-to-end latency
--constraints "e2e_latency:p95<2s"   # 95th percentile under 2s

# Combined
--constraints "ttft:median<300ms;itl:p95<10ms;e2e_latency:p95<2s"
```

## Development

Code formatting and linting:

```bash
ruff format
ruff check
```

Type checking:

```bash
mypy src/
```

## Config Search / Exploration

**Yes, llm-optimizer natively supports configuration space search and exploration.** It can automatically sweep over multiple configurations to find the best one that meets SLO compliance targets (latency, throughput, etc.).

### Grid Search Functionality

The main `llm-optimizer` command supports grid search over both server and client parameters via the `--server-args` and `--client-args` flags. The tool automatically generates all combinations and runs benchmarks for each.

**Grid Search Syntax:**

- **List values:** `key=[val1,val2,val3]` - Tests each value
- **Range:** `key=range(start,end,step)` - Generates arithmetic sequence
- **Paired parameters:** `key1*key2=[(a,b),(c,d)]` - Tests coupled parameter pairs
- **Multiple parameters:** Separate with semicolons to create Cartesian product

**Example - 27 configurations (3x3x3):**

```bash
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,4),(2,2),(4,1)];chunked_prefill_size=[2048,4096,8192]" \
  --client-args "max_concurrency=[50,100,200];num_prompts=1000" \
  --output-json sglang_results.json
```

This tests 3 TP/DP combinations x 3 prefill sizes = 9 server configs against 3 concurrency values.

**Complex parameter grid with range:**

```bash
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,8),(2,4),(4,2)];schedule_conservativeness=[0.3,0.6,1.0];chunked_prefill_size=range(2048,8193,2048)" \
  --client-args "max_concurrency=range(50,201,50);request_rate=[10,20,50]" \
  --gpus 8 \
  --output-json complex_benchmark.json
```

### SLO-Constrained Search

The `--constraints` flag filters results to only configurations that meet SLO requirements for latency and throughput metrics. The tool benchmarks all configurations but highlights/filters only SLO-compliant ones.

**Constraint syntax:**

```bash
# Single constraint
--constraints "ttft<300ms"

# Multiple constraints (all must be satisfied)
--constraints "ttft:median<300ms;itl:p95<10ms;e2e_latency:p95<2s"
```

Supported metrics:
- `ttft` - Time to first token
- `itl` - Inter-token latency
- `e2e_latency` - End-to-end latency

Supported statistics: `mean` (default), `median`, `p95`, `p99`

**Example - Latency-optimized search with constraints:**

```bash
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size*data_parallel_size=[(1,2),(2,1)];max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[8,16,32];num_prompts=500" \
  --constraints "ttft<200ms;itl:p99<10ms" \
  --output-json latency_optimized.json
```

The results JSON includes a `best_configurations` section identifying:
- Best input/output throughput (unconstrained)
- Best input/output throughput meeting constraints
- Full test results for all configurations

### Auto-Tuning via Performance Estimation

The `estimate` subcommand performs theoretical optimization without benchmarking. It uses roofline analysis and GPU specs to suggest optimal configurations.

**Key features:**
- Automatically calculates optimal batch sizes, prefill chunk sizes, concurrency levels
- Generates framework-specific server commands
- Supports constraint-aware recommendations
- Works for both throughput and latency optimization targets

**Example with constraints:**

```bash
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512 \
  --gpu H100 \
  --num-gpus 4 \
  --constraints "ttft:mean<300ms;itl:p95<50ms" \
  --target throughput \
  --framework both
```

This generates ready-to-run commands for vLLM and SGLang with parameters tuned for the specified hardware and constraints.

### Implementation Details

Configuration generation is handled by:
- `src/llm_optimizer/args.py` - Parses grid search syntax and generates Cartesian products
- `src/llm_optimizer/tuning/strategy.py` - Framework-specific optimization strategies
- `src/llm_optimizer/tuning/generation.py` - Generates parameter ranges and base configurations
- `src/llm_optimizer/performance.py` - SLO constraint parsing and validation

The tuning system supports:
- Conservative, aggressive, and memory-efficient configuration presets
- GPU-aware parameter calculation based on VRAM, bandwidth, and TFLOPS
- TP/DP combination generation
- Roofline-based arithmetic intensity analysis for compute vs memory-bound detection
