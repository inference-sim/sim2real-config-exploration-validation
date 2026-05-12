# LLMServingSim

## What it does

LLMServingSim is a cycle-level simulator for LLM serving infrastructure. It combines a Python frontend (mirroring vLLM's continuous-batching scheduler) with the ASTRA-Sim C++ analytical network backend, driven by per-hardware latency data captured via a vLLM-based layerwise profiler. It supports heterogeneous accelerators, disaggregated memory tiers (CPU/CXL/PIM), MoE routing, and multi-instance parallelism (TP/PP/EP/DP).

## Installation

### Prerequisites
- Docker (recommended)
- Or: Python 3.8+, vLLM v0.19.0, ASTRA-Sim dependencies

### Setup via Docker

1. Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
```

2. Launch simulator container:
```bash
./scripts/docker-sim.sh
```

3. Build ASTRA-Sim + Chakra:
```bash
./scripts/compile.sh
```

The simulator container (`astrasim/tutorial-micro2024`) auto-installs Python dependencies: `pyyaml`, `pyinstrument`, `transformers`, `datasets`, `msgspec`, `scikit-learn`, `xgboost`, `matplotlib`, `pandas`, `numpy`.

### Setup for profiling/benchmarking

Launch the vLLM container (includes vLLM v0.19.0 + dependencies):
```bash
./scripts/docker-vllm.sh
```

## Available Commands

### 1. Simulator (`python -m serving`)

Main entry point for running simulations.

**Key arguments:**

- `--cluster-config PATH` - Path to cluster config JSON (defines topology, hardware, memory hierarchy)
- `--dataset PATH` - Path to .jsonl request trace file
- `--output PATH` - Path for per-request CSV output with latency metrics (TTFT, TPOT, ITL)
- `--num-reqs N` - Number of requests/sessions to load from dataset (0 = all)

**Scheduling & batching:**

- `--max-num-seqs N` - Maximum sequences per batch (default: 128)
- `--max-num-batched-tokens N` - Maximum tokens per iteration (default: 2048)
- `--long-prefill-token-threshold N` - Per-request token cap per iteration for chunked prefill (default: 0). When >0, limits how many tokens a single prefill request consumes per scheduling step. When 0, no per-request cap is applied; a single prefill can consume the entire `max-num-batched-tokens` budget (after decode requests take their 1-token-each share). The guard is `if 0 < threshold < remaining` (`scheduler.py` lines 134, 399), so 0 means uncapped, not disabled. Only used when `--enable-chunked-prefill` is active.
- `--block-size N` - KV cache block size in tokens (default: 16)
- `--prioritize-prefill` - Prioritize prefill over decode requests

**Model configuration:**

- `--dtype {float16,bfloat16,float32,fp8,int8}` - Model weight precision (defaults to model config's torch_dtype)
- `--kv-cache-dtype {auto,fp8}` - KV cache data type (default: auto)
- `--skip-prefill` - Skip prefill phase, decode only

**Routing policies:**

- `--request-routing-policy {LOAD,RR,RAND,CUSTOM}` - Request routing across instances (default: LOAD)
- `--expert-routing-policy {BALANCED,RR,RAND,CUSTOM}` - MoE expert token routing (default: BALANCED)

**Feature flags:**

- `--enable-prefix-caching` / `--no-enable-prefix-caching` - Prefix caching via RadixAttention (default: enabled)
- `--enable-chunked-prefill` / `--no-enable-chunked-prefill` - Split long prefills across iterations (default: enabled)
- `--enable-prefix-sharing` - Second-tier prefix cache pooling across instances
- `--prefix-storage {None,CPU,CXL}` - Storage medium for second-tier prefix cache
- `--enable-local-offloading` - Weight offloading to local NPU memory
- `--enable-attn-offloading` - Attention offloading to PIM devices
- `--enable-sub-batch-interleaving` - Overlap XPU and PIM computation (requires --enable-attn-offloading)
- `--enable-block-copy` / `--no-enable-block-copy` - Replay block traces across layers for MoE (default: enabled)

**Logging:**

- `--log-level {WARNING,INFO,DEBUG}` - Verbosity level (default: WARNING)
- `--log-interval SECONDS` - Interval between throughput/memory logs (default: 1.0)

**Network backend:**

- `--network-backend {analytical,ns3}` - Network simulation backend (default: analytical; ns3 is WIP)

**Example invocations:**

```bash
# Basic single instance simulation
python -m serving --cluster-config configs/cluster/single_node_single_instance.json \
    --dtype float16 --block-size 16 \
    --dataset workloads/example_trace.jsonl --output outputs/example_run.csv \
    --num-reqs 10

# Multi-instance with prefix cache pooling
python -m serving --cluster-config configs/cluster/single_node_multi_instance.json \
    --dtype float16 --block-size 16 \
    --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
    --dataset workloads/example_trace.jsonl --output outputs/prefix_pool_run.csv \
    --num-reqs 10

# MoE with DP+EP and agentic sessions
python -m serving --cluster-config configs/cluster/single_node_moe_dp_ep_instance.json \
    --dtype float16 --block-size 16 \
    --dataset workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl \
    --output outputs/moe_dp_ep_run.csv \
    --num-reqs 1

# PIM with sub-batch interleaving
python -m serving --cluster-config configs/cluster/single_node_pim_instance.json \
    --dtype float16 --block-size 16 \
    --enable-attn-offloading --enable-sub-batch-interleaving \
    --dataset workloads/example_trace.jsonl --output outputs/pim_sub_batch_run.csv \
    --num-reqs 10
```

See `serving/run.sh` for additional examples.

### 2. Profiler (`python -m profiler`)

Captures per-hardware latency data via vLLM's layerwise profiling.

**How profiling works:**

The profiler uses vLLM's built-in `layerwise_profile()` context manager, which records CUDA kernel events (`cuda_time_us`, `invocations`) per `nn.Module`. Per-invocation latency = `cuda_time_us / invocations` (`hooks/timings.py:131`).

- **Measurement protocol per shot:** 1 warmup forward (discarded), then N timed forwards (default N=3, via `--measurement-iterations`) inside `layerwise_profile`, averaged (`hooks/extension.py:85-127`). N=3 cuts DVFS/boost jitter from ~15-25% to ~5%.
- **Single-GPU TP emulation:** Every TP degree is profiled on **one GPU** by shrinking the model config via `hf_overrides` (dividing `hidden_size`, `num_attention_heads`, etc. by TP in `engine.py:108-198`). This makes per-rank kernel shapes match real TP=N deployment. Profiling Llama-3-70B at TP=4 does not need 4 GPUs.
- **1-layer model:** The model is shrunk to `num_hidden_layers: 1`. Each forward still passes through embedding, attention, MLP, lm_head, sampler. `layerwise_profile` decomposes per-layer timings. Since decoder layers are architecturally identical, one layer represents all.
- **Synthetic shots:** The profiler builds fake `SchedulerOutput` objects (`hooks/batch.py:120-213`) that bypass vLLM's scheduler, controlling batch shapes precisely (prefill chunk size, KV history, decode count). Uses `num_computed_tokens=history` to make vLLM treat tokens as "already cached".
- **Shot execution:** Shots are sent to the worker via `llm.collective_rpc("fire", ...)`, which calls `model_runner.execute_model()` inside `layerwise_profile()` context (`core/runner.py:60-143`).

**CSV categories and sweep axes:**

| CSV | What it measures | Sweep axes | Lookup method in simulator |
|-----|-----------------|------------|---------------------------|
| `dense.csv` | Token-linear layers (qkv_proj, gate_up_proj, down_proj, layernorm, etc.) | `tokens`: 1 to max_batched_tokens (fine-grained at low end: 1-15 by 1, 16-63 by 4, 64+ by 16; `categories.py:87`) | 1D linear interpolation on tokens (`trace_generator.py:507`) |
| `per_sequence.csv` | Sequence-linear layers (lm_head, sampler) that scale with batch cardinality, not token count | `sequences`: 1 to max_num_seqs (same grid as dense) | 1D linear interpolation on sequences (`trace_generator.py:518`) |
| `attention.csv` | Unified 4D attention grid covering pure-prefill, pure-decode, and mixed batches | `prefill_chunk` (0 or geometric 16..max_batched_tokens), `kv_prefill` (0 or geometric 16..max_kv), `n_decode` (0 or geometric 1..max_seqs), `kv_decode` (0 or geometric 16..max_kv) (`categories.py:297-442`) | 4D: nearest-neighbor on (prefill_chunk, n_decode), bilinear on (kv_prefill, kv_decode), with skew alpha correction (`trace_generator.py:713-758`) |
| `moe.csv` | MoE block (gate + grouped experts); only at tp=1 | `tokens` (power-of-2 grid), `activated_experts` (power-of-2 from top_k to min(num_experts, tokens*top_k)) (`categories.py:464-527`) | 2D lookup on (tokens, activated_experts) (`trace_generator.py:795`) |
| `skew.csv` | FlashAttention varlen kernel cost shift with non-uniform KV distributions | `n` (decode reqs), `ratio` (fraction "big"), `pc` (prefill chunk), `kp` (prefill KV), `kvs` (small KV), `skew` (kv_big/kvs ratio) (`skew.py:1-499`). Each case measures 3 latencies: t_mean (uniform mean KV), t_max (uniform max KV), t_skew (actual bimodal mix). | N/A (input to skew_fit.csv) |
| `skew_fit.csv` | Fitted 5-axis weighted-LS alpha table from skew.csv | Bucketized grid over `pc`, `n`, `skew_rate`, `kv_big`, `kp` (`fit_alpha.py`). Alpha in [0,1]: `t_predicted = t_mean + alpha * (t_max - t_mean)` | Used by `_lookup_attention_with_skew()` to blend mean/max attention lookups (`trace_generator.py:713`) |

**How the simulator consumes profiling data:**

CSVs are loaded by `_load_perf_db()` (`trace_generator.py:325`), converting `time_us` to `latency_ns` (x1000). Per-layer lookups produce nanosecond latencies for a given batch shape, which are emitted as text traces, converted to Chakra protobuf, and fed to ASTRA-Sim's network backend.

**Subcommands:**

- `profile <model>` - Full sweep across all TP degrees and categories
- `slice <model>` - Refresh one (TP, category) pair

**Required arguments:**

- `<model>` - Model path (HF-style `org/name`) under `configs/model/`
- `--hardware NAME` - Hardware identifier (e.g., H100, A6000)

**Optional arguments:**

- `--tp DEGREES` - Comma-separated TP degrees to sweep (e.g., "1,2,4"; must include 1; default: "1")
- `--variant NAME` - Output folder label (default: auto-derived from dtypes, e.g., "bf16", "bf16-kvfp8")
- `--dtype {bfloat16,float16,float32,fp8}` - Model weight dtype (default: from model config's torch_dtype)
- `--kv-cache-dtype {auto,fp8,fp16,bf16}` - KV cache dtype (default: auto)
- `--max-num-batched-tokens N` - vLLM's max batched tokens (default: 2048)
- `--max-num-seqs N` - vLLM's max sequences (default: 256)
- `--attention-max-kv N` - Upper bound for kv axes (default: 16384)
- `--attention-chunk-factor F` - Geometric factor for prefill_chunk axis (default: 2.0)
- `--attention-kv-factor F` - Geometric factor for kv axes (default: 2.0)
- `--measurement-iterations N` - Timed forwards per shot (default: 3)
- `--skip-skew` - Skip heterogeneous-decode skew profiling
- `--skew-n-factor F` / `--skew-pc-factor F` / `--skew-kp-factor F` / `--skew-kvs-factor F` - Geometric factors for skew sweep axes (default: 2.0)
- `--force` - Wipe existing CSVs and re-profile from scratch (default: resume mode)
- `--silent` / `--verbose` - Verbosity shortcuts (default: INFO)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Explicit log level override

**Output structure:**

```
profiler/perf/<hardware>/<model>/<variant>/
  meta.yaml                    # Profiler version, engine config, sweep specs
  tp<N>/
    dense.csv                  # Layer, tokens, time_us
    per_sequence.csv           # Layer, sequences, time_us
    attention.csv              # prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
    moe.csv                    # tokens, activated_experts, time_us (MoE only)
    skew.csv                   # Raw heterogeneous-decode shots
    skew_fit.csv               # Fitted per-bucket alpha table
```

**Example invocations:**

Edit `profiler/profile.sh` with desired MODEL/HARDWARE/TP_DEGREES, then:

```bash
# From repo root, inside vLLM container
./profiler/profile.sh
```

Or invoke directly:

```bash
python -m profiler profile Qwen/Qwen3-32B --hardware H100 --tp 1,2,4 \
    --max-num-seqs 256 --max-num-batched-tokens 2048 \
    --measurement-iterations 3
```

For TP refresh or category-specific updates:

```bash
python -m profiler slice Qwen/Qwen3-32B --hardware H100 --tp-refresh 2 --group attention
```

### 3. Benchmark (`python -m bench`)

Runs vLLM end-to-end benchmarks and validates simulator accuracy.

**Subcommands:**

- `run` - Execute vLLM benchmark and record results
- `validate` - Compare bench run against simulator output

**`bench run` key arguments:**

- `--model NAME` - HuggingFace model name
- `--dataset PATH` - Path to .jsonl workload file
- `--run-id ID` - Identifier for this run (creates `bench/results/<run_id>/`)
- `--tensor-parallel-size N` - TP degree (default: 1)
- `--dtype {float16,bfloat16,float32}` - Model weight dtype
- `--max-num-seqs N` - vLLM max sequences
- `--max-num-batched-tokens N` - vLLM max batched tokens
- `--num-reqs N` - Number of requests to load (0 = all)

**`bench validate` key arguments:**

- `--bench-run ID` - Bench run ID (path to `bench/results/<run_id>/`)
- `--sim-output PATH` - Simulator output CSV path
- `--plot-dir PATH` - Directory for validation plots (optional)

**Example invocations:**

```bash
# Run vLLM benchmark
python -m bench run \
    --model meta-llama/Llama-3.1-8B \
    --dataset workloads/example_trace.jsonl \
    --run-id llama31_8b_h100 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --num-reqs 100

# Validate against simulator
python -m bench validate \
    --bench-run bench/results/llama31_8b_h100 \
    --sim-output outputs/example_run.csv \
    --plot-dir outputs/validation_plots/
```

### 4. Workload Generators (`python -m workloads.generators`)

Convert ShareGPT and other datasets to LLMServingSim's JSONL format.

**Location:** `workloads/generators/`

**Example datasets:** `workloads/example_trace.jsonl`, `workloads/swe-bench-*.jsonl`

**JSONL format:**

Flat requests:
```json
{"request_id": "1", "arrival_time": 0.0, "input_toks": 512, "output_toks": 128}
```

Agentic sessions (SWE-bench, tool-calling agents):
```json
{
  "session_id": "s1",
  "arrival_time": 0.0,
  "sub_requests": [
    {"input_toks": 100, "output_toks": 50, "tool_duration_ns": 5000000},
    {"input_toks": 80, "output_toks": 40, "tool_duration_ns": 0}
  ]
}
```

## Configuration Files

### Cluster configs (`configs/cluster/`)

Define hardware topology, instance layout, parallelism, and memory hierarchy.

**Instance Architecture:**

Each entry in the `instances[]` array represents an **independent Kubernetes-style serving replica** (analogous to a separate vLLM pod). Each instance has:
- Its own `Scheduler` object with separate request queue (`request`, `inflight`, `done` lists)
- Its own `MemoryModel` managing KV cache and GPU/CPU memory
- Independent scheduling decisions (batch formation, preemption, chunked prefill)

Requests are distributed across instances via `--request-routing-policy` (LOAD/RR/RAND). The `Router` selects a target instance at arrival time and adds the request to that instance's scheduler queue.

**Key fields:**

- `instances[].hardware` - Must match a directory under `profiler/perf/<hardware>/`
- `instances[].model_name` - Must match a config under `configs/model/<model>.json`
- `instances[].tp_size` - Tensor parallel degree (within a single instance)
- `instances[].pp_size` - Pipeline parallel degree (optional, default: 1)
- `instances[].ep_size` - Expert parallel degree for MoE (optional, default: tp_size for MoE, 1 for dense)
- `instances[].dp_group` - Data parallel group ID string (see below)
- `instances[].npu_mem.mem_bw` - NPU memory bandwidth (GB/s)
- `instances[].cpu_mem.mem_bw` - CPU memory bandwidth (GB/s)
- `instances[].link_bw` - Inter-node bandwidth (GB/s)
- `instances[].link_latency` - Inter-node link latency (ns)

**dp_group semantics (MoE expert sharding across replicas):**

For MoE models, instances with the same `dp_group` string form a **data parallel group** that shares experts via ALLTOALL collectives:
- `ep_size` in the config is the **total EP degree across the DP group**
- Each instance handles `local_ep = ep_size / dp_group_size` experts
- Example: 2 instances with `ep_size=2, dp_group="A"` means each instance processes 1 expert, synchronized via cross-instance ALLTOALL
- DP group members must have identical `tp_size` and `ep_size` (`config_builder.py` validates this)
- Batches are synchronized: all DP group members defer trace generation until every member has scheduled, then pad to uniform `max_total_len` (vLLM CUDA-graph DP padding behavior)

`dp_group` represents **data parallelism via replica sharding**, not vLLM-internal worker groups. It is distinct from TP (intra-instance parallelism via ALLREDUCE) and EP (cross-instance expert sharding via ALLTOALL).

**Examples:** `configs/cluster/single_node_single_instance.json`, `configs/cluster/single_node_moe_dp_ep_instance.json`

### Model configs (`configs/model/`)

Subset of HuggingFace `config.json` containing fields the simulator needs.

**Required fields:** `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`, `intermediate_size`, `vocab_size`

**Optional fields:** `head_dim`, `num_local_experts`, `num_experts_per_tok`, `torch_dtype`

**Examples:** `configs/model/meta-llama/Llama-3.1-8B.json`, `configs/model/Qwen/Qwen3-32B.json`

## Working Directory

The simulator changes cwd to `astra-sim/` on startup. All relative paths in the codebase resolve from there. Paths to `configs/`, `workloads/`, `profiler/` are prefixed with `../` in code.

## Configuration Space Search / Optimization

**LLMServingSim does NOT natively support automatic configuration space exploration or optimization.**

The simulator operates as a single-shot evaluator: you provide a cluster configuration (hardware, parallelism, memory hierarchy) and workload, and it returns latency/throughput metrics for that specific configuration. To find optimal configurations that meet SLO targets (latency, throughput, cost), you must:

1. **Manually iterate** - Edit cluster config JSONs and re-run simulations
2. **Write custom sweep scripts** - Create bash/Python scripts that loop over parameter combinations and invoke `python -m serving` for each
3. **Integrate external optimizers** - Wrap the simulator in your own Bayesian optimization, grid search, or genetic algorithm framework

The artifact evaluation scripts (`evaluation/figure_*.sh` on the `ispass26-artifact` branch) demonstrate manual parameter sweeps - each script runs the simulator multiple times with different configurations and aggregates results, but this is for research figure generation, not automated optimization.

**What you get:**
- Fast cycle-accurate simulation for a given config
- Per-request latency metrics (TTFT, TPOT, ITL) in CSV output
- Validation tools to check sim accuracy against real vLLM (`python -m bench validate`)

**What you don't get:**
- Built-in SLO constraint checking
- Automatic hyperparameter tuning
- Configuration recommendation or ranking
- Cost/performance Pareto frontier search
- Integration with Ray Tune, Optuna, or similar frameworks

If you need configuration optimization, you must build it on top of LLMServingSim's evaluation capabilities.

## Documentation

- **Website:** https://llmservingsim.ai
- **Full docs:** https://llmservingsim.ai/docs/getting-started/overview
- **Repository:** https://github.com/casys-kaist/LLMServingSim
- **Contributor guide:** https://llmservingsim.ai/docs/contributor/welcome
