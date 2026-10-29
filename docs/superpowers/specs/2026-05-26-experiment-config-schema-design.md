# Experiment Config Schema Design

## Problem

The experiment config (`experiments/default_config.yaml`) was too verbose. Shared parameters were repeated per tool, and it was unclear how conceptually-equivalent fields (e.g., hardware as `H100` vs `h100_sxm`) mapped across tools. A reader wanting to run a single tool standalone couldn't easily construct the invocation from the config, and the full-experiment view had unnecessary redundancy.

## Design

The config has four sections:

1. **workload** - references a workload file and declares the shape (ISL/OSL distributions, arrival, num_requests, seed)
2. **concept_map** - the central registry of all experiment parameters (fixed and swept), showing how each tool consumes them
3. **tools** - per-tool search strategy and hyperparameters only
4. **analysis** - post-processing settings

### Concept Map

The concept_map is the core innovation. Each entry represents a single experimental concept with:

- `value:` - the canonical value (scalar = fixed parameter, list = search dimension)
- Per-tool child entries showing `arg:` (the tool's native argument name) and optionally `value:` (override when the tool needs a different representation)

**Reading rules:**
- If a tool entry has no `value:`, it receives the top-level `value:`
- If a tool entry has a `value:`, that overrides the canonical for that tool
- If a tool entry has only `note:`, the concept doesn't map to a direct argument (e.g., embedded in a file, handled internally)
- Tools not listed under a concept don't use that parameter

**Two mapping types captured:**
1. Same value, different arg name (e.g., `output_seq_len`: value 256, but arg is `osl` for aiconfigurator, `--output-len` for llm-optimizer)
2. Different value AND arg name (e.g., `hardware`: canonical `H100_SXM_80GB` becomes `H100` for blis, `h100_sxm` for aiconfigurator)

### Use Cases

- **Run full experiment**: shared values declared once in concept_map, `tools:` section drives the orchestrator
- **Run single tool**: reader finds the tool's entries across concept_map, reads the `arg:` and `value:` fields, constructs the CLI invocation

## Schema

```yaml
experiment:
  name: config-exploration-llama3-8b
  description: Llama 3.1 8B on H100, SLO 300ms TTFT

workload:
  file: workloads/chatbot.yaml
  input_seq_len: {mean: 512, max: 2048}
  output_seq_len: {mean: 256, max: 1024}
  arrival: poisson
  num_requests: 10000
  seed: 42

concept_map:
  # ─── Model & Hardware ───────────────────────────────────────────────────────
  model:
    value: meta-llama/Llama-3.1-8B
    blis: {arg: "--model"}
    aiconfigurator: {arg: "model_path"}
    llm-optimizer: {arg: "--model"}

  hardware:
    value: H100_SXM_80GB
    blis: {arg: "--hardware", value: "H100"}
    aiconfigurator: {arg: "system", value: "h100_sxm"}
    llm-optimizer: {arg: "--gpu", value: "H100"}

  # ─── Workload & SLO ─────────────────────────────────────────────────────────
  input_seq_len:
    value: 512
    blis: {note: "embedded in workload preset/file"}
    aiconfigurator: {arg: "isl"}
    llm-optimizer: {arg: "--input-len"}

  output_seq_len:
    value: 256
    blis: {note: "embedded in workload preset/file"}
    aiconfigurator: {arg: "osl"}
    llm-optimizer: {arg: "--output-len"}

  slo_ttft_ms:
    value: 300
    blis: {note: "implicit in fitness-weights scoring"}
    aiconfigurator: {arg: "ttft"}
    llm-optimizer: {arg: "--constraints", value: "ttft:mean<300ms"}

  seed:
    value: 42
    blis: {arg: "--seed"}

  # ─── Fixed Parameters ───────────────────────────────────────────────────────
  max_gpus:
    value: 8
    aiconfigurator: {arg: "total_gpus", note: "iterates 1..value"}
    llm-optimizer: {arg: "--num-gpus", note: "iterates unique counts"}

  gpu_cost_per_hour:
    value: 3.20

  gpu_memory_utilization:
    value: 0.9
    blis: {arg: "--gpu-memory-utilization"}
    llm-optimizer: {note: "internal default"}

  model_dtype:
    value: bfloat16
    blis: {note: "default in trained-physics model"}
    llm-optimizer: {arg: "--dtype"}

  kv_cache_dtype:
    value: auto
    llm-optimizer: {arg: "--kv-cache-dtype"}

  latency_model:
    value: trained-physics
    blis: {arg: "--latency-model"}

  fitness_weights:
    value: "throughput:0.4,p99_ttft:0.3,p99_e2e:0.3"
    blis: {arg: "--fitness-weights"}

  framework:
    value: vllm
    aiconfigurator: {arg: "backend"}
    llm-optimizer: {arg: "--framework"}

  optimization_target:
    value: throughput
    llm-optimizer: {arg: "--target"}

  # ─── Topology (search dimensions) ──────────────────────────────────────────
  tp:
    value: [1, 2, 4, 8]
    blis: {arg: "--tp"}
    aiconfigurator: {note: "enumerated internally"}
    llm-optimizer: {note: "determined internally"}

  pp:
    value: [1, 2, 4]
    blis: {note: "N/A (no PP support, fixed at 1)"}
    aiconfigurator: {note: "enumerated internally"}
    llm-optimizer: {note: "determined internally"}

  deployment_replicas:
    value: "1..floor(max_gpus / (tp * pp))"
    blis: {arg: "--num-instances", note: "1..floor(max_gpus / tp)"}
    aiconfigurator: {note: "enumerated internally"}
    llm-optimizer: {note: "N/A (always 1)"}

  data_parallel_size:
    value: "1..floor(max_gpus / (tp * pp))"
    llm-optimizer: {arg: "--data-parallel-size"}

  # ─── Batching (search dimensions) ──────────────────────────────────────────
  max_num_seqs:
    value: [32, 64, 128, 256, 512]
    blis: {arg: "--max-num-running-reqs"}
    aiconfigurator: {note: "batch size, swept internally (~80-100 values)"}
    llm-optimizer: {arg: "--max-num-seqs"}

  max_batched_tokens:
    value: [2048, 4096, 8192]
    blis: {arg: "--max-num-scheduled-tokens"}
    llm-optimizer: {arg: "--max-num-batched-tokens"}

  enable_chunked_prefill:
    value: [true, false]
    llm-optimizer: {arg: "--enable-chunked-prefill"}

  chunked_prefill_threshold:
    value: [0, 1024, 2048, 4096]
    blis: {arg: "--long-prefill-token-threshold", note: "0=disabled, >0=chunk size"}

  block_size:
    value: [16, 32]
    blis: {arg: "--block-size-in-tokens"}
    llm-optimizer: {arg: "--block-size"}

  prefix_caching:
    value: [true, false]
    blis: {note: "always active; routing scorers control prefix-aware behavior"}
    llm-optimizer: {arg: "--enable-prefix-caching"}

  max_concurrency:
    value: [32, 64, 128, 256]
    llm-optimizer: {arg: "--max-concurrency"}

  # ─── Scheduling & Routing (search dimensions) ──────────────────────────────
  scheduler:
    value: [fcfs, priority-fcfs, sjf, reverse-priority]
    blis: {arg: "--scheduler"}

  routing:
    value: [round-robin, least-loaded]
    blis: {arg: "--routing-policy", note: "applies when deployment_replicas > 1"}

  routing_weighted:
    value:
      - "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
      - "queue-depth:1,kv-utilization:1"
      - "precise-prefix-cache:2,load-balance:1"
      - "vllm-dp:1"
    blis: {arg: "--routing-scorers", note: "used with --routing-policy weighted"}

  admission:
    value: [always-admit, tier-shed]
    blis: {arg: "--admission-policy", note: "applies when deployment_replicas > 1"}

  preemption:
    value: [fcfs, priority]
    blis: {arg: "--preemption-policy"}

tools:
  blis:
    search:
      strategy: adaptive-hierarchical
      budget: 20
      fixed_rate: 200
      tp_candidates: lean
      phase1_k: 1
      phase1_profiles: bracket
      skip_phase2: false

  aiconfigurator:
    search:
      strategy: native

  llm-optimizer:
    search:
      strategy: native

analysis:
  top_k: 3
  min_throughput_tok_s: 200
```

## Implementation Notes

- The concept_map is primarily a documentation/reference structure. The Python runners compute tool-specific values from the shared fields using lookup tables (hardware mapping) and templates (constraint string formatting).
- `run_all.py` reads `concept_map` entries to extract canonical values where needed (e.g., `concept_map.max_gpus.value` for GPU budget).
- Search dimensions (list values) are consumed by the config generators (`blis_configs.py`, `llm_optimizer_configs.py`) to build the sweep space.
- Tools with `strategy: native` (aiconfigurator, llm-optimizer) handle their own search internally; the orchestrator just passes fixed parameters and collects results.
