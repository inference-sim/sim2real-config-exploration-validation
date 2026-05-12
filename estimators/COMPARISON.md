# Estimator Comparison

Side-by-side comparison of capabilities across all five estimators. Focuses on differences and unique features rather than shared parameters.

## At a Glance

| Capability | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Language | Go | Python + C++ | Python | Python | Python |
| Simulation type | Discrete-event | Cycle-level | Analytical model | Event-driven | Live benchmarking |
| GPU required | No | No (profiling needs GPU) | No | No (profiling needs GPU) | Yes (runs real server) |
| Native config search | No | No | Yes | Yes | Yes |
| Output format | JSON metrics | CSV per-request | DataFrame / YAML configs | W&B + CSV + Chrome trace | JSON results |

## SLO-Aware Search

This is the key differentiator for the paper's config exploration experiments.

| | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|
| **SLO metric** | TTFT, TPOT (independently), or combined request latency (TTFT + TPOT x tokens) | Scheduling delay only (time waiting in queue before execution begins) | TTFT, ITL, E2E latency |
| **Tail latency support** | No. Thresholds are single-point analytical predictions, not percentiles. The model predicts one TTFT and one TPOT per config at given batch size/sequence length. | Yes. Configurable quantile (e.g., P99) via `--scheduling-delay-slo-quantile`. Evaluates the distribution from simulated requests. | Yes. Supports mean, median, P95, P99 per metric. Measures from actual benchmark distribution. |
| **What it filters** | Configs whose predicted TTFT > threshold or predicted TPOT > threshold are eliminated | Binary search finds max QPS where scheduling delay at chosen quantile stays under threshold | Benchmark results where measured stats exceed constraints are excluded from "best" selection |
| **Constraint syntax** | `--ttft 300 --tpot 10` (ms, fixed point) | `--scheduling-delay-slo-value 5.0 --scheduling-delay-slo-quantile 0.99` | `--constraints "ttft:p95<300ms;itl:p99<50ms;e2e_latency:median<2s"` |
| **Limitation** | No distribution awareness; assumes steady-state. Cannot express "P99 TTFT < X". | Only constrains scheduling delay, not TTFT/TPOT/E2E directly (though scheduling delay correlates). | Requires real GPUs and server for every config point. |

**inference-sim and LLMServingSim** produce per-request latency distributions (TTFT, ITL, E2E) but have no built-in constraint filtering or search. They output the raw data; SLO checking must be done externally.

## Configuration Dimensions (All Arguments)

### Model Selection

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Model spec | `--model` (HF ID) | `--cluster-config` (JSON ref to `configs/model/`) | `--model-path` (HF ID) | `--replica_config_model_name` | `--model` (HF ID) |
| Dtype | N/A (uses roofline) | `--dtype` (float16, bfloat16, fp8, int8) | N/A (FP8 indicated in model name) | N/A (from profiling data) | `--precision` (fp16, bf16, fp8) |
| KV cache dtype | N/A | `--kv-cache-dtype` (auto, fp8) | N/A | N/A | N/A |

### Hardware and Parallelism

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| GPU type | `--hardware` (H100, A100, etc.) | cluster config JSON `hardware` field | `--system` (h200_sxm, h100_sxm, b200_sxm, gb200, a100_sxm, l40s, gb300) | `--replica_config_device` (a100, h100) | `--gpu` (H100, H200, A100, L20, L40, B100, B200) |
| Tensor parallelism | `--tp` | cluster config `tp_size` | `--tp-size` (estimate) / auto-swept (default) | `--replica_config_tensor_parallel_size` | `tp_size` in `--server-args` |
| Pipeline parallelism | N/A | cluster config `pp_size` | `--pp-size` (estimate) / auto-swept (default) | `--replica_config_num_pipeline_stages` | `pipeline_parallel_size` in `--server-args` |
| Data parallelism | N/A | cluster config `dp_group` (MoE expert sharding: instances with same ID sync via ALLTOALL) | N/A (uses replicas) | N/A | `data_parallel_size` in `--server-args` (**vLLM-internal DP**; see note below) |
| Expert parallelism | N/A | cluster config `ep_size` | N/A | N/A | N/A |
| Total GPUs / instances | `--num-instances` (K8s replicas) | cluster config `instances[]` array length (K8s replicas) | `--total-gpus` (derives K8s replicas) | `--cluster_config_num_replicas` (K8s replicas) | `--gpus` / `--num-gpus` (single deployment) |

#### Kubernetes Replicas vs. vLLM-Internal Data Parallelism

The multi-instance concepts across tools represent two fundamentally different architectures:

**Kubernetes replicas** (inference-sim, LLMServingSim, AIConfigurator, Vidur): Independent vLLM pods, each with its own process, KV cache, scheduler, and request queue. An external scheduler (llm-d inference scheduler) routes requests across pods. This is the deployment model for llm-d validation.

| Tool | Parameter | Fidelity |
|---|---|---|
| inference-sim | `--num-instances` | Full simulation: separate KV cache, scheduler, queue per instance (`cluster.go:274-341`) |
| LLMServingSim | `instances[]` array in cluster config | Full simulation: separate `Scheduler`, `MemoryModel`, request queue per instance (`__main__.py:303-325`) |
| AIConfigurator | derived from `--total-gpus` | Linear scaling assumption: `cluster_rate = single_rate * replicas` (`report_and_save.py:109`). No queuing or routing simulation. |
| Vidur | `--cluster_config_num_replicas` | Full simulation: separate `ReplicaScheduler`, memory planner, queue per replica (`cluster.py:27-29`, `base_global_scheduler.py:26-37`) |

**vLLM-internal DP** (llm-optimizer only): Multiple worker groups within a single vLLM process/deployment. vLLM spawns DP workers internally and handles routing between them. There is one pod, one deployment, one KV cache pool (partitioned across DP groups).

| Tool | Parameter | Behavior |
|---|---|---|
| llm-optimizer | `data_parallel_size` in `--server-args` | Passes `--data-parallel-size` to a single `vllm serve` command (`server_utils.py:43-53`). vLLM handles internal routing. No external scheduler. |

These are not interchangeable. A config with `num_instances=4` (4 Kubernetes pods, external routing) behaves differently from `data_parallel_size=4` (1 pod with 4 internal DP groups, vLLM-internal routing). For llm-d validation, the deployment harness uses Kubernetes replicas with the llm-d inference scheduler for routing.

### Workload Configuration

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Input length | via workload spec/preset | dataset JSONL `input_toks` | `--isl` (default: 4000) | trace file or `--trace_request_length_generator_config_max_tokens` | `--input-len` (estimate) / `random_input_len` (benchmark) |
| Output length | via workload spec/preset | dataset JSONL `output_toks` | `--osl` (default: 1000) | trace file | `--output-len` (estimate) / `random_output_len` (benchmark) |
| Request rate (QPS) | `--rate` | arrival times in dataset JSONL | N/A | `--poisson_request_interval_generator_config_qps` | N/A (uses concurrency) |
| Num requests | `--num-requests` | `--num-reqs` | N/A | `--synthetic_request_generator_config_num_requests` | `num_prompts` in `--client-args` |
| Concurrency | N/A (open-loop only) | N/A | N/A | N/A | `max_concurrency` in `--client-args` |
| Workload source | `--workload` preset (chatbot, summarization, contentgen, multidoc) or `--workload-spec` YAML | `--dataset` JSONL (flat requests or agentic sessions) | N/A (synthetic steady-state only) | trace CSV or synthetic generation | `dataset_name` (sharegpt, random) |
| Prefix caching | via workload spec prefix fields | `--enable-prefix-caching`, `--enable-prefix-sharing`, `--prefix-storage` (None, CPU, CXL) | `--prefix` (prefix length in tokens) | N/A | N/A |
| Arrival pattern | Poisson (via `--rate`) | Explicit timestamps in JSONL | N/A | `--interval_generator_config_type` (poisson, gamma, trace, static) | N/A (closed-loop concurrency) |

### Cluster-Level Routing (Across Instances)

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Routing policy | `--routing-policy`: round-robin, least-loaded, weighted | `--request-routing-policy`: LOAD, RR, RAND, CUSTOM | N/A | `--global_scheduler_config_type`: round_robin (default), lor, random | N/A |
| Routing scorers (weighted) | `--routing-scorers` (e.g., `precise-prefix-cache:2,queue-depth:1,kv-utilization:1`) | N/A (LOAD uses vLLM-style weighted least-loaded) | N/A | N/A | N/A |
| PD pool routing | `--prefill-routing-scorers`, `--decode-routing-scorers` (separate scorer configs per pool) | N/A | N/A | N/A | N/A |

### Instance-Internal Scheduling (Within One Replica)

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Scheduler type | `--scheduler`: fcfs, priority-fcfs, sjf, reverse-priority | N/A (mirrors vLLM continuous batching) | N/A | `--replica_scheduler_config_type`: sarathi, vllm, orca, lightllm, faster_transformer | N/A (passthrough to framework) |
| Max batch size / seqs | N/A (continuous batching) | `--max-num-seqs` (default: 128) | `--batch-size` (estimate mode) | `--sarathi_scheduler_config_batch_size_cap` | `max_num_seqs` in `--server-args` |
| Max batched tokens | N/A | `--max-num-batched-tokens` (default: 2048) | N/A | N/A | `max_num_batched_tokens` in `--server-args` |
| Chunked prefill | N/A | `--enable-chunked-prefill` (boolean) + `--long-prefill-token-threshold` (per-request cap; 0=uncapped, not disabled; guard: `if 0 < threshold < remaining`) | N/A | `--sarathi_scheduler_config_chunk_size` | `chunked_prefill_size` in `--server-args` |
| Prefill priority | N/A | `--prioritize-prefill` | N/A | N/A | N/A |
| Preemption policy | `--preemption-policy`: fcfs (tail-of-batch), priority (least-urgent SLO tier) | N/A | N/A | N/A | N/A |
| Admission control | `--admission-policy`: always-admit, token-bucket, tier-shed, gaie-legacy, reject-all | N/A | N/A | N/A | N/A |

### MoE Expert Routing (Token-Level)

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Expert routing policy | N/A | `--expert-routing-policy`: BALANCED, RR, RAND, CUSTOM | N/A | N/A | N/A |

### PD Disaggregation (Prefill/Decode Separation)

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Disagg mode | `--pd-decider`: never, always, prefix-threshold | N/A | `--estimate-mode disagg` / auto in `cli default` | N/A | N/A |
| Prefill workers / TP | `--prefill-tp`, `--prefill-hardware`, `--prefill-latency-model` | N/A | `--prefill-num-workers`, `--prefill-batch-size`, `--prefill-tp-size` | N/A | N/A |
| Decode workers / TP | `--decode-tp`, `--decode-hardware`, `--decode-latency-model` | N/A | `--decode-num-workers`, `--decode-batch-size`, `--decode-tp-size` | N/A | N/A |
| KV transfer modeling | `--pd-transfer-bandwidth` (GB/s), `--pd-transfer-base-latency` (ms) | N/A | N/A (analytical) | N/A | N/A |
| Prefix threshold | `--pd-prefix-threshold` (tokens; disaggregate when non-cached tokens exceed) | N/A | N/A | N/A | N/A |

### SLO and Constraint Arguments

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| TTFT target | N/A | N/A | `--ttft` (ms, single-point prediction) | N/A | `ttft<Xms` or `ttft:stat<Xms` in `--constraints` |
| TPOT / ITL target | N/A | N/A | `--tpot` (ms, single-point prediction) | N/A | `itl:stat<Xms` in `--constraints` |
| E2E latency target | N/A | N/A | `--request-latency` (ms, TTFT + TPOT x tokens) | N/A | `e2e_latency:stat<Xs` in `--constraints` |
| Scheduling delay SLO | N/A | N/A | N/A | `--scheduling-delay-slo-value` (seconds) | N/A |
| SLO quantile | N/A | N/A | N/A (no percentile support) | `--scheduling-delay-slo-quantile` (e.g., 0.99) | stat = mean, median, p95, p99 |
| Fitness scoring | `--fitness-weights` (weighted metric score for manual comparison) | N/A | N/A | N/A | N/A |

### Output and Metrics

| Parameter | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| Output path | `--metrics-path` (JSON) | `--output` (CSV) | `--save-dir` (deployment configs) | `--metrics_config_output_dir` | `--output-dir` or `--output-json` |
| Trace export | `--trace-output` (TraceV2 YAML+CSV) | N/A | N/A | Chrome trace (auto) | N/A |
| W&B integration | N/A | N/A | N/A | `--metrics_config_wandb_project` | N/A |
| Visualization | N/A | N/A | `aiconfigurator webapp` (Gradio) | `streamlit run dashboard/main.py` | `llm-optimizer visualize` (HTML Pareto) |
| Deterministic seed | `--seed` | N/A | N/A (analytical, deterministic) | N/A | N/A |

### Output Metrics Produced

| Metric | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| TTFT (mean, pN) | Yes | Yes | Yes (single predicted value) | Yes | Yes (measured distribution) |
| TPOT / ITL (mean, pN) | Yes | Yes | Yes (single predicted value) | Yes | Yes (measured distribution) |
| E2E latency | Yes | Yes | Yes (predicted) | Yes | Yes (measured) |
| Throughput (tok/s) | Yes | Yes | Yes (predicted) | Yes | Yes (measured) |
| Preemption count | Yes | N/A | N/A | Yes | N/A |
| MFU | N/A | N/A | N/A | Yes | N/A |
| Power estimate | N/A | N/A | Yes | N/A | N/A |
| Cost / capacity-per-dollar | N/A | N/A | N/A | Yes | N/A |
| Per-request detail | Yes (via `--results-path` JSON) | Yes (CSV per request) | N/A | Yes (per-request CSV) | Yes (per-request in JSON) |

## Unique Capabilities by Tool

Arguments listed here are NOT covered in the shared tables above. These are knobs only one tool provides.

### inference-sim only

**Cluster-level routing:**
- `--routing-policy`: round-robin, least-loaded, weighted
- `--routing-scorers`: comma-separated `name:weight` pairs (e.g., `precise-prefix-cache:2,queue-depth:1,kv-utilization:1`)
- `--routing-latency`: simulated routing decision latency in microseconds

**Admission control:**
- `--admission-policy`: always-admit, token-bucket, tier-shed, gaie-legacy, reject-all
- `--admission-latency`: simulated admission decision latency in microseconds

**PD disaggregation:**
- `--pd-decider`: never, always, prefix-threshold
- `--pd-transfer-bandwidth`: KV transfer bandwidth in GB/s (NIXL RDMA, default: 25)
- `--pd-transfer-base-latency`: KV transfer base latency in ms (default: 0.05)
- `--pd-prefix-threshold`: non-cached token threshold for prefix-threshold decider
- `--prefill-routing-scorers` / `--decode-routing-scorers`: per-pool scorer weights
- `--prefill-tp` / `--decode-tp`: per-pool tensor parallelism
- `--prefill-hardware` / `--decode-hardware`: per-pool GPU type
- `--prefill-latency-model` / `--decode-latency-model`: per-pool latency backend
- `--prefill-max-model-len` / `--decode-max-model-len`: per-pool context length

**Preemption:**
- `--preemption-policy`: fcfs (tail-of-batch eviction), priority (least-urgent SLO tier)

**Latency model selection:**
- `--latency-model`: roofline (analytical, default) or trained-physics (learned coefficients)

**Workload management:**
- `blis convert preset` / `blis convert servegen` / `blis convert inference-perf`: format converters
- `blis compose --from spec1.yaml --from spec2.yaml`: merge workload specs
- `--trace-output`: export simulation results as TraceV2 (YAML header + CSV data)

**Calibration pipeline:**
- `blis observe`: dispatch to real server, record TraceV2
- `blis replay`: replay trace through simulator
- `blis calibrate`: compare real vs simulated (produces calibration report JSON)
- `--record-itl` / `--itl-output`: inter-token latency recording for calibration
- `--rtt-ms` / `--network-rtt-us`: network RTT compensation

**Scoring:**
- `--fitness-weights`: weighted metric score for comparing configs manually
- `--seed`: deterministic simulation for reproducible comparisons

### LLMServingSim only

**Memory hierarchy (in cluster config JSON):**
- `instances[].npu_mem.mem_bw`: NPU memory bandwidth (GB/s)
- `instances[].cpu_mem.mem_bw`: CPU memory bandwidth (GB/s)
- `instances[].link_bw`: inter-node bandwidth (GB/s)
- `instances[].link_latency`: inter-node link latency (ns)

**Memory offloading:**
- `--enable-local-offloading`: weight offloading to local NPU memory
- `--enable-attn-offloading`: attention offloading to PIM devices
- `--enable-sub-batch-interleaving`: overlap XPU and PIM computation (requires `--enable-attn-offloading`)
- `--enable-prefix-sharing`: second-tier prefix cache pooling across instances
- `--prefix-storage`: storage medium for shared prefix cache (None, CPU, CXL)

**MoE expert routing:**
- `--expert-routing-policy`: BALANCED, RR, RAND, CUSTOM (trained load-balanced gate)
- `--enable-block-copy` / `--no-enable-block-copy`: replay block traces across layers for MoE

**Network simulation:**
- `--network-backend`: analytical (default) or ns3 (WIP)

**Profiler tool (`python -m profiler`):**
- `profile <model> --hardware NAME --tp DEGREES`: full sweep across TP degrees and categories
- `slice <model> --hardware NAME --tp-refresh N --group GROUP`: refresh one (TP, category) pair
- `--variant`: output folder label (e.g., "bf16-kvfp8")
- `--measurement-iterations`: timed forwards per shot
- `--attention-max-kv` / `--attention-chunk-factor` / `--attention-kv-factor`: sweep axis controls
- `--skip-skew` / `--skew-n-factor` / `--skew-pc-factor`: heterogeneous-decode skew profiling

**Benchmark validation (`python -m bench`):**
- `bench run --model NAME --dataset PATH --run-id ID`: execute vLLM benchmark
- `bench validate --bench-run ID --sim-output PATH --plot-dir PATH`: compare bench vs sim

### AIConfigurator only

**Performance estimation mode:**
- `--database-mode`: SILICON (real hardware measurements, default), HYBRID (silicon when available, else SOL+empirical), EMPIRICAL (SOL+empirical for all), SOL (speed-of-light only)
- `--systems-paths`: override system YAML/data search paths (comma-separated; `default` = built-in)

**Request latency mode:**
- `--request-latency`: end-to-end target in ms (TTFT + TPOT x (OSL-1)); auto-enumerates valid TTFT/TPOT splits that satisfy this budget

**Backend selection:**
- `--backend`: trtllm (default, full support), vllm (dense models only), sglang (dense + MoE)
- `--backend-version`: pin to specific backend version

**Deployment artifact generation:**
- `--save-dir`: generates ready-to-use Dynamo deployment configs (YAML, scripts)

**Disaggregated estimation arguments (unique to `cli estimate --estimate-mode disagg`):**
- `--prefill-batch-size` / `--prefill-num-workers`
- `--decode-batch-size` / `--decode-num-workers`
- `--prefill-tp-size` / `--prefill-pp-size`
- `--decode-tp-size` / `--decode-pp-size`

**Experiment runner:**
- `aiconfigurator cli exp --yaml-path FILE`: run custom multi-experiment batches from YAML
- `aiconfigurator cli generate`: quick naive config without sweep (fast deployment)
- `aiconfigurator cli support`: pre-flight check if model/hardware/backend is supported

**Web interface:**
- `aiconfigurator webapp`: Gradio UI at 127.0.0.1:7860

### Vidur only

**Scheduler implementations (models internal scheduling logic of each framework):**
- `--replica_scheduler_config_type`: sarathi, vllm, orca, lightllm, faster_transformer
- `--sarathi_scheduler_config_chunk_size`: chunked prefill size for Sarathi scheduler
- `--sarathi_scheduler_config_batch_size_cap`: max batch size for Sarathi
- `--vllm_scheduler_config_max_tokens_in_batch`: token limit for vLLM scheduler

**Execution time prediction:**
- `--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size`
- `--random_forrest_execution_time_predictor_config_prediction_max_batch_size`
- `--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request`

**Config explorer (`python -m vidur.config_optimizer.config_explorer.main`):**
- `--config-path`: YAML defining search space (models, devices, schedulers, batch sizes, TP/PP dims)
- `--scheduling-delay-slo-value`: target scheduling delay threshold (seconds, default: 5.0)
- `--scheduling-delay-slo-quantile`: percentile for SLO (default: 0.99)
- `--max-iterations`: max binary search iterations per configuration (default: 20)
- `--min-search-granularity`: minimum search granularity as % of QPS
- `--time-limit`: time limit per simulation run (minutes)
- `--num-threads`: parallel Ray workers
- `--cache-dir`: cache simulation results for faster iteration

**Config space YAML fields:**
- `clusters[].device`, `clusters[].num_gpus`, `clusters[].gpus_per_node`
- `schedulers[].scheduler`, `schedulers[].chunk_size`
- `traces[].trace_file`, `traces[].max_seq_len`, `traces[].num_requests`, `traces[].start_qps`
- `batch_sizes`, `tp_dimensions`, `pp_dimensions` (lists for Cartesian product)
- `models[].name`, `models[].identifier`

**Analysis tools:**
- `python -m vidur.config_optimizer.analyzer.generate_pareto_curves --results-dir DIR`
- `streamlit run vidur/config_optimizer/analyzer/dashboard/main.py` (interactive Streamlit dashboard)

**W&B integration:**
- `--metrics_config_wandb_project` / `--metrics_config_wandb_group`

### llm-optimizer only

**Grid search syntax (in `--server-args` and `--client-args`):**
- List values: `key=[val1,val2,val3]`
- Range: `key=range(start,end,step)`
- Paired parameters: `key1*key2=[(a,b),(c,d)]`
- Multiple params separated by semicolons create Cartesian product

**Constraint syntax (in `--constraints`):**
- `ttft<Xms` / `ttft:stat<Xms` (stat = mean, median, p95, p99)
- `itl<Xms` / `itl:stat<Xms`
- `e2e_latency<Xs` / `e2e_latency:stat<Xs`
- Multiple constraints separated by semicolons (all must be satisfied)

**Execution control:**
- `--dry-run`: preview all config combinations without running
- `--continue` / `-c`: resume from existing results, skip completed configs
- `--rest`: rest time in seconds between runs (default: 10)
- `--mute-server`: suppress server stdout
- `--ready-endpoint`: health check endpoint (default: /health)

**Performance estimation (`llm-optimizer estimate`):**
- `--input-len` / `--output-len`: sequence lengths for estimation
- `--target`: optimization target (throughput or latency)
- `--framework`: sglang, vllm, or both
- `--dataset`: random or sharegpt
- `--interactive`: guided interactive mode

**Framework-specific server params (via `--server-args`):**
- SGLang: `tp_size`, `dp_size`, `chunked_prefill_size`, `schedule_conservativeness`, `schedule_policy`
- vLLM: `tensor_parallel_size`, `data_parallel_size`, `max_num_batched_tokens`, `max_num_seqs`

**Client params (via `--client-args`):**
- `max_concurrency`: concurrent requests
- `num_prompts`: total requests
- `dataset_name`: sharegpt or random
- `random_input_len` / `random_output_len`: sequence lengths for random dataset
- `request_rate`: fixed request rate

**Visualization (`llm-optimizer visualize`):**
- `--data-file`: JSON results (comma-separated for multiple)
- `--config`: visualization config file
- `-o` / `--output`: output HTML path
- `--serve` / `--port`: start HTTP server for interactive viewing


## Environment Variables

| Variable | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| `HF_TOKEN` | Yes (fetch model configs from HuggingFace) | Yes (profiler needs gated model access) | Yes (priority 1 for HF auth) | No | Yes (gated models) |
| `HUGGING_FACE_HUB_TOKEN` | No | No | Yes (priority 2 for HF auth) | No | No |
| `OPENAI_API_KEY` | No | No | No | No | Yes (benchmark client auth header) |
| `SGLANG_USE_MODELSCOPE` | No | No | No | No | Yes (use ModelScope instead of HF) |
| `NO_COLOR` | No | No | Yes (disable ANSI colors) | No | No |
| `WANDB_MODE` | No | No | No | Yes (`disabled` to skip W&B logging) | No |
| `PYTHONHASHSEED` | No | No | No | Set internally (reproducibility) | No |
| `CUDA_VISIBLE_DEVICES` | No | No | No | Yes (profiling only, set by Ray) | No |

## GPU Type Specification (Alignment Challenge)

Each tool uses a different identifier scheme, property set, and config format for GPUs. This is the primary alignment challenge for cross-tool experiments.

### Identifier Mapping

| GPU | inference-sim (`--hardware`) | LLMServingSim (cluster config `hardware`) | AIConfigurator (`--system`) | Vidur (`--replica_config_device`) | llm-optimizer (`--gpu`) |
|---|---|---|---|---|---|
| H100 SXM | `H100` | `"H100"` (free-form) | `h100_sxm` | `h100` | `H100` |
| H200 SXM | N/A | N/A | `h200_sxm` | N/A | `H200` |
| A100 SXM 80GB | `A100-SXM` | `"A100"` (free-form) | `a100_sxm` | `a100` | `A100` |
| A100 40GB | N/A | user-defined | N/A | N/A | `A100-40GB` |
| L40S | `L40S` | user-defined | `l40s` | N/A | `L40` |
| A40 | N/A | N/A | N/A | `a40` | N/A |
| B200 | N/A | N/A | `b200_sxm` | N/A | `B200` |
| B100 | N/A | N/A | N/A | N/A | `B100` |
| GB200 | N/A | N/A | `gb200` | N/A | N/A |
| GB300 | N/A | N/A | `gb300` | N/A | N/A |

### GPU Properties Modeled Per Tool

| Property | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| FP16/BF16 TFLOPS | Yes (`TFlopsPeak`) | No (empirical profiling instead) | Yes (`float16_tc_flops`) | Yes (`fp16_tflops`) | Yes (`FP16_TFLOPS`) |
| FP8 TFLOPS | Yes (`TFlopsFP8`) | No | Yes (`fp8_tc_flops`) | No | Yes (`FP8_TFLOPS`) |
| FP4 TFLOPS | No | No | Yes (`fp4_tc_flops`, Blackwell only) | No | No |
| INT8 TFLOPS | No | No | Yes (`int8_tc_flops`) | No | No |
| Memory capacity | Yes (`MemoryGiB`) | Yes (`mem_size` GB) | Yes (`mem_capacity` bytes) | Yes (`total_memory_gb`) | Yes (`VRAM_GB`) |
| Memory bandwidth | Yes (`BwPeakTBs` TB/s) | Yes (`mem_bw` GB/s) | Yes (`mem_bw` bytes/s) | No (from profiling data) | Yes (`Memory_Bandwidth_GBs`) |
| Memory latency | No | Yes (`mem_latency` ns) | Yes (`mem_empirical_constant_latency`) | No | No |
| MFU (prefill/decode) | Yes (`mfuPrefill`, `mfuDecode`) | No (implicit in profiled data) | Yes (`mem_bw_empirical_scaling_factor`) | No | No |
| TDP / Power | No | Yes (power model: idle/standby/active W) | Yes (`power` watts) | No | No |
| Architecture name | No | No | No | No | Yes (`Architecture`) |
| Memory type | No | No | No | No | Yes (`Memory_Type`) |
| SM version | No | No | Yes (`sm_version`) | No | No |
| Intra-node BW (NVLink) | No | Yes (`link_bw` GB/s in cluster config) | Yes (`intra_node_bw` bytes/s) | No (from profiled collectives) | No |
| Inter-node BW | No | Yes (`link_bw` GB/s) | Yes (`inter_node_bw` bytes/s) | No | No |
| PCIe BW | No | No | Yes (`pcie_bw` bytes/s) | No | No |
| P2P latency | No | Yes (`link_latency` ns) | Yes (`p2p_latency` seconds) | No | No |
| GPUs per node | No | cluster config | Yes (`num_gpus_per_node`) | Yes (in network device config) | No |
| NCCL memory overhead | No | No | Yes (`nccl_mem` dict) | No | No |

### Config Format and Location

| Tool | Format | Location | How to add new GPU |
|---|---|---|---|
| inference-sim | JSON | `hardware_config.json` at repo root (or `--hardware-config`) | Add entry to JSON with TFlopsPeak, TFlopsFP8, BwPeakTBs, mfuPrefill, mfuDecode, MemoryGiB |
| LLMServingSim | JSON (cluster config) | `configs/cluster/*.json` per-instance | User defines mem_size/mem_bw in cluster config; run profiler on real hardware for latency data |
| AIConfigurator | YAML | `src/aiconfigurator/systems/{name}.yaml` | Create new YAML with gpu, node, misc sections; add performance data CSVs |
| Vidur | Python dataclass | `vidur/config/device_sku_config.py` | Add new dataclass with fp16_tflops and total_memory_gb; profile network/compute CSVs |
| llm-optimizer | Python dict | `src/llm_optimizer/predefined/gpus.py` | Add entry to `GPU_SPECS` dict with TFLOPS, bandwidth, VRAM, architecture, memory type |

### Performance Modeling Approach

| Tool | How it predicts latency |
|---|---|
| inference-sim | Analytical roofline (compute vs memory-bound per phase) with optional trained-physics coefficients. Uses TFLOPS + bandwidth + MFU. |
| LLMServingSim | Purely empirical: replays profiled per-layer kernel latencies from real GPU measurements. No analytical model. |
| AIConfigurator | Silicon-calibrated: uses measured GEMM/attention/NCCL performance data per system. Falls back to analytical SOL model. |
| Vidur | Random forest ML model trained on profiled compute/network CSV data. Predicts per-operation latency. |
| llm-optimizer | Live measurement (benchmark mode) or roofline analysis (estimate mode). |

## GPU Cost Modeling

| | inference-sim | LLMServingSim | AIConfigurator | Vidur | llm-optimizer |
|---|---|---|---|---|---|
| **Has cost modeling** | Yes | No | Partial (webapp only) | Yes | No |
| **Where defined** | `cost_per_hour` in node pool config YAML | N/A | User inputs `gpu_cost_per_hr` in web UI | Hardcoded in `config_optimizer/analyzer/constants.py` | N/A |
| **Unit** | $/hour per node | N/A | $/hour per GPU (user-provided) | $/hour per GPU | N/A |
| **Used for** | Autoscaler cost-aware decisions, routing state | N/A | "GPU hours per 1000 requests" metric | Pareto analysis: capacity-per-dollar curves | N/A |
| **Values (H100)** | User-configured | N/A | User-provided at runtime | $4.25/hr (CoreWeave, Feb 2024) | N/A |
| **Values (A100)** | User-configured | N/A | User-provided at runtime | $2.21/hr (CoreWeave, Feb 2024) | N/A |

### Alignment Issues for Shared Experiments

These are the key challenges when running the same config across all tools:

1. **GPU identifier mismatch**: Every tool uses a different string/enum. Need a translation layer (e.g., `H100` vs `h100` vs `h100_sxm`).

2. **Property units differ**: Memory bandwidth is TB/s (inference-sim), GB/s (LLMServingSim, llm-optimizer), or bytes/s (AIConfigurator). Memory capacity is GiB (inference-sim), GB (others).

3. **What "GPU type" means varies**: inference-sim and llm-optimizer treat it as a spec sheet lookup. LLMServingSim treats it as a profiling directory name. AIConfigurator bundles node topology with GPU specs. Vidur combines a compute device enum with a separate network topology enum.

4. **FLOPS values disagree**: H100 FP16 TFLOPS varies across tools (989.5 in inference-sim, 989 in AIConfigurator, 1000 in Vidur). These differences come from whether specs reflect peak sustained or burst, and whether they include sparsity.

5. **No shared cost source**: Vidur hardcodes CoreWeave 2024 prices. inference-sim requires user config. AIConfigurator defers to the user at runtime. LLMServingSim and llm-optimizer have no cost model.

6. **Network topology handled differently**: AIConfigurator explicitly models NVLink/PCIe/InfiniBand bandwidths. LLMServingSim uses per-link bandwidth in cluster config. Vidur uses profiled collective timings. inference-sim and llm-optimizer abstract away interconnect entirely.

## Config Search Mechanisms (Detail)

### AIConfigurator
- **Method:** Exhaustive sweep over valid parallelism/worker configurations using analytical model
- **Command:** `aiconfigurator cli default`
- **Search space:** TP, PP, replicas, prefill/decode worker counts and batch sizes (disaggregated serving)
- **Constraint:** Fixed TTFT and TPOT thresholds in ms (single-point predictions, no distribution)
- **Speed:** Seconds (no simulation or GPU needed per config point)
- **Output:** Ranked configurations with predicted throughput, deployment-ready configs

### Vidur
- **Method:** Binary search over QPS per configuration; Cartesian product over config dimensions
- **Command:** `python -m vidur.config_optimizer.config_explorer.main`
- **Search space:** Models, devices, schedulers, batch sizes, TP/PP dimensions (YAML-defined)
- **Constraint:** Scheduling delay at configurable quantile (e.g., P99 < 5s)
- **Speed:** Minutes (runs full simulation per QPS probe, but parallelizable via Ray)
- **Output:** Max sustainable QPS per config, Pareto curves, interactive dashboard

### llm-optimizer
- **Method:** Grid search with live benchmarking of every configuration
- **Command:** `llm-optimizer` with `--server-args` / `--client-args` grid syntax
- **Search space:** TP/DP combinations, batch sizes, prefill chunk sizes, concurrency levels
- **Constraint:** TTFT, ITL, E2E at mean/median/p95/p99
- **Speed:** Minutes to hours (starts real server per config, runs benchmark workload)
- **Output:** Best configs with/without constraints, full benchmark results JSON, Pareto dashboard

### inference-sim and LLMServingSim (no native search)
- **Method:** Single-point evaluation only
- **Advantage:** Fast per-evaluation (no GPU), so external sweeps are cheap
- **For search:** Wrap in external script; parse JSON/CSV output; implement your own optimization loop
