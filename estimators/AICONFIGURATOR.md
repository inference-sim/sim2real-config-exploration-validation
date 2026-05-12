# AIConfigurator

## Overview

AIConfigurator optimizes LLM inference deployment configurations for disaggregated serving. Given a model, GPU count, and hardware type, it searches the configuration space to find strong prefill/decode worker setups that meet SLA targets (TTFT, TPOT). It models performance using collected silicon data and generates deployment-ready configuration files for Dynamo.

## Installation

### From PyPI

```bash
pip install aiconfigurator==0.8.0
```

### From Source

```bash
# 1. Install Git LFS
apt-get install git-lfs  # (Linux)
# brew install git-lfs   # (macOS)

# 2. Clone and pull LFS data
git clone https://github.com/ai-dynamo/aiconfigurator.git
git lfs pull

# 3. Create virtual environment
python -m venv myenv && source myenv/bin/activate  # requires Python 3.10+

# 4. Install
pip install .

# 5. Install with optional webapp support
pip install .[webapp]
```

## Commands

### `aiconfigurator cli default`

Search configuration space for agg vs disagg comparison. Returns best deployment configuration.

**Required arguments:**
- `--model-path` (alias `--model`): HuggingFace model ID or local path
- `--total-gpus`: Total GPUs for deployment. AIConfigurator derives `replicas = total_gpus / (tp * pp)` and models multi-instance throughput as pure linear scaling: `cluster_request_rate = single_replica_rate * num_replicas` (`report_and_save.py:109`). Each replica is conceptually an independent Kubernetes-style pod, but there is no simulation of per-replica queuing, KV cache, or routing effects. Not equivalent to vLLM's internal `--data-parallel-size`.
- `--system`: System name (`h200_sxm`, `h100_sxm`, `b200_sxm`, `gb200`, `a100_sxm`)

**Key optional arguments:**
- `--backend`: Backend (`trtllm` [default], `vllm`, `sglang`)
- `--ttft`: Max time to first token in ms (default: 2000)
- `--tpot`: Max time per output token in ms (default: 30)
- `--isl`: Input sequence length (default: 4000)
- `--osl`: Output sequence length (default: 1000)
- `--prefix`: Prefix length for prefix caching (default: 0)
- `--save-dir`: Directory to generate Dynamo deployment configs
- `--database-mode`: Performance estimation mode (`SILICON` [default], `HYBRID`, `EMPIRICAL`, `SOL`)
- `--systems-paths`: Override system YAML/data search paths (comma-separated; `default` = built-in path)

**Example:**
```bash
aiconfigurator cli default --model Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --ttft 300 --tpot 10
```

**Python API:**
```python
from aiconfigurator.cli import cli_default

result = cli_default(
    model_path="Qwen/Qwen3-32B-FP8",
    total_gpus=32,
    system="h200_sxm",
    ttft=300,
    tpot=10
)
print(result.best_configs["disagg"].head())
```

---

### `aiconfigurator cli exp`

Run custom experiments defined in a YAML file or dictionary.

**Required arguments:**
- `--yaml-path`: Path to YAML experiment config file

**Example:**
```bash
aiconfigurator cli exp --yaml-path my_experiments.yaml
```

**Python API:**
```python
from aiconfigurator.cli import cli_exp

# From file
result = cli_exp(yaml_path="experiments.yaml")

# From dict
result = cli_exp(config={
    "my_exp": {
        "serving_mode": "disagg",
        "model_path": "Qwen/Qwen3-32B-FP8",
        "total_gpus": 32,
        "system_name": "h200_sxm",
        "isl": 4000,
        "osl": 1000,
    }
})
```

---

### `aiconfigurator cli generate`

Generate a naive working configuration without parameter sweep. Fast setup for quick deployment.

**Required arguments:**
- `--model-path`: HuggingFace model ID or local path
- `--total-gpus`: Total GPUs for deployment
- `--system`: System name

**Key optional arguments:**
- `--backend`: Backend (default: `trtllm`)
- `--save-dir`: Output directory for generated configs

**Example:**
```bash
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
```

**Python API:**
```python
from aiconfigurator.cli import cli_generate

result = cli_generate(
    model_path="Qwen/Qwen3-32B-FP8",
    total_gpus=8,
    system="h200_sxm"
)
print(result["parallelism"])  # {'tp': 1, 'pp': 1, 'replicas': 8, 'gpus_used': 8}
```

---

### `aiconfigurator cli estimate`

Single-point performance estimation. Predict TTFT, TPOT, and power for a specific configuration (no sweep).

**Required arguments:**
- `--model-path`: HuggingFace model ID or local path
- `--system`: System name

**Key optional arguments:**
- `--estimate-mode`: `agg` (default) or `disagg`
- `--backend`: Backend (default: `trtllm`)
- `--isl`: Input sequence length (default: 1024)
- `--osl`: Output sequence length (default: 1024)
- `--batch-size`: Batch size (default: 128)
- `--tp-size`: Tensor parallelism size (default: 1)
- `--pp-size`: Pipeline parallelism size (default: 1)
- `--print-per-ops-latency`: Print per-operation latency breakdown

**Disagg-specific arguments:**
- `--prefill-batch-size`, `--prefill-num-workers`: Prefill worker config (required for disagg)
- `--decode-batch-size`, `--decode-num-workers`: Decode worker config (required for disagg)
- `--prefill-tp-size`, `--prefill-pp-size`: Prefill parallelism overrides
- `--decode-tp-size`, `--decode-pp-size`: Decode parallelism overrides

**Example:**
```bash
aiconfigurator cli estimate --model-path Qwen/Qwen3-32B --system h200_sxm --tp-size 2 --batch-size 64 --isl 2048 --osl 512
```

**Python API:**
```python
from aiconfigurator.cli.api import cli_estimate

# Aggregated
result = cli_estimate(
    "Qwen/Qwen3-32B", "h100_sxm",
    batch_size=64, isl=2048, osl=512, tp_size=2
)
print(f"TTFT: {result.ttft:.2f} ms, TPOT: {result.tpot:.2f} ms")

# Disaggregated
result = cli_estimate(
    "Qwen/Qwen3-32B", "h100_sxm", mode="disagg",
    prefill_batch_size=4, prefill_num_workers=2,
    decode_batch_size=64, decode_num_workers=2
)
```

---

### `aiconfigurator cli support`

Check if a model/hardware combination is supported for agg and disagg modes (optional pre-flight check).

**Required arguments:**
- `--model-path`: HuggingFace model ID or local path
- `--system`: System name

**Key optional arguments:**
- `--backend`: Filter by backend (default: `trtllm`)
- `--backend-version`: Filter by backend version (default: latest)

**Example:**
```bash
aiconfigurator cli support --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm
```

**Python API:**
```python
from aiconfigurator.cli import cli_support

agg_supported, disagg_supported = cli_support(
    model_path="Qwen/Qwen3-32B-FP8",
    system="h200_sxm",
    backend="trtllm"
)
print(f"Agg: {agg_supported}, Disagg: {disagg_supported}")
```

---

### `aiconfigurator webapp`

Launch web interface at `127.0.0.1:7860` (requires `pip install aiconfigurator[webapp]`).

**Example:**
```bash
aiconfigurator webapp
```

---

## Common Flags Across Commands

- `--backend`: Inference backend - `trtllm` (default), `vllm`, `sglang`
- `--save-dir`: Directory to save generated artifacts (configs, scripts, YAML)
- `--systems-paths`: Override system data search paths (comma-separated; `default` = built-in)
- `--database-mode`: Performance estimation mode:
  - `SILICON` (default): Uses collected silicon data (reproducible)
  - `HYBRID`: Silicon data when available, else SOL+empirical
  - `EMPIRICAL`: SOL+empirical for all
  - `SOL`: Speed-of-light only (research purpose)

---

## Example Workflow

```bash
# 1. Check support (optional)
aiconfigurator cli support --model Qwen/Qwen3-32B-FP8 --system h200_sxm

# 2. Find optimal configuration
aiconfigurator cli default \
  --model Qwen/Qwen3-32B-FP8 \
  --total-gpus 32 \
  --system h200_sxm \
  --ttft 300 \
  --tpot 10 \
  --isl 4000 \
  --osl 500 \
  --save-dir ./deployment

# 3. Quick naive config (alternative to step 2)
aiconfigurator cli generate \
  --model Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm \
  --save-dir ./quick_deploy
```

---

## Supported Models

- GPT, LLAMA (2, 3), QWEN, DEEPSEEK_V3
- MOE models
- HuggingFace model IDs for dense models (non-MoE)

## Supported Systems

- `h100_sxm`, `h200_sxm`, `b200_sxm`, `gb200`, `a100_sxm`, `l40s`, `gb300`

## Supported Backends

- **TensorRT-LLM** (`trtllm`) - default, full support
- **vLLM** (`vllm`) - dense models only, currently being evaluated
- **SGLang** (`sglang`) - dense and MoE models, currently being evaluated

---

## References

- Paper: [AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving](https://arxiv.org/abs/2601.06288)
- GitHub: [https://github.com/ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator)
- Full CLI guide: `docs/cli_user_guide.md`
- Advanced tuning: `docs/advanced_tuning.md`
- Deployment guide: `docs/dynamo_deployment_guide.md`
