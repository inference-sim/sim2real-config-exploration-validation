# Results: Llama-2-70b Config Exploration (300ms SLO, 5 rps)

Experiment: sim2real config exploration validation for the BLIS paper.
Model: meta-llama/Llama-2-70b-hf on H100 SXM 80GB GPUs.
Workload: shared_prefix_codegen (ISL mean=566, OSL mean=247, constant 5 rps, 5000 requests).
SLO: 300ms mean TTFT.
Serving: vLLM v0.22.1 via llm-d.

## Directory Structure

```
results-llama2-70b-300ms-5rps/
  raw/                          # Estimator search outputs (predicted configs)
    blis.jsonl                  #   420 configs from BLIS NSGA-II search
    aiconfigurator.jsonl        #   13 configs from AIConfigurator sweep
    llm-optimizer.jsonl         #    7 configs from LLM-Optimizer grid
  validation/                   # Real cluster measurements (per-request traces)
    <run>/shared_prefix_codegen/
      trace_header.yaml         #   Metadata (model, time unit, workload spec)
      trace_data.csv            #   Per-request: TTFT, latency, tokens, status
      server_logs/              #   vLLM server logs (if collected)
      epp_logs/                 #   EPP router logs (if collected)
    plans/                      #   Helm plans used for each deployment
```

## Paper-Relevant Runs

| Run Name    | Paper Config           | TP x Rep | GPUs | max_seqs | Rate | Requests | Result     |
|-------------|------------------------|----------|------|----------|------|----------|------------|
| blis3       | BLIS TP4, 64 seqs      | 4x1      | 4    | 64       | 5    | 5000     | 96ms (pass)|
| blis4       | BLIS TP4, 256 seqs     | 4x1      | 4    | 256      | 5    | 5000     | 112ms (pass)|
| aicfg2      | AICfg TP2x2, 20 seqs   | 2x2      | 4    | 20       | 5    | 5000     | 732ms (fail)|
| aicfg3      | AICfg TP2x3, 20 seqs   | 2x3      | 6    | 20       | 5    | 5000     | 132ms (pass)|
| llmopt1     | LLMOpt TP2, 4 seqs     | 2x1      | 2    | 4        | 5    | 5000     | 454s (fail)|
| llmopt3     | LLMOpt TP4, 8 seqs     | 4x1      | 4    | 8        | 5    | 5000     | 464s (fail)|

## Additional Runs (ablation / headroom tests)

| Run Name         | Description                       | TP x Rep | Rate | Requests |
|------------------|-----------------------------------|----------|------|----------|
| blis1            | BLIS TP4 seqs=64 (initial)        | 4x1      | 5    | 5000     |
| blis2            | BLIS TP4 seqs=64 (repeat)         | 4x1      | 5    | 5000     |
| blis1-rate05     | BLIS TP4 seqs=64 at 0.5 rps       | 4x1      | 0.5  | 5000     |
| blis3-rate8      | BLIS TP4 seqs=64 at 8 rps         | 4x1      | 8    | 2000     |
| aicfg1           | AICfg TP2x2 (initial)             | 2x2      | 5    | 5000     |
| tp4x1-rate01     | Baseline TP4 at 0.1 rps           | 4x1      | 0.1  | 5000     |
| tp4x1-rate1      | Baseline TP4 at 1 rps             | 4x1      | 1    | 5000     |
| tp4x1-seqs4-rate05 | TP4 seqs=4 at 0.5 rps (isolate batch effect) | 4x1 | 0.5 | 5000 |
| tp4x2-baseline   | TP4x2 (8 GPU) baseline            | 4x2      | 5    | 5000     |

## Trace CSV Schema

Columns in `trace_data.csv`:
- `request_id`: sequential request ID
- `client_id`: workload client (e.g. "codegen")
- `input_tokens`, `output_tokens`: actual token counts
- `arrival_time_us`: request arrival timestamp (microseconds)
- `send_time_us`: request sent to server
- `first_chunk_time_us`: first token received (TTFT = first_chunk - send)
- `last_chunk_time_us`: last token received
- `num_chunks`: streaming chunks received
- `status`: "ok" or error
- `finish_reason`: "stop" or "length"

## Quick Analysis

```python
import pandas as pd

df = pd.read_csv("validation/blis3/shared_prefix_codegen/trace_data.csv")
df["ttft_ms"] = (df["first_chunk_time_us"] - df["send_time_us"]) / 1000
print(f"Mean TTFT: {df['ttft_ms'].mean():.1f} ms")
print(f"P99 TTFT: {df['ttft_ms'].quantile(0.99):.1f} ms")
print(f"Completion: {(df['status']=='ok').mean()*100:.0f}%")
```
