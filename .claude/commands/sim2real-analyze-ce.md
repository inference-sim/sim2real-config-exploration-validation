# sim2real-analyze-ce

Generate a predicted vs. real Pareto chart comparing simulator predictions against actual measurements from a sim2real config-exploration run.

## Inputs

- `results/processed/top3_selection.json` — simulator predictions (per-tool top-3 configs)
- `sim2real/workspace/runs/<run>/results/` — collected real measurement traces (trace_data.csv per baseline)

## Steps

1. Load predicted metrics from `top3_selection.json` (mean TTFT, max throughput per config)
2. Load real measurements from each baseline's `trace_data.csv`:
   - Compute mean TTFT: mean of `(first_chunk_time_us - send_time_us)` across all ok requests, converted to ms
   - Compute max throughput: `num_ok_requests / duration_seconds` in tokens/sec using total output tokens and wall-clock time
3. Generate a Pareto chart (mean TTFT x-axis vs. max throughput y-axis) with:
   - Hollow markers for predicted points (same style as top3_pareto.png: circles for BLIS, triangles-up for AIConfigurator, triangles-down for llm-optimizer)
   - Filled markers for real measurements at the same positions
   - Drift arrows from predicted to real for each config
   - Red X on points where real TTFT violates SLO (>300ms)
   - SLO vertical line at 300ms
   - Pareto frontier line connecting predicted points per tool
   - Annotations showing TP=N R=M and cost/hr
4. Save chart to `sim2real/workspace/runs/<run>/sim2real_pareto.png` and `.pdf`

## Chart Style

Match the style from `results/figures/top3_pareto.png`:
- Colors: BLIS=#1f77b4, AIConfigurator=#2ca02c, llm-optimizer=#9467bd
- Markers: BLIS=circle, AIConfigurator=triangle-up, llm-optimizer=triangle-down
- Frontier lines connecting each tool's points
- Linear scale on both axes
- Annotations: "TP=N R=M\n$X/hr" near each point
- Title: "Sim2Real: Predicted vs. Actual (Top-3 per Tool)"
- Legend with tool names and predicted/actual distinction

## Mapping baselines to predictions

Map baseline names to top3_selection.json entries by tool group and index:
- blis1/blis2/blis3 → top3_selection["blis"][0/1/2]
- aiconfigurator1/2/3 → top3_selection["aiconfigurator"][0/1/2]
- llmoptimizer1/2/3 → top3_selection["llm-optimizer"][0/1/2]

Only plot baselines that have collected results. Note failed baselines in the chart as "(OOM)" or "(failed)" in the legend.

## Computing real metrics from trace_data.csv

```python
import pandas as pd

df = pd.read_csv(trace_path)
df = df[df["status"] == "ok"]

# TTFT in ms (time_unit is microseconds)
df["ttft_us"] = df["first_chunk_time_us"] - df["send_time_us"]
mean_ttft_ms = df["ttft_us"].mean() / 1000.0

# Throughput: total output tokens / wall-clock duration
total_output_tokens = df["output_tokens"].sum()
duration_s = (df["last_chunk_time_us"].max() - df["send_time_us"].min()) / 1e6
max_throughput_tok_s = total_output_tokens / duration_s
```

## Output

After generating the chart, display it and print a summary table:

| Baseline | TP | R | Predicted TTFT | Real TTFT | Predicted Tput | Real Tput | TTFT Drift |
|----------|----|----|---------------|-----------|----------------|-----------|------------|

Include drift percentage: `(real - predicted) / predicted * 100`
