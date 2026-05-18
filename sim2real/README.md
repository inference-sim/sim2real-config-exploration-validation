# sim2real

Sim2real translation bundle for the config-exploration-validation experiment. Deploys simulator-recommended configurations against real llm-d clusters and collects latency metrics for drift analysis.

## Prerequisites

- A local copy of the sim2real pipeline repo (https://github.com/inference-sim/sim2real) 
- Python 3.11+
- Cluster access configured for llm-d-benchmark

## 1. Generate Baselines

From this directory, run the scenario generator against the parent repo's selection results:

```bash
cd sim2real
python generate_scenarios.py ../results/processed/top3_selection.json baselines
```

This produces scenario YAML files in `baselines/` (one per estimator recommendation). See `generate_scenarios.README.md` for field mapping details.

_Note: if new fields have been introduced into the input, they will be flagged. In thi case, it may be necessary to modify `generate_scenarios.py`_

## 2. Create transfer.yaml

Create `transfer.yaml` at the bundle root to define which baselines and workloads to include:

```yaml
kind: sim2real-transfer
version: 3

scenario: config-exploration-validation

baselines:
  - name: blis1
    scenario: baselines/blis-1.yaml
  - name: blis2
    scenario: baselines/blis-2.yaml
  - name: blis3
    scenario: baselines/blis-3.yaml

workloads:
  - workloads/w1_mid.yaml
```

Each baseline entry references a scenario file generated in step 1. Workloads define the traffic patterns applied during validation.

_Note: baseline names cannot contain hyphens._

## 3. Run the Pipeline

### Setup

Set environment variables

```bash
export SIM2REAL=/path/to/chekout/of/inference-sim/sim2real

export NAMESPACES="comma,separated,list,of,namespaces"
export HF_TOKEN=<huggingface token with access to required models>

export RUN=<run_name>
```

Set up Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r $SIM2REAL/requirements.txt
```

### Initialize the run

```bash
python $SIM2REAL/pipeline/setup.py --run $RUN [--no-cluster]
```

_Note: Accept defaults for any options.

Where `$RUN` is the run identifier (e.g., `run-001`).

### Prepare artifacts

```bash
python $SIM2REAL/pipeline/prepare.py [assemble]
```

### Deploy and execute

Run all workloads remotely:

```bash
python $SIM2REAL/pipeline/deploy.py run --skip-build-epp --remote
```

A subset of pairs can be run using the `--only`, `--packages` and `--workload` options.

Follow real-time progress, if desired:

```shell
k logs -f job/sim2real-orchestrator
```

See the status of all pairs:

```shell
python $SIM2REAL/pipeline/deploy.py status
```

At any time you can collect results from completed executions. This only collects data that has not already been collected:

```shell
python $SIM2REAL/pipeline/deploy.py collect
```

## 4. Analyze Results

Launch Claude Code with access to the sim2real repo and invoke the analysis skill:

```bash
claude --dangerously-skip-permissions --add-dir $SIM2REAL
```

Then use the `/sim2real-analyze` command to generate per-workload latency comparison tables (TTFT/TPOT/E2E baseline vs treatment), charts, and HTML reports.

Corresponding simulation data is available in `../results/raw/`.

## Troubleshooting

Follow progress of tasks:

```shell
watch tkn tr list -n <namespace>
```

Prevent llmdbenchmark teardown. This allows you to debug the infernce engines.

```shell
python $SIM2REAL/pipeline/deploy.py run --skip-build-epp --remote --skip-teardown
```

Stop orchestration:

```shell
python $SIM2REAL/pipeline/deploy.py stop
```

Cleanup any cluster artifacts (`PipelineRun` objects and deployed helm charts):

```shell
python $SIM2REAL/pipeline/deploy.py reset
```

Erase any collected data:

```shell
python $SIM2REAL/pipeline/deploy.py wipe
```
