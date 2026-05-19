export SIM2REAL=/Users/jchen/go/src/inference-sim/sim2real
export NAMESPACES=jchen
export RUN=config-explore-run1

# Initialize
python $SIM2REAL/pipeline/setup.py --run $RUN

# Prepare artifacts
python $SIM2REAL/pipeline/prepare.py assemble

# Deploy and run all baselines against all workloads
python $SIM2REAL/pipeline/deploy.py run --skip-build-epp --remote

# Run only the chatbot workload
python $SIM2REAL/pipeline/deploy.py run --skip-build-epp --remote --workload workloads/chatbot.yaml

# Status
python $SIM2REAL/pipeline/deploy.py status

# Collect results
python $SIM2REAL/pipeline/deploy.py collect
