**Run Summary: `config-explore-llama2-70b`**
Generated: 2026-06-18T17:30:02+00:00 | Scenario: config-exploration-llama2-70b

**Mode:** Baseline-only (blis1)

**Config Under Test:**
- Model: meta-llama/Llama-2-70b-hf
- vLLM: v0.15.1
- TP=4, replicas=1, max_num_seqs=64, max_num_batched_tokens=8192
- Routing: weighted-scoring (kv-utilization:5, load-balance:2, prefix-cache:3, queue-depth:2)
- Scheduler: sjf
- BLIS predicted TTFT: 46.3ms (target SLO: 50ms)

**Packages**

- `sim2real/workspace/runs/config-explore-llama2-70b/cluster/pipelinerun-shared-prefix-codegen-blis1.yaml`

**Workloads**

- shared_prefix_codegen (rate=10 rps, ISL=566, OSL=247)

**Checklist**
- [-] Translation skipped (baseline-only)
- [x] Assembly complete
- [ ] Deploy to cluster
- [ ] Collect results
