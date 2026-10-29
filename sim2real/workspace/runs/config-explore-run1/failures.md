# Run Failures: config-explore-run1

## aiconfigurator1 (TP=1, replicas=1, max_num_batched_tokens=245760)
- **Cause:** CUDA OOM. Model + KV cache consumes ~74 GiB, CUDA graph needs 5.25 GiB but only 5.1 GiB free.

## aiconfigurator2 (TP=1, replicas=2, max_num_batched_tokens=245760)
- **Cause:** Same CUDA OOM as aiconfigurator1.

## aiconfigurator3 (TP=1, replicas=3, max_num_batched_tokens=245760)
- **Cause:** Same CUDA OOM as aiconfigurator1.

## llmoptimizer3 (TP=3, replicas=1, max_num_batched_tokens=49152)
- **Cause:** Decode pod crash-looping. Likely TP=3 with max_num_batched_tokens=49152 exceeds available memory, or 3 GPUs not schedulable on a single node.

## Summary
- **Succeeded (5/9):** blis1, blis2, blis3, llmoptimizer1, llmoptimizer2
- **Failed (4/9):** aiconfigurator1-3 (OOM), llmoptimizer3 (crash-loop)
