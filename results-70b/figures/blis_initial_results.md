## Initial Results: BLIS on llm-d (Llama-3.1-70B, H100)

We deployed BLIS-recommended configurations on llm-d and validated against real traffic at 10 QPS Poisson. BLIS's best 4-GPU config (TP4x1, max-num-seqs=512) achieved 15,056 tok/s estimated max throughput with TTFT=156ms, while its 8-GPU config (TP8x1) achieved 13,155 tok/s with TTFT=197ms. Throughput predictions were within 15-18% of actual measurements.

For comparison, AIConfigurator's best stable config required 8 GPUs (TP4x2, max-num-seqs=96) and achieved 8,188 tok/s, 1.8x lower throughput than BLIS at the same or greater GPU cost. llm-optimizer's config (max-num-seqs=8) failed entirely with 68% request timeouts. The critical differentiator is parameter coverage: BLIS explores batch sizes up to 512, matching real vLLM continuous batching behavior, while the other tools are limited to smaller batch sizes (96 and 8 respectively) that cannot sustain moderate concurrency.

Note: this comparison reflects engine-level parameters only (tensor parallelism, batch size, KV cache blocks). BLIS models more knobs than the other tools, which is precisely the source of its advantage in config exploration.
