# BLIS Config Sweep Space

## Topology

- TP: {1, 2, 4, 8}
- Replicas: {1..floor(max_gpus/TP)}

## Batching

- max_num_seqs: {32, 64, 128, 256, 512}
- max_batched_tokens: {2048, 4096, 8192}
- chunked_prefill_threshold: {0 (disabled), 1024, 2048, 4096}
  - Prune: threshold >= max_batched_tokens

## Scheduling

- Scheduler: {fcfs, priority-fcfs, sjf, reverse-priority}
- Admission policy: {always-admit, tier-shed} (multi-replica only)
- Preemption policy: {fcfs, priority}

## Routing (multi-replica only)

Simple: {round-robin, least-loaded}

Weighted scorer configs:
1. `precise-prefix-cache:2,queue-depth:1,kv-utilization:1`
2. `queue-depth:1,kv-utilization:1`
3. `precise-prefix-cache:2,load-balance:1`

## Other

- Block size: {16, 32}

## Fixed

- Latency model: trained-physics
- GPU memory utilization: 0.9
- dtype: bfloat16
