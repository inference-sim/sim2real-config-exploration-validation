from experiments.config.topology import TopologyNoPP, enumerate_topologies_no_pp

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
MAX_BATCHED_TOKENS = (2048, 4096, 8192)
CHUNKED_PREFILL_THRESHOLDS = (1024, 2048, 4096)
BLOCK_SIZES = (16, 32)
SCHEDULERS = ("fcfs", "priority-fcfs", "sjf", "reverse-priority")
ADMISSION_POLICIES = ("always-admit", "tier-shed")
PREEMPTION_POLICIES = ("fcfs", "priority")
SIMPLE_ROUTING = ("round-robin", "least-loaded")
WEIGHTED_ROUTING = (
    "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    "queue-depth:1,kv-utilization:1",
    "precise-prefix-cache:2,load-balance:1",
    "vllm-dp:1",
)


def _batching_combos() -> list[tuple[int, int, int]]:
    combos = []
    for seqs in MAX_NUM_SEQS:
        for tokens in MAX_BATCHED_TOKENS:
            combos.append((seqs, tokens, 0))
            for thresh in CHUNKED_PREFILL_THRESHOLDS:
                if thresh < tokens:
                    combos.append((seqs, tokens, thresh))
    return combos


def generate_blis_configs(
    max_gpus: int = 8,
    topologies: list[TopologyNoPP] | None = None,
) -> list[dict]:
    if topologies is None:
        topologies = enumerate_topologies_no_pp(max_gpus)
    batching = _batching_combos()
    configs = []

    for topo in topologies:
        for seqs, tokens, threshold in batching:
            base = {
                "tp": topo.tp,
                "replicas": topo.replicas,
                "max_num_seqs": seqs,
                "max_batched_tokens": tokens,
                "chunked_prefill_threshold": threshold,
            }
            if topo.is_multi_replica:
                for sched in SCHEDULERS:
                    for admission in ADMISSION_POLICIES:
                        for preemption in PREEMPTION_POLICIES:
                            for block in BLOCK_SIZES:
                                for policy in SIMPLE_ROUTING:
                                    configs.append({
                                        **base,
                                        "block_size": block,
                                        "scheduler": sched,
                                        "admission_policy": admission,
                                        "preemption_policy": preemption,
                                        "routing_policy": policy,
                                        "routing_scorers": None,
                                    })
                                for scorers in WEIGHTED_ROUTING:
                                    configs.append({
                                        **base,
                                        "block_size": block,
                                        "scheduler": sched,
                                        "admission_policy": admission,
                                        "preemption_policy": preemption,
                                        "routing_policy": "weighted",
                                        "routing_scorers": scorers,
                                    })
            else:
                for sched in SCHEDULERS:
                    for preemption in PREEMPTION_POLICIES:
                        for block in BLOCK_SIZES:
                            configs.append({
                                **base,
                                "block_size": block,
                                "scheduler": sched,
                                "admission_policy": None,
                                "preemption_policy": preemption,
                                "routing_policy": None,
                                "routing_scorers": None,
                            })
    return configs
