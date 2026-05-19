from experiments.config.topology import TopologyNoPP, enumerate_topologies_no_pp

MAX_NUM_SEQS_ALL = (32, 64, 128, 256, 512)
MAX_NUM_SEQS_PRUNED = (32, 64, 128, 256)
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


PRIORITY_AWARE_SCHEDULERS = ("priority-fcfs", "reverse-priority")


def _batching_combos(seqs_values: tuple[int, ...] = MAX_NUM_SEQS_ALL) -> list[tuple[int, int, int]]:
    combos = []
    for seqs in seqs_values:
        for tokens in MAX_BATCHED_TOKENS:
            combos.append((seqs, tokens, 0))
            for thresh in CHUNKED_PREFILL_THRESHOLDS:
                if thresh < tokens:
                    combos.append((seqs, tokens, thresh))
    return combos


def _is_valid_multi_replica_policy(sched: str, admission: str, preemption: str) -> bool:
    """Apply Rules 2/3 from the spec. Valid combos:
    - Non-priority schedulers: always-admit + fcfs only
    - Priority-aware schedulers: always-admit + fcfs, OR tier-shed + priority
    """
    if sched not in PRIORITY_AWARE_SCHEDULERS:
        return admission == "always-admit" and preemption == "fcfs"
    if admission == "tier-shed":
        return preemption == "priority"
    if preemption == "priority":
        return admission == "tier-shed"
    return admission == "always-admit" and preemption == "fcfs"


def _is_valid_single_replica_policy(sched: str, preemption: str) -> bool:
    """Apply Rule 3: priority preemption only with priority-aware schedulers."""
    if preemption == "priority" and sched not in PRIORITY_AWARE_SCHEDULERS:
        return False
    return True


def generate_blis_configs(
    max_gpus: int = 8,
    topologies: list[TopologyNoPP] | None = None,
) -> list[dict]:
    """Generate full (unpruned) config space. Kept for backward compatibility."""
    if topologies is None:
        topologies = enumerate_topologies_no_pp(max_gpus)
    batching = _batching_combos(MAX_NUM_SEQS_ALL)
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


def generate_pruned_blis_configs(
    max_gpus: int = 8,
    topologies: list[TopologyNoPP] | None = None,
) -> list[dict]:
    """Generate pruned config space (~30,240 configs).

    Pruning rules:
    1. Drop max_num_seqs=512 (impractical KV cache at ISL mean 512)
    2. tier-shed admission only with priority-aware schedulers
    3. priority preemption only with priority-aware schedulers
    """
    if topologies is None:
        topologies = enumerate_topologies_no_pp(max_gpus)
    batching = _batching_combos(MAX_NUM_SEQS_PRUNED)
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
                            if not _is_valid_multi_replica_policy(sched, admission, preemption):
                                continue
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
                        if not _is_valid_single_replica_policy(sched, preemption):
                            continue
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
