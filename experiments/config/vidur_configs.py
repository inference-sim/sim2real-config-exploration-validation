from experiments.config.topology import Topology, enumerate_topologies

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
BLOCK_SIZES = (16, 32)
VLLM_MAX_TOKENS = (2048, 4096, 8192)
SARATHI_CHUNK_SIZES = (1024, 2048, 4096)
ROUTING_POLICIES = ("round_robin", "lor", "random")


def _scheduler_variants() -> list[dict]:
    variants = []
    for tokens in VLLM_MAX_TOKENS:
        variants.append({
            "scheduler_type": "vllm",
            "max_tokens_in_batch": tokens,
            "chunk_size": None,
        })
    for chunk in SARATHI_CHUNK_SIZES:
        variants.append({
            "scheduler_type": "sarathi",
            "max_tokens_in_batch": None,
            "chunk_size": chunk,
        })
    variants.append({
        "scheduler_type": "orca",
        "max_tokens_in_batch": None,
        "chunk_size": None,
    })
    return variants


def generate_vidur_configs(
    max_gpus: int = 8,
    topologies: list[Topology] | None = None,
) -> list[dict]:
    if topologies is None:
        topologies = enumerate_topologies(max_gpus)
    scheduler_variants = _scheduler_variants()
    configs = []

    for topo in topologies:
        for seqs in MAX_NUM_SEQS:
            for sched in scheduler_variants:
                for block in BLOCK_SIZES:
                    base = {
                        "tp": topo.tp,
                        "pp": topo.pp,
                        "replicas": topo.replicas,
                        "max_num_seqs": seqs,
                        "block_size": block,
                        **sched,
                    }
                    if topo.is_multi_replica:
                        for routing in ROUTING_POLICIES:
                            configs.append({**base, "routing": routing})
                    else:
                        configs.append({**base, "routing": None})
    return configs
