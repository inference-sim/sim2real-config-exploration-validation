from experiments.config.topology import Topology, enumerate_topologies

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
MAX_BATCHED_TOKENS = (2048, 4096, 8192)
CHUNKED_PREFILL_THRESHOLDS_ENABLED = (0, 1024, 2048, 4096)
BLOCK_SIZES = (16, 32)
PREFIX_CACHING = (True, False)
ROUTING_POLICIES = ("LOAD", "RR", "RAND")


def generate_llmservingsim_configs(
    max_gpus: int = 8,
    topologies: list[Topology] | None = None,
) -> list[dict]:
    if topologies is None:
        topologies = enumerate_topologies(max_gpus)
    configs = []

    for topo in topologies:
        for seqs in MAX_NUM_SEQS:
            for tokens in MAX_BATCHED_TOKENS:
                for block in BLOCK_SIZES:
                    for prefix in PREFIX_CACHING:
                        # Disabled chunked prefill
                        base = {
                            "tp": topo.tp,
                            "pp": topo.pp,
                            "replicas": topo.replicas,
                            "max_num_seqs": seqs,
                            "max_batched_tokens": tokens,
                            "enable_chunked_prefill": False,
                            "chunked_prefill_threshold": None,
                            "block_size": block,
                            "prefix_caching": prefix,
                        }
                        if topo.is_multi_replica:
                            for routing in ROUTING_POLICIES:
                                configs.append({**base, "routing_policy": routing})
                        else:
                            configs.append({**base, "routing_policy": None})

                        # Enabled chunked prefill with each threshold
                        for thresh in CHUNKED_PREFILL_THRESHOLDS_ENABLED:
                            if thresh > 0 and thresh >= tokens:
                                continue
                            enabled_base = {
                                **base,
                                "enable_chunked_prefill": True,
                                "chunked_prefill_threshold": thresh,
                            }
                            if topo.is_multi_replica:
                                for routing in ROUTING_POLICIES:
                                    configs.append({**enabled_base, "routing_policy": routing})
                            else:
                                configs.append({**enabled_base, "routing_policy": None})
    return configs
