from experiments.config.topology import TopologyDP, enumerate_topologies_dp

MAX_NUM_SEQS = (32, 64, 128, 256, 512)
MAX_BATCHED_TOKENS = (2048, 4096, 8192)
CHUNKED_PREFILL = (True, False)
BLOCK_SIZES = (16, 32)
PREFIX_CACHING = (True, False)
MAX_CONCURRENCY = (32, 64, 128, 256)


def generate_llm_optimizer_configs(
    max_gpus: int = 8,
    topologies: list[TopologyDP] | None = None,
) -> list[dict]:
    if topologies is None:
        topologies = enumerate_topologies_dp(max_gpus)
    configs = []

    for topo in topologies:
        for seqs in MAX_NUM_SEQS:
            for tokens in MAX_BATCHED_TOKENS:
                for chunked in CHUNKED_PREFILL:
                    for block in BLOCK_SIZES:
                        for prefix in PREFIX_CACHING:
                            for concurrency in MAX_CONCURRENCY:
                                configs.append({
                                    "tp": topo.tp,
                                    "pp": topo.pp,
                                    "dp": topo.dp,
                                    "max_num_seqs": seqs,
                                    "max_batched_tokens": tokens,
                                    "enable_chunked_prefill": chunked,
                                    "block_size": block,
                                    "prefix_caching": prefix,
                                    "max_concurrency": concurrency,
                                    "routing_policy": None,
                                })
    return configs


def build_grid_search_args(max_gpus: int = 8) -> dict:
    topologies = enumerate_topologies_dp(max_gpus)
    tp_pp_dp_pairs = [(t.tp, t.pp, t.dp) for t in topologies]
    pairs_str = ",".join(f"({tp},{pp},{dp})" for tp, pp, dp in tp_pp_dp_pairs)

    seqs_str = ",".join(str(s) for s in MAX_NUM_SEQS)
    tokens_str = ",".join(str(t) for t in MAX_BATCHED_TOKENS)
    block_str = ",".join(str(b) for b in BLOCK_SIZES)
    concurrency_str = ",".join(str(c) for c in MAX_CONCURRENCY)

    server_args = [
        f"tensor_parallel_size*pipeline_parallel_size*data_parallel_size=[{pairs_str}]",
        f"max_num_seqs=[{seqs_str}]",
        f"max_num_batched_tokens=[{tokens_str}]",
        "enable_chunked_prefill=[true,false]",
        f"block_size=[{block_str}]",
        "enable_prefix_caching=[true,false]",
    ]

    client_args = [
        f"max_concurrency=[{concurrency_str}]",
        "num_prompts=10000",
        "dataset_name=sharegpt",
    ]

    return {
        "server_args": server_args,
        "client_args": client_args,
    }
