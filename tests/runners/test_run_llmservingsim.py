from experiments.runners.run_llmservingsim import build_llmservingsim_args


def test_build_args_with_chunked_prefill():
    config = {
        "tp": 2, "pp": 1, "replicas": 2, "max_num_seqs": 128,
        "max_batched_tokens": 4096, "enable_chunked_prefill": True,
        "chunked_prefill_threshold": 1024, "block_size": 16,
        "prefix_caching": False, "routing_policy": "LOAD",
    }
    args = build_llmservingsim_args(config, dataset_path="trace.jsonl")
    assert "--enable-chunked-prefill" in args
    assert "--long-prefill-token-threshold" in args


def test_build_args_no_chunked_prefill():
    config = {
        "tp": 1, "pp": 1, "replicas": 1, "max_num_seqs": 64,
        "max_batched_tokens": 2048, "enable_chunked_prefill": False,
        "chunked_prefill_threshold": None, "block_size": 16,
        "prefix_caching": True, "routing_policy": None,
    }
    args = build_llmservingsim_args(config, dataset_path="trace.jsonl")
    assert "--enable-chunked-prefill" not in args
    assert "--enable-prefix-caching" in args
