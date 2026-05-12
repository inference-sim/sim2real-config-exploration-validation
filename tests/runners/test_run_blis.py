from experiments.runners.run_blis import build_blis_args


def test_build_blis_args_multi_replica():
    config = {
        "tp": 2, "replicas": 4, "max_num_seqs": 128,
        "max_batched_tokens": 4096, "chunked_prefill_threshold": 1024,
        "block_size": 16, "scheduler": "priority-fcfs",
        "admission_policy": "tier-shed", "preemption_policy": "priority",
        "routing_policy": "weighted",
        "routing_scorers": "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=50.0)
    assert "--model" in args
    assert "--tp" in args
    assert "--num-instances" in args
    assert "--routing-scorers" in args


def test_build_blis_args_single_replica():
    config = {
        "tp": 4, "replicas": 1, "max_num_seqs": 256,
        "max_batched_tokens": 8192, "chunked_prefill_threshold": 0,
        "block_size": 32, "scheduler": "fcfs",
        "admission_policy": None, "preemption_policy": "fcfs",
        "routing_policy": None, "routing_scorers": None,
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=10.0)
    assert "--routing-policy" not in args
    assert "--admission-policy" not in args


def test_build_blis_args_disabled_chunked_prefill():
    config = {
        "tp": 1, "replicas": 1, "max_num_seqs": 64,
        "max_batched_tokens": 2048, "chunked_prefill_threshold": 0,
        "block_size": 16, "scheduler": "fcfs",
        "admission_policy": None, "preemption_policy": "fcfs",
        "routing_policy": None, "routing_scorers": None,
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=10.0)
    assert "--long-prefill-token-threshold" in args
    idx = args.index("--long-prefill-token-threshold")
    assert args[idx + 1] == "0"
