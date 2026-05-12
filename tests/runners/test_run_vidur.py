from experiments.runners.run_vidur import build_vidur_config_yaml


def test_build_config_yaml():
    config = {
        "tp": 2, "pp": 1, "replicas": 2, "max_num_seqs": 128,
        "scheduler_type": "vllm", "max_tokens_in_batch": 4096,
        "chunk_size": None, "block_size": 16, "routing": "round_robin",
    }
    yaml_str = build_vidur_config_yaml(config, model="meta-llama/Llama-3.1-8B")
    assert "tensor_parallel_size" in yaml_str
    assert "vllm" in yaml_str
    assert "round_robin" in yaml_str


def test_build_config_yaml_single_replica():
    config = {
        "tp": 4, "pp": 1, "replicas": 1, "max_num_seqs": 256,
        "scheduler_type": "sarathi", "max_tokens_in_batch": None,
        "chunk_size": 2048, "block_size": 32, "routing": None,
    }
    yaml_str = build_vidur_config_yaml(config, model="meta-llama/Llama-3.1-8B")
    assert "global_scheduler_config" not in yaml_str
    assert "chunk_size" in yaml_str
