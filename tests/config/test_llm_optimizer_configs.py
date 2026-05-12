from experiments.config.llm_optimizer_configs import (
    generate_llm_optimizer_configs,
    build_grid_search_args,
)


def test_generates_configs():
    configs = generate_llm_optimizer_configs()
    assert len(configs) > 0


def test_uses_dp_not_replicas():
    configs = generate_llm_optimizer_configs()
    for c in configs:
        assert "dp" in c
        assert "replicas" not in c


def test_dp_validity():
    configs = generate_llm_optimizer_configs()
    for c in configs:
        assert c["tp"] * c["pp"] * c["dp"] <= 8


def test_no_routing_config():
    configs = generate_llm_optimizer_configs()
    for c in configs:
        assert c.get("routing_policy") is None


def test_config_count_in_expected_range():
    configs = generate_llm_optimizer_configs()
    assert 10_000 <= len(configs) <= 13_000, f"Got {len(configs)} configs"


def test_grid_search_args_structure():
    args = build_grid_search_args()
    assert "server_args" in args
    assert "client_args" in args
    assert isinstance(args["server_args"], list)
    assert isinstance(args["client_args"], list)


def test_grid_search_args_contain_tp_dp_pairs():
    args = build_grid_search_args()
    server_str = " ".join(args["server_args"])
    assert "tensor_parallel_size" in server_str
