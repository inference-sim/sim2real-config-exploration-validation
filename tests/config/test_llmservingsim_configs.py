from experiments.config.llmservingsim_configs import generate_llmservingsim_configs


def test_generates_configs():
    configs = generate_llmservingsim_configs()
    assert len(configs) > 0


def test_all_configs_have_required_fields():
    configs = generate_llmservingsim_configs()
    required = {"tp", "pp", "replicas", "max_num_seqs", "max_batched_tokens",
                "enable_chunked_prefill", "chunked_prefill_threshold",
                "block_size", "prefix_caching"}
    for c in configs:
        assert required.issubset(c.keys()), f"Missing: {required - c.keys()}"


def test_multi_replica_has_routing():
    configs = generate_llmservingsim_configs()
    multi = [c for c in configs if c["replicas"] > 1]
    assert all(c["routing_policy"] in ("LOAD", "RR", "RAND") for c in multi)


def test_single_replica_no_routing():
    configs = generate_llmservingsim_configs()
    single = [c for c in configs if c["replicas"] == 1]
    assert all(c.get("routing_policy") is None for c in single)


def test_disabled_chunked_prefill_has_null_threshold():
    configs = generate_llmservingsim_configs()
    for c in configs:
        if not c["enable_chunked_prefill"]:
            assert c["chunked_prefill_threshold"] is None


def test_threshold_pruning():
    configs = generate_llmservingsim_configs()
    for c in configs:
        thresh = c["chunked_prefill_threshold"]
        if thresh is not None and thresh > 0:
            assert thresh < c["max_batched_tokens"]


def test_config_count_in_expected_range():
    configs = generate_llmservingsim_configs()
    assert 12_000 <= len(configs) <= 16_000, f"Got {len(configs)} configs"


def test_accepts_custom_topologies():
    from experiments.config.topology import Topology
    topos = [Topology(tp=2, pp=1, replicas=1)]
    configs = generate_llmservingsim_configs(topologies=topos)
    assert all(c["tp"] == 2 and c["pp"] == 1 for c in configs)
