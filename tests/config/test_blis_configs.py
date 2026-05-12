from experiments.config.blis_configs import generate_blis_configs


def test_generates_configs():
    configs = generate_blis_configs()
    assert len(configs) > 0


def test_all_configs_have_required_fields():
    configs = generate_blis_configs()
    required = {"tp", "replicas", "max_num_seqs", "max_batched_tokens",
                "chunked_prefill_threshold", "block_size", "scheduler",
                "preemption_policy"}
    for c in configs:
        assert required.issubset(c.keys()), f"Missing fields in {c}"


def test_single_replica_configs_have_no_routing():
    configs = generate_blis_configs()
    single = [c for c in configs if c["replicas"] == 1]
    assert len(single) > 0
    for c in single:
        assert c["routing_policy"] is None
        assert c["admission_policy"] is None


def test_multi_replica_configs_have_routing():
    configs = generate_blis_configs()
    multi = [c for c in configs if c["replicas"] > 1]
    assert len(multi) > 0
    for c in multi:
        assert c["routing_policy"] is not None
        assert c["admission_policy"] is not None


def test_no_pp_in_any_config():
    configs = generate_blis_configs()
    for c in configs:
        assert "pp" not in c


def test_chunked_prefill_threshold_pruning():
    configs = generate_blis_configs()
    for c in configs:
        threshold = c["chunked_prefill_threshold"]
        if threshold > 0:
            assert threshold < c["max_batched_tokens"]


def test_config_count_in_expected_range():
    configs = generate_blis_configs()
    assert 90_000 <= len(configs) <= 110_000, f"Got {len(configs)} configs"


def test_accepts_custom_topologies():
    from experiments.config.topology import TopologyNoPP
    topos = [TopologyNoPP(tp=1, replicas=1)]
    configs = generate_blis_configs(topologies=topos)
    assert all(c["tp"] == 1 and c["replicas"] == 1 for c in configs)
