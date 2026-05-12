from experiments.config.vidur_configs import generate_vidur_configs


def test_generates_configs():
    configs = generate_vidur_configs()
    assert len(configs) > 0


def test_scheduler_variants():
    configs = generate_vidur_configs()
    schedulers = {c["scheduler_type"] for c in configs}
    assert "vllm" in schedulers
    assert "sarathi" in schedulers
    assert "orca" in schedulers


def test_vllm_scheduler_has_max_tokens_in_batch():
    configs = generate_vidur_configs()
    vllm_configs = [c for c in configs if c["scheduler_type"] == "vllm"]
    assert all(c["max_tokens_in_batch"] is not None for c in vllm_configs)


def test_sarathi_scheduler_has_chunk_size():
    configs = generate_vidur_configs()
    sarathi_configs = [c for c in configs if c["scheduler_type"] == "sarathi"]
    assert all(c["chunk_size"] is not None for c in sarathi_configs)


def test_orca_scheduler_minimal():
    configs = generate_vidur_configs()
    orca_configs = [c for c in configs if c["scheduler_type"] == "orca"]
    assert all(c.get("max_tokens_in_batch") is None for c in orca_configs)
    assert all(c.get("chunk_size") is None for c in orca_configs)


def test_multi_replica_has_routing():
    configs = generate_vidur_configs()
    multi = [c for c in configs if c["replicas"] > 1]
    assert all(c["routing"] in ("round_robin", "lor", "random") for c in multi)


def test_config_count_in_expected_range():
    configs = generate_vidur_configs()
    assert 3_000 <= len(configs) <= 4_500, f"Got {len(configs)} configs"
