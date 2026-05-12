from experiments.config.aiconfigurator_configs import generate_aiconfigurator_configs


def test_generates_configs():
    configs = generate_aiconfigurator_configs()
    assert len(configs) > 0


def test_configs_are_topology_triples():
    configs = generate_aiconfigurator_configs()
    required = {"tp", "pp", "replicas"}
    for c in configs:
        assert required.issubset(c.keys())


def test_count_matches_topology():
    configs = generate_aiconfigurator_configs()
    assert len(configs) == 25


def test_no_routing_config():
    configs = generate_aiconfigurator_configs()
    for c in configs:
        assert c.get("routing_policy") is None
