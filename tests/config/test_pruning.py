from experiments.config.pruning import prune_topologies, PruneConfig, PruneResult


def _mock_evaluator_factory(ttft_by_tp: dict):
    def evaluate(topo_dict, config):
        tp = topo_dict["tp"]
        ttft = ttft_by_tp.get(tp, 999.0)
        return ttft, "ok"
    return evaluate


def test_prune_passes_low_ttft():
    from experiments.config.topology import enumerate_topologies_no_pp
    topos = enumerate_topologies_no_pp(max_gpus=8)
    evaluator = _mock_evaluator_factory({1: 100.0, 2: 150.0, 4: 200.0, 8: 250.0})
    config = PruneConfig(slo_ttft_ms=300.0, threshold_multiplier=1.5)
    passing, results = prune_topologies(topos, evaluator, config)
    assert len(passing) == len(topos)


def test_prune_removes_high_ttft():
    from experiments.config.topology import enumerate_topologies_no_pp
    topos = enumerate_topologies_no_pp(max_gpus=8)
    evaluator = _mock_evaluator_factory({1: 100.0, 2: 150.0, 4: 500.0, 8: 600.0})
    config = PruneConfig(slo_ttft_ms=300.0, threshold_multiplier=1.5)
    passing, results = prune_topologies(topos, evaluator, config)
    for t in passing:
        assert t.tp in (1, 2)


def test_prune_handles_errors():
    from experiments.config.topology import enumerate_topologies_no_pp
    topos = enumerate_topologies_no_pp(max_gpus=8)
    def fail_evaluator(topo_dict, config):
        return None, "timeout"
    config = PruneConfig()
    passing, results = prune_topologies(topos, fail_evaluator, config)
    assert len(passing) == 0
    assert all(not r.passes for r in results)


def test_prune_threshold_boundary():
    from experiments.config.topology import TopologyNoPP
    topos = [TopologyNoPP(tp=1, replicas=1)]
    config = PruneConfig(slo_ttft_ms=300.0, threshold_multiplier=1.5)
    threshold = 450.0

    just_below = lambda td, c: (449.0, "ok")
    passing, _ = prune_topologies(topos, just_below, config)
    assert len(passing) == 1

    just_above = lambda td, c: (451.0, "ok")
    passing, _ = prune_topologies(topos, just_above, config)
    assert len(passing) == 0
