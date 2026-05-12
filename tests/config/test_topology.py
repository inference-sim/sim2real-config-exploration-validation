from experiments.config.topology import (
    enumerate_topologies,
    enumerate_topologies_no_pp,
    enumerate_topologies_dp,
)


def test_pp_topologies_count():
    triples = enumerate_topologies(max_gpus=8)
    assert len(triples) == 25


def test_pp_topologies_validity():
    for t in enumerate_topologies(max_gpus=8):
        assert t.tp * t.pp * t.replicas <= 8
        assert t.tp in (1, 2, 4, 8)
        assert t.pp in (1, 2, 4)
        assert t.replicas >= 1


def test_pp_single_replica_count():
    triples = enumerate_topologies(max_gpus=8)
    single = [t for t in triples if t.replicas == 1]
    assert len(single) == 9


def test_pp_multi_replica_count():
    triples = enumerate_topologies(max_gpus=8)
    multi = [t for t in triples if t.replicas > 1]
    assert len(multi) == 16


def test_no_pp_topologies_count():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    assert len(triples) == 15


def test_no_pp_topologies_validity():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    for t in triples:
        assert t.tp * t.replicas <= 8
        assert t.tp in (1, 2, 4, 8)
        assert t.replicas >= 1


def test_no_pp_single_replica_count():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    single = [t for t in triples if t.replicas == 1]
    assert len(single) == 4


def test_no_pp_multi_replica_count():
    triples = enumerate_topologies_no_pp(max_gpus=8)
    multi = [t for t in triples if t.replicas > 1]
    assert len(multi) == 11


def test_dp_topologies_same_count_as_pp():
    dp_triples = enumerate_topologies_dp(max_gpus=8)
    pp_triples = enumerate_topologies(max_gpus=8)
    assert len(dp_triples) == len(pp_triples)


def test_dp_topologies_validity():
    for t in enumerate_topologies_dp(max_gpus=8):
        assert t.tp * t.pp * t.dp <= 8


def test_topology_deterministic_order():
    a = enumerate_topologies(max_gpus=8)
    b = enumerate_topologies(max_gpus=8)
    assert a == b
