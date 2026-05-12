from experiments.analysis.select import select_top_k


def _make_point(cost, throughput):
    return {"cost_per_hour": cost, "max_throughput_tok_s": throughput, "meets_slo": True}


def test_selects_cheapest_3():
    front = [
        _make_point(6.40, 300),
        _make_point(12.80, 600),
        _make_point(19.20, 900),
        _make_point(25.60, 1100),
    ]
    selected = select_top_k(front, k=3, min_throughput=200)
    assert len(selected) == 3
    assert selected[0]["cost_per_hour"] == 6.40


def test_filters_below_min_throughput():
    front = [
        _make_point(6.40, 50),
        _make_point(12.80, 600),
        _make_point(19.20, 900),
    ]
    selected = select_top_k(front, k=3, min_throughput=200)
    assert len(selected) == 2


def test_fewer_than_k():
    front = [_make_point(6.40, 300)]
    selected = select_top_k(front, k=3, min_throughput=200)
    assert len(selected) == 1


def test_empty_front():
    assert select_top_k([], k=3, min_throughput=200) == []
