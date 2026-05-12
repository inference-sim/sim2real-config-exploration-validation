from experiments.analysis.pareto import compute_pareto_front


def _make_point(cost, throughput, meets_slo=True):
    return {"cost_per_hour": cost, "max_throughput_tok_s": throughput, "meets_slo": meets_slo}


def test_simple_pareto():
    points = [
        _make_point(10, 500),
        _make_point(10, 800),
        _make_point(20, 1000),
        _make_point(20, 600),
        _make_point(30, 900),
    ]
    front = compute_pareto_front(points)
    assert len(front) == 2
    costs = {p["cost_per_hour"] for p in front}
    assert costs == {10, 20}


def test_filters_slo_violations():
    points = [
        _make_point(10, 800, meets_slo=True),
        _make_point(5, 1200, meets_slo=False),
    ]
    front = compute_pareto_front(points)
    assert len(front) == 1
    assert front[0]["cost_per_hour"] == 10


def test_empty_input():
    assert compute_pareto_front([]) == []


def test_all_slo_violations():
    points = [_make_point(10, 500, meets_slo=False)]
    assert compute_pareto_front(points) == []


def test_single_point():
    points = [_make_point(10, 500)]
    front = compute_pareto_front(points)
    assert len(front) == 1
