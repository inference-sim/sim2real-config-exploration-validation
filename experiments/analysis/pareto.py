def _get_results(p: dict) -> dict:
    """Access results whether the record is flat or nested."""
    if "results" in p and isinstance(p["results"], dict):
        return p["results"]
    return p


def compute_pareto_front(points: list[dict]) -> list[dict]:
    """Compute Pareto front from full config records.

    Minimizes cost_per_hour, maximizes max_throughput_tok_s.
    Filters to SLO-meeting configs first.
    """
    viable = [p for p in points if _get_results(p).get("meets_slo", False)]
    if not viable:
        return []

    viable.sort(key=lambda p: (
        _get_results(p)["cost_per_hour"],
        -_get_results(p)["max_throughput_tok_s"],
    ))

    front = []
    max_throughput = float("-inf")
    for p in viable:
        if _get_results(p)["max_throughput_tok_s"] > max_throughput:
            front.append(p)
            max_throughput = _get_results(p)["max_throughput_tok_s"]

    return front
