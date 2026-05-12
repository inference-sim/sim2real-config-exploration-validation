def _get_results(p: dict) -> dict:
    """Access results whether the record is flat or nested."""
    if "results" in p and isinstance(p["results"], dict):
        return p["results"]
    return p


def select_top_k(
    pareto_front: list[dict],
    k: int = 3,
    min_throughput: float = 200.0,
) -> list[dict]:
    """Select top-k cheapest configs from Pareto front above min throughput."""
    eligible = [
        p for p in pareto_front
        if _get_results(p)["max_throughput_tok_s"] >= min_throughput
    ]
    eligible.sort(key=lambda p: _get_results(p)["cost_per_hour"])
    return eligible[:k]
