def compute_drift(
    predicted: dict,
    actual: dict,
    slo_ttft_mean_ms: float = 300.0,
) -> dict:
    """Compute sim2real drift between predicted and actual measurements."""
    return {
        "drift_ttft_ms": actual["ttft_mean_ms"] - predicted["ttft_mean_ms"],
        "drift_throughput_tok_s": actual["max_throughput_tok_s"] - predicted["max_throughput_tok_s"],
        "drift_ttft_pct": (
            (actual["ttft_mean_ms"] - predicted["ttft_mean_ms"]) / predicted["ttft_mean_ms"] * 100
            if predicted["ttft_mean_ms"] > 0 else 0
        ),
        "drift_throughput_pct": (
            (actual["max_throughput_tok_s"] - predicted["max_throughput_tok_s"])
            / predicted["max_throughput_tok_s"] * 100
            if predicted["max_throughput_tok_s"] > 0 else 0
        ),
        "slo_violation": actual["ttft_mean_ms"] > slo_ttft_mean_ms,
        "predicted_ttft_mean_ms": predicted["ttft_mean_ms"],
        "actual_ttft_mean_ms": actual["ttft_mean_ms"],
        "predicted_throughput_tok_s": predicted["max_throughput_tok_s"],
        "actual_throughput_tok_s": actual["max_throughput_tok_s"],
    }
