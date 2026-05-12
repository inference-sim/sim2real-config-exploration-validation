from experiments.analysis.drift import compute_drift


def test_compute_drift():
    predicted = {"ttft_mean_ms": 200.0, "max_throughput_tok_s": 800.0}
    actual = {"ttft_mean_ms": 250.0, "max_throughput_tok_s": 720.0}
    drift = compute_drift(predicted, actual, slo_ttft_mean_ms=300)
    assert drift["drift_ttft_ms"] == 50.0
    assert drift["drift_throughput_tok_s"] == -80.0
    assert drift["slo_violation"] is False


def test_slo_violation_detected():
    predicted = {"ttft_mean_ms": 280.0, "max_throughput_tok_s": 500.0}
    actual = {"ttft_mean_ms": 320.0, "max_throughput_tok_s": 450.0}
    drift = compute_drift(predicted, actual, slo_ttft_mean_ms=300)
    assert drift["slo_violation"] is True


def test_negative_drift_means_better():
    predicted = {"ttft_mean_ms": 250.0, "max_throughput_tok_s": 600.0}
    actual = {"ttft_mean_ms": 220.0, "max_throughput_tok_s": 650.0}
    drift = compute_drift(predicted, actual, slo_ttft_mean_ms=300)
    assert drift["drift_ttft_ms"] == -30.0
    assert drift["drift_throughput_tok_s"] == 50.0
