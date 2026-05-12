from experiments.runners.run_llm_optimizer import (
    parse_estimate_output, build_llm_optimizer_cmd,
)


def test_parse_estimate_output_full():
    output = """
=== Performance Analysis ===

Best Throughput (unconstrained):
  Output: 4536.4 tokens/s
  Requests: 47.34 req/s

Performance under Constraints (ttft:mean<300ms):
  Concurrency: 16
  TTFT: 212.4 ms
  ITL: 3.5 ms
  Output throughput: 4200.0 tokens/s

Empirical Optimal Concurrency: 24
"""
    metrics = parse_estimate_output(output, num_gpus=1)
    assert metrics is not None
    assert metrics["max_throughput_tok_s"] == 4536.4
    assert metrics["max_throughput_qps"] == 47.34
    assert metrics["ttft_mean_ms"] == 212.4
    assert metrics["tpot_mean_ms"] == 3.5
    assert metrics["constrained_throughput_tok_s"] == 4200.0
    assert metrics["concurrency"] == 16
    assert metrics["optimal_concurrency"] == 24
    assert metrics["num_gpus"] == 1


def test_parse_estimate_output_no_data():
    output = "Error: model not found\n"
    metrics = parse_estimate_output(output, num_gpus=2)
    assert metrics is None


def test_parse_estimate_output_partial():
    output = """
Best Throughput (unconstrained):
  Output: 8000.0 tokens/s
  Requests: 80.0 req/s
"""
    metrics = parse_estimate_output(output, num_gpus=2)
    assert metrics is not None
    assert metrics["max_throughput_tok_s"] == 8000.0
    assert metrics["num_gpus"] == 2


def test_build_cmd():
    cmd = build_llm_optimizer_cmd(
        model="meta-llama/Llama-3.1-8B",
        output_json="results/raw/llm_optimizer.json",
    )
    assert "llm-optimizer" in cmd[0]
    assert "--framework" in cmd
    assert "--constraints" in cmd
    assert "--continue" in cmd
    assert "--server-args" in cmd
    assert "--client-args" in cmd
