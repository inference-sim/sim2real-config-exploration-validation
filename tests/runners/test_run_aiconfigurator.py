from experiments.runners.run_aiconfigurator import parse_estimate_output
from experiments.schema.output import WorkloadInfo


def test_parse_estimate_output_basic():
    output = """
============================================================
  Performance Estimate (agg)
============================================================
  Model:            meta-llama/Llama-3.1-8B
  System:           h100_sxm
  Backend:          vllm (0.14.0)
------------------------------------------------------------
  ISL:              512
  OSL:              256
  Batch Size:       128
  TP Size:          2
  PP Size:          1
------------------------------------------------------------
  TTFT:             77.347 ms
  TPOT:             14.878 ms
  Request Latency:  3871.247 ms
  Power (per GPU):  0.0 W
------------------------------------------------------------
  tokens/s:         8,539.69
  tokens/s/gpu:     8,539.69
  tokens/s/user:    67.21
  seq/s:            33.489
  Concurrency:      128
  Memory (GPU):     30.53 GB
============================================================
"""
    metrics = parse_estimate_output(output, tp=2, pp=1, batch_size=128)
    assert metrics is not None
    assert metrics["tp"] == 2
    assert metrics["pp"] == 1
    assert metrics["batch_size"] == 128
    assert metrics["ttft_ms"] == 77.347
    assert metrics["tpot_ms"] == 14.878
    assert metrics["tokens_per_sec"] == 8539.69
    assert metrics["qps"] == 33.489


def test_parse_estimate_output_no_data():
    output = "Error: model not supported\n"
    metrics = parse_estimate_output(output, tp=1, pp=1, batch_size=32)
    assert metrics is None
