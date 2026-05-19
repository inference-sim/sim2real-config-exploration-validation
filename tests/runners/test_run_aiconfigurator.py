import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.runners.run_aiconfigurator import AIConfiguratorRunner
from experiments.schema.output import WorkloadInfo


WORKLOAD = WorkloadInfo(
    model="meta-llama/Llama-3.1-8B",
    hardware="H100_SXM_80GB",
    preset="chatbot",
    num_requests=10000,
    isl_mean=512,
    isl_max=2048,
    osl_mean=256,
    osl_max=1024,
    arrival_pattern="poisson",
    slo_ttft_mean_ms=300,
    seed=42,
)


def _make_result(rows: list[dict]) -> MagicMock:
    result = MagicMock()
    result.best_configs = {"agg": pd.DataFrame(rows)}
    return result


def test_run_full_sweep_replica_derived_from_total_gpus(tmp_path):
    """replicas = total_gpus // num_total_gpus, not read from DataFrame."""
    # num_total_gpus=1 means each replica uses 1 GPU.
    # When total_gpus=2, replicas = 2 // 1 = 2.
    rows = [{
        "tp": 1, "pp": 1, "num_total_gpus": 1, "bs": 64,
        "ttft": 120.0, "tpot": 15.0,
        "tokens/s": 4000.0, "request_rate": 20.0,
    }]

    with patch("experiments.runners.run_aiconfigurator.cli_default", return_value=_make_result(rows)):
        runner = AIConfiguratorRunner(WORKLOAD, tmp_path / "out.jsonl", max_gpus=2)
        n = runner.run_full_sweep()

    assert n == 2  # total_gpus=1 (replicas=1) and total_gpus=2 (replicas=2)
    lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()

    # last record is for total_gpus=2, so replicas=2
    record = json.loads(lines[-1])
    assert record["vllm_args"]["num_replicas"] == 2
    assert record["vllm_args"]["max_num_seqs"] == 64
    assert record["results"]["ttft_mean_ms"] == 120.0
    assert record["results"]["meets_slo"] is True
    # cluster throughput = 4000 * 2 replicas
    assert record["results"]["max_throughput_tok_s"] == pytest.approx(8000.0)
    # cluster qps = 20 * 2 replicas
    assert record["results"]["max_throughput_qps"] == pytest.approx(40.0)
    # cost = num_gpus * gpu_cost = 2 * 3.20
    assert record["results"]["cost_per_hour"] == pytest.approx(6.40)


def test_run_full_sweep_routing_null_for_single_replica(tmp_path):
    rows = [{
        "tp": 1, "pp": 1, "num_total_gpus": 1, "bs": 128,
        "ttft": 80.0, "tpot": 12.0,
        "tokens/s": 5000.0, "request_rate": 19.5,
    }]
    with patch("experiments.runners.run_aiconfigurator.cli_default", return_value=_make_result(rows)):
        runner = AIConfiguratorRunner(WORKLOAD, tmp_path / "out.jsonl", max_gpus=1)
        runner.run_full_sweep()

    record = json.loads((tmp_path / "out.jsonl").read_text().strip())
    assert record["vllm_args"]["num_replicas"] == 1
    assert record["routing_config"] is None


def test_run_full_sweep_routing_roundrobin_for_multi_replica(tmp_path):
    # num_total_gpus=1, total_gpus=3 -> replicas=3
    rows = [{
        "tp": 1, "pp": 1, "num_total_gpus": 1, "bs": 64,
        "ttft": 100.0, "tpot": 10.0,
        "tokens/s": 3000.0, "request_rate": 11.7,
    }]
    with patch("experiments.runners.run_aiconfigurator.cli_default", return_value=_make_result(rows)):
        runner = AIConfiguratorRunner(WORKLOAD, tmp_path / "out.jsonl", max_gpus=3)
        runner.run_full_sweep()

    lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
    # total_gpus=3 is the last record; replicas=3 -> round-robin
    record = json.loads(lines[-1])
    assert record["vllm_args"]["num_replicas"] == 3
    assert record["routing_config"]["strategy"] == "round-robin"


def test_run_full_sweep_skips_failed_call(tmp_path):
    call_count = 0

    def flaky_cli_default(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs["total_gpus"] == 2:
            raise RuntimeError("simulated failure")
        return _make_result([{
            "tp": 1, "pp": 1, "num_total_gpus": 1, "bs": 32,
            "ttft": 50.0, "tpot": 8.0,
            "tokens/s": 2000.0, "request_rate": 7.8,
        }])

    with patch("experiments.runners.run_aiconfigurator.cli_default", side_effect=flaky_cli_default):
        runner = AIConfiguratorRunner(WORKLOAD, tmp_path / "out.jsonl", max_gpus=3)
        n = runner.run_full_sweep()

    assert call_count == 3
    assert n == 2  # total_gpus=2 skipped; 1 and 3 each write one record


def test_run_full_sweep_empty_df_skipped(tmp_path):
    empty_result = MagicMock()
    empty_result.best_configs = {"agg": pd.DataFrame()}

    with patch("experiments.runners.run_aiconfigurator.cli_default", return_value=empty_result):
        runner = AIConfiguratorRunner(WORKLOAD, tmp_path / "out.jsonl", max_gpus=2)
        n = runner.run_full_sweep()

    assert n == 0
    assert not (tmp_path / "out.jsonl").exists()


def test_run_full_sweep_slo_flag(tmp_path):
    """ttft > slo_ttft_mean_ms sets meets_slo=False."""
    rows = [{
        "tp": 1, "pp": 1, "num_total_gpus": 1, "bs": 512,
        "ttft": 450.0, "tpot": 20.0,
        "tokens/s": 8000.0, "request_rate": 31.0,
    }]
    with patch("experiments.runners.run_aiconfigurator.cli_default", return_value=_make_result(rows)):
        runner = AIConfiguratorRunner(WORKLOAD, tmp_path / "out.jsonl", max_gpus=1)
        runner.run_full_sweep()

    record = json.loads((tmp_path / "out.jsonl").read_text().strip())
    assert record["results"]["meets_slo"] is False
