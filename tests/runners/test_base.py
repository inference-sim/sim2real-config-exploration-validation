import json
import tempfile
from pathlib import Path

from experiments.runners.base import BaseRunner
from experiments.schema.output import (
    ConfigResult, WorkloadInfo, VllmArgs, Metadata, Results,
)


def _make_workload():
    return WorkloadInfo(
        model="meta-llama/Llama-3.1-8B", hardware="H100_SXM_80GB",
        preset="servegen_m-mid", num_requests=10000,
        isl_mean=512, isl_max=2048, osl_mean=256, osl_max=1024,
        arrival_pattern="poisson", slo_ttft_mean_ms=300, seed=42,
    )


def _make_vllm_args(**overrides):
    defaults = dict(
        tensor_parallel_size=2, pipeline_parallel_size=1,
        num_instances=1, data_parallel_size=1,
        max_num_seqs=128, max_num_batched_tokens=4096,
        enable_chunked_prefill=False, block_size=16,
    )
    defaults.update(overrides)
    return VllmArgs(**defaults)


class DummyRunner(BaseRunner):
    tool_name = "test-tool"
    timeout_seconds = 5

    def evaluate_config(self, config: dict) -> ConfigResult:
        return ConfigResult(
            tool=self.tool_name,
            workload=self.workload,
            vllm_args=_make_vllm_args(),
            results=Results(
                max_throughput_tok_s=100.0, max_throughput_qps=1.0,
                ttft_mean_ms=200.0, meets_slo=True,
                cost_per_hour=6.40, cost_per_1k_tokens=0.01,
            ),
            metadata=Metadata(status="ok", config_hash="abcd1234"),
        )


def test_append_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.jsonl"
        runner = DummyRunner(workload=_make_workload(), output_path=output)
        result = runner.evaluate_config({})
        runner.append_result(result)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["tool"] == "test-tool"


def test_resume_skips_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.jsonl"
        runner = DummyRunner(workload=_make_workload(), output_path=output)
        result = runner.evaluate_config({})
        runner.append_result(result)
        completed = runner.load_completed_hashes()
        assert "abcd1234" in completed


def test_run_batch_skips_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.jsonl"
        runner = DummyRunner(workload=_make_workload(), output_path=output)
        configs = [{"tensor_parallel_size": 2}, {"tensor_parallel_size": 4}]
        runner.run_batch(configs, hash_fn=lambda c: f"hash-{c['tensor_parallel_size']}")
        assert output.read_text().count("\n") == 2

        runner2 = DummyRunner(workload=_make_workload(), output_path=output)
        runner2.run_batch(configs, hash_fn=lambda c: f"hash-{c['tensor_parallel_size']}")
        assert output.read_text().count("\n") == 2
