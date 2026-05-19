import tempfile
from pathlib import Path

from experiments.runners.run_vidur import VidurRunner
from experiments.schema.output import WorkloadInfo


def _make_workload(**overrides):
    defaults = dict(
        model="meta-llama/Llama-3.1-8B", hardware="H100_SXM_80GB",
        preset="chatbot", num_requests=10000,
        isl_mean=512, isl_max=2048, osl_mean=256, osl_max=1024,
        arrival_pattern="poisson", slo_ttft_mean_ms=300, seed=42,
    )
    defaults.update(overrides)
    return WorkloadInfo(**defaults)


def test_build_vidur_args_vllm_scheduler():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 4, "pp": 1, "replicas": 2,
            "scheduler": "vllm", "max_num_seqs": 256,
            "max_batched_tokens": 8192, "routing": "round_robin",
        }
        args = runner._build_vidur_args(config, qps=20.0, output_dir="/tmp/test")

        assert "--replica_config_device" in args
        assert "h100" in args
        assert "--replica_config_tensor_parallel_size" in args
        assert "4" in args
        assert "--vllm_scheduler_config_batch_size_cap" in args
        assert "256" in args
        assert "--vllm_scheduler_config_max_tokens_in_batch" in args
        assert "8192" in args
        assert "--global_scheduler_config_type" in args
        assert "round_robin" in args
        assert "--poisson_request_interval_generator_config_qps" in args
        assert "20.00" in args


def test_build_vidur_args_sarathi_scheduler():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 8, "pp": 1, "replicas": 1,
            "scheduler": "sarathi", "max_num_seqs": 512,
            "chunk_size": 4096,
        }
        args = runner._build_vidur_args(config, qps=10.0, output_dir="/tmp/test")

        assert "--sarathi_scheduler_config_batch_size_cap" in args
        assert "512" in args
        assert "--sarathi_scheduler_config_chunk_size" in args
        assert "4096" in args
        assert "--vllm_scheduler_config_batch_size_cap" not in args
        assert "--global_scheduler_config_type" not in args


def test_build_vidur_args_no_routing_single_replica():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 2, "pp": 1, "replicas": 1,
            "scheduler": "vllm", "max_num_seqs": 128,
            "max_batched_tokens": 4096,
        }
        args = runner._build_vidur_args(config, qps=5.0, output_dir="/tmp/test")
        assert "--global_scheduler_config_type" not in args


def test_parse_request_metrics():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Request Id,request_e2e_time,prefill_e2e_time,request_num_prefill_tokens\n")
        f.write("0,100.0,0.150,512\n")
        f.write("1,95.0,0.200,512\n")
        f.write("2,110.0,0.250,512\n")
        f.flush()

        result = VidurRunner._parse_request_metrics(Path(f.name))
        assert result is not None
        assert abs(result["ttft_mean_ms"] - 200.0) < 1.0
        assert result["ttft_p50_ms"] > 0
        assert result["ttft_p99_ms"] >= result["ttft_p50_ms"]


def test_parse_request_metrics_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Request Id,request_e2e_time,prefill_e2e_time,request_num_prefill_tokens\n")
        f.flush()
        result = VidurRunner._parse_request_metrics(Path(f.name))
        assert result is None


def test_enumerate_configs_lean():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
            lean=True,
        )
        configs = runner._enumerate_configs()
        assert len(configs) > 10
        assert len(configs) < 100

        for c in configs:
            assert c["tp"] * c["pp"] * c["replicas"] <= 8
            assert c["scheduler"] in ("vllm", "sarathi")


def test_enumerate_configs_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
            lean=False,
        )
        configs = runner._enumerate_configs()
        assert len(configs) > 100

        schedulers = {c["scheduler"] for c in configs}
        assert "orca" in schedulers


def test_config_to_vllm_args_sarathi():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 4, "pp": 2, "replicas": 1,
            "scheduler": "sarathi", "max_num_seqs": 256,
            "chunk_size": 2048,
        }
        vllm_args = runner._config_to_vllm_args(config)
        assert vllm_args.tensor_parallel_size == 4
        assert vllm_args.pipeline_parallel_size == 2
        assert vllm_args.enable_chunked_prefill is True
        assert vllm_args.max_num_batched_tokens == 2048


def test_config_to_routing():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = VidurRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config_multi = {"tp": 2, "pp": 1, "replicas": 4, "routing": "lor"}
        routing = runner._config_to_routing(config_multi)
        assert routing is not None
        assert routing.strategy == "least-outstanding"

        config_single = {"tp": 8, "pp": 1, "replicas": 1}
        assert runner._config_to_routing(config_single) is None
