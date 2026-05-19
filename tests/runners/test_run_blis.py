import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from experiments.runners.run_blis import build_blis_args, BLISRunner
from experiments.schema.output import WorkloadInfo


def _make_workload(**overrides):
    defaults = dict(
        model="meta-llama/Llama-3.1-8B", hardware="H100_SXM_80GB",
        preset="chatbot", num_requests=1000,
        isl_mean=512, isl_max=2048, osl_mean=256, osl_max=1024,
        arrival_pattern="poisson", slo_ttft_mean_ms=300, seed=42,
    )
    defaults.update(overrides)
    return WorkloadInfo(**defaults)


def test_build_blis_args_multi_replica():
    config = {
        "tp": 2, "replicas": 4, "max_num_seqs": 128,
        "max_batched_tokens": 4096, "chunked_prefill_threshold": 1024,
        "block_size": 16, "scheduler": "priority-fcfs",
        "admission_policy": "tier-shed", "preemption_policy": "priority",
        "routing_policy": "weighted",
        "routing_scorers": "precise-prefix-cache:2,queue-depth:1,kv-utilization:1",
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=50.0)
    assert "--model" in args
    assert "--tp" in args
    assert "--num-instances" in args
    assert "--routing-scorers" in args


def test_build_blis_args_single_replica():
    config = {
        "tp": 4, "replicas": 1, "max_num_seqs": 256,
        "max_batched_tokens": 8192, "chunked_prefill_threshold": 0,
        "block_size": 32, "scheduler": "fcfs",
        "admission_policy": None, "preemption_policy": "fcfs",
        "routing_policy": None, "routing_scorers": None,
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=10.0)
    assert "--routing-policy" not in args
    assert "--admission-policy" not in args


def test_build_blis_args_disabled_chunked_prefill():
    config = {
        "tp": 1, "replicas": 1, "max_num_seqs": 64,
        "max_batched_tokens": 2048, "chunked_prefill_threshold": 0,
        "block_size": 16, "scheduler": "fcfs",
        "admission_policy": None, "preemption_policy": "fcfs",
        "routing_policy": None, "routing_scorers": None,
    }
    args = build_blis_args(config, model="meta-llama/Llama-3.1-8B", rate=10.0)
    assert "--long-prefill-token-threshold" in args
    idx = args.index("--long-prefill-token-threshold")
    assert args[idx + 1] == "0"


# ── Search mode tests ─────────────────────────────────────────────────────


def test_build_search_args_includes_workload_and_fitness():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BLISRunner(
            workload=_make_workload(preset="chatbot"),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 4, "replicas": 2, "max_num_seqs": 512,
            "max_batched_tokens": 8192, "chunked_prefill_threshold": 4096,
            "block_size": 16, "scheduler": "fcfs",
            "routing_policy": "least-loaded", "routing_scorers": None,
            "admission_policy": "always-admit", "preemption_policy": "fcfs",
        }
        args = runner._build_search_args(config, rate=10.0, num_requests=1000, seed=42)

        assert "--workload" in args
        idx = args.index("--workload")
        assert args[idx + 1] == "chatbot"

        assert "--fitness-weights" in args
        idx = args.index("--fitness-weights")
        assert "throughput" in args[idx + 1]


def test_build_search_args_omits_workload_for_distribution():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BLISRunner(
            workload=_make_workload(preset="distribution"),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 8, "replicas": 1, "max_num_seqs": 256,
            "max_batched_tokens": 4096, "chunked_prefill_threshold": 1024,
            "block_size": 16, "scheduler": "fcfs",
            "routing_policy": None, "routing_scorers": None,
            "admission_policy": None, "preemption_policy": "fcfs",
        }
        args = runner._build_search_args(config, rate=10.0, num_requests=1000, seed=42)
        assert "--workload" not in args


def test_parse_cluster_metrics_single_block():
    stdout = """
Some output
=== Simulation Metrics ===
{"instance_id": "cluster", "tokens_per_sec": 5000, "ttft_mean_ms": 120.5, "responses_per_sec": 8.2}
Done.
"""
    metrics = BLISRunner._parse_cluster_metrics(stdout)
    assert metrics["tokens_per_sec"] == 5000
    assert metrics["ttft_mean_ms"] == 120.5


def test_parse_cluster_metrics_multi_block():
    stdout = """
=== Simulation Metrics ===
{"instance_id": "inst-0", "tokens_per_sec": 2500}
=== Simulation Metrics ===
{"instance_id": "cluster", "tokens_per_sec": 5000, "ttft_mean_ms": 100.0}
"""
    metrics = BLISRunner._parse_cluster_metrics(stdout)
    assert metrics["instance_id"] == "cluster"
    assert metrics["tokens_per_sec"] == 5000


def test_parse_cluster_metrics_empty():
    assert BLISRunner._parse_cluster_metrics("no metrics here") == {}


def test_make_bracket_profiles_k1():
    profiles = BLISRunner._make_bracket_profiles(1)
    assert len(profiles) == 1
    assert profiles[0]["max_num_seqs"] == 512
    assert profiles[0]["max_batched_tokens"] == 8192
    assert profiles[0]["chunked_prefill_threshold"] == 4096


def test_make_bracket_profiles_k2():
    profiles = BLISRunner._make_bracket_profiles(2)
    assert len(profiles) == 2
    assert profiles[0]["max_num_seqs"] == 512
    assert profiles[1]["max_num_seqs"] == 32


def test_make_phase1_config_multi_replica():
    profile = {"max_num_seqs": 256, "max_batched_tokens": 4096, "chunked_prefill_threshold": 1024}
    config = BLISRunner._make_phase1_config(4, 2, profile)
    assert config["tp"] == 4
    assert config["replicas"] == 2
    assert config["routing_policy"] == "least-loaded"
    assert config["admission_policy"] == "always-admit"
    assert config["scheduler"] == "fcfs"
    assert config["max_num_seqs"] == 256


def test_make_phase1_config_single_replica():
    profile = {"max_num_seqs": 512, "max_batched_tokens": 8192, "chunked_prefill_threshold": 4096}
    config = BLISRunner._make_phase1_config(8, 1, profile)
    assert config["tp"] == 8
    assert config["replicas"] == 1
    assert config["routing_policy"] is None
    assert config["admission_policy"] is None
