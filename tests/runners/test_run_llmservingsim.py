import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.runners.run_llmservingsim import LLMServingSimRunner
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


def test_build_cluster_config_single_instance():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {"tp": 4, "pp": 1, "replicas": 1}
        cluster = runner._build_cluster_config(config)

        assert cluster["num_nodes"] == 1
        assert cluster["link_bw"] == 900
        node = cluster["nodes"][0]
        assert node["num_instances"] == 1
        assert len(node["instances"]) == 1

        inst = node["instances"][0]
        assert inst["tp_size"] == 4
        assert inst["hardware"] == "H100"
        assert inst["npu_mem"]["mem_size"] == 80
        assert inst["npu_mem"]["mem_bw"] == 3350


def test_build_cluster_config_multi_instance():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {"tp": 2, "pp": 1, "replicas": 4}
        cluster = runner._build_cluster_config(config)

        node = cluster["nodes"][0]
        assert node["num_instances"] == 4
        assert len(node["instances"]) == 4
        for inst in node["instances"]:
            assert inst["tp_size"] == 2


def test_generate_workload_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
            num_requests=100,
        )
        wl_path = Path(tmpdir) / "workload.jsonl"
        runner._generate_workload_jsonl(wl_path, rate=10.0, num_requests=100)

        lines = wl_path.read_text().strip().split("\n")
        assert len(lines) == 100

        first = json.loads(lines[0])
        assert first["input_toks"] == 512
        assert first["output_toks"] == 256
        assert first["arrival_time_ns"] > 0

        # Verify monotonically increasing arrival times
        prev_t = 0
        for line in lines:
            rec = json.loads(line)
            assert rec["arrival_time_ns"] >= prev_t
            prev_t = rec["arrival_time_ns"]


def test_generate_workload_jsonl_deterministic():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(seed=42),
            output_path=Path(tmpdir) / "out.jsonl",
            num_requests=50,
        )
        p1 = Path(tmpdir) / "wl1.jsonl"
        p2 = Path(tmpdir) / "wl2.jsonl"
        runner._generate_workload_jsonl(p1, rate=5.0, num_requests=50)
        runner._generate_workload_jsonl(p2, rate=5.0, num_requests=50)
        assert p1.read_text() == p2.read_text()


def test_parse_output_csv():
    csv_content = (
        "instance id,request id,model,input,output,arrival,end_time,"
        "latency,queuing_delay,TTFT,TPOT,ITL\n"
        "0,req-1,model,512,256,0,500000000,500000000,10000000,"
        "100000000,2000000,[2000000]\n"
        "0,req-2,model,512,256,100000000,700000000,600000000,20000000,"
        "150000000,1800000,[1800000]\n"
        "0,req-3,model,512,256,200000000,900000000,700000000,30000000,"
        "200000000,2200000,[2200000]\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()

        result = LLMServingSimRunner._parse_output_csv(Path(f.name))
        assert result is not None
        # Mean TTFT: (100 + 150 + 200) / 3 = 150ms
        assert abs(result["ttft_mean_ms"] - 150.0) < 1.0
        # Mean TPOT: (2 + 1.8 + 2.2) / 3 = 2.0ms
        assert abs(result["tpot_mean_ms"] - 2.0) < 0.1


def test_parse_output_csv_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("instance id,request id,model,input,output,arrival,"
                "end_time,latency,queuing_delay,TTFT,TPOT,ITL\n")
        f.flush()
        result = LLMServingSimRunner._parse_output_csv(Path(f.name))
        assert result is None


def test_enumerate_configs_lean():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
            lean=True,
        )
        configs = runner._enumerate_configs()
        assert len(configs) > 5
        assert len(configs) < 50

        for c in configs:
            assert c["tp"] in (1, 2, 4)
            assert c["tp"] * c.get("pp", 1) * c["replicas"] <= 8


def test_enumerate_configs_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
            lean=False,
        )
        configs = runner._enumerate_configs()
        assert len(configs) > 100

        block_sizes = {c["block_size"] for c in configs}
        assert 32 in block_sizes


def test_config_to_vllm_args():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config = {
            "tp": 2, "pp": 1, "replicas": 4,
            "max_num_seqs": 256, "max_batched_tokens": 8192,
            "long_prefill_token_threshold": 1024,
            "block_size": 32, "prefix_caching": True,
        }
        vllm_args = runner._config_to_vllm_args(config)
        assert vllm_args.tensor_parallel_size == 2
        assert vllm_args.num_replicas == 4
        assert vllm_args.enable_chunked_prefill is True
        assert vllm_args.enable_prefix_caching is True
        assert vllm_args.block_size == 32


def test_config_to_routing():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LLMServingSimRunner(
            workload=_make_workload(),
            output_path=Path(tmpdir) / "out.jsonl",
        )
        config_multi = {"tp": 2, "pp": 1, "replicas": 4, "routing_policy": "LOAD"}
        routing = runner._config_to_routing(config_multi)
        assert routing is not None
        assert routing.strategy == "least-loaded"

        config_rr = {"tp": 1, "pp": 1, "replicas": 2, "routing_policy": "RR"}
        assert runner._config_to_routing(config_rr).strategy == "round-robin"

        config_single = {"tp": 4, "pp": 1, "replicas": 1}
        assert runner._config_to_routing(config_single) is None
