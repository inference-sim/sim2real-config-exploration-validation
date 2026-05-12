import json
from experiments.schema.output import (
    ConfigResult, WorkloadInfo, VllmArgs, RoutingConfig,
    ToolConfig, Results, Metadata, compute_config_hash,
)


def _make_workload(**overrides):
    defaults = dict(
        model="meta-llama/Llama-3.1-8B", hardware="H100_SXM_80GB",
        preset="servegen_m-mid", num_requests=10000,
        isl_mean=512, isl_max=2048, osl_mean=256, osl_max=1024,
        arrival_pattern="poisson", slo_ttft_mean_ms=300, seed=42,
    )
    defaults.update(overrides)
    return WorkloadInfo(**defaults)


def _make_vllm_args(**overrides):
    defaults = dict(
        tensor_parallel_size=2, pipeline_parallel_size=1,
        num_instances=4, data_parallel_size=1,
        max_num_seqs=128, max_num_batched_tokens=4096,
        enable_chunked_prefill=True, block_size=16,
    )
    defaults.update(overrides)
    return VllmArgs(**defaults)


def test_full_config_result_roundtrip():
    result = ConfigResult(
        tool="inference-sim",
        workload=_make_workload(),
        vllm_args=_make_vllm_args(),
        routing_config=RoutingConfig(
            strategy="weighted-scoring",
            scorers="prefix-cache:2,queue-depth:1,kv-utilization:1",
            picker="max-score",
        ),
        tool_config=ToolConfig(
            scheduler="priority-fcfs",
            admission_policy="tier-shed",
            preemption_policy="priority",
        ),
        results=Results(
            max_throughput_tok_s=850.0, max_throughput_qps=12.5,
            ttft_mean_ms=245.0, ttft_p50_ms=220.0, ttft_p99_ms=380.0,
            tpot_mean_ms=18.5, meets_slo=True,
            cost_per_hour=25.60, cost_per_1k_tokens=0.0084,
        ),
        metadata=Metadata(
            status="ok", tool_version="v0.4.2",
            wall_clock_seconds=2.3, num_rate_probes=8, config_hash="a3f8c1d2",
        ),
    )
    data = json.loads(result.model_dump_json())
    assert data["tool"] == "inference-sim"
    assert data["vllm_args"]["tensor_parallel_size"] == 2
    assert data["results"]["meets_slo"] is True
    roundtrip = ConfigResult.model_validate(data)
    assert roundtrip.tool == result.tool


def test_failed_config_has_null_results():
    result = ConfigResult(
        tool="llm-optimizer",
        workload=_make_workload(),
        vllm_args=_make_vllm_args(tensor_parallel_size=8, num_instances=1),
        results=None,
        metadata=Metadata(status="oom", wall_clock_seconds=5.2),
    )
    data = json.loads(result.model_dump_json())
    assert data["results"] is None
    assert data["metadata"]["status"] == "oom"


def test_single_replica_has_null_routing():
    result = ConfigResult(
        tool="vidur",
        workload=_make_workload(),
        vllm_args=_make_vllm_args(tensor_parallel_size=4, num_instances=1),
        routing_config=None,
        tool_config=ToolConfig(vidur_scheduler_type="vllm"),
        results=Results(
            max_throughput_tok_s=600.0, max_throughput_qps=9.0,
            ttft_mean_ms=180.0, meets_slo=True,
            cost_per_hour=12.80, cost_per_1k_tokens=0.006,
        ),
        metadata=Metadata(status="ok", wall_clock_seconds=45.0, config_hash="b1c2d3e4"),
    )
    assert result.routing_config is None
    data = json.loads(result.model_dump_json())
    assert data["routing_config"] is None


def test_config_hash_deterministic():
    vllm = _make_vllm_args()
    routing = RoutingConfig(strategy="weighted-scoring", scorers="prefix-cache:2,queue-depth:1", picker="max-score")
    tool = ToolConfig(scheduler="fcfs")
    h1 = compute_config_hash(vllm, routing, tool)
    h2 = compute_config_hash(vllm, routing, tool)
    assert h1 == h2
    assert len(h1) == 8
