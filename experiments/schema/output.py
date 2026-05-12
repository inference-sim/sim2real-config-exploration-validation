import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class WorkloadInfo(BaseModel):
    model: str
    hardware: str
    preset: str
    num_requests: int
    isl_mean: int
    isl_max: int
    osl_mean: int
    osl_max: int
    arrival_pattern: str
    slo_ttft_mean_ms: int
    seed: int
    trace_file: Optional[str] = None


class VllmArgs(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    num_instances: int
    data_parallel_size: int = 1
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_chunked_prefill: bool
    block_size: int
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    swap_space: int = 4


class RoutingConfig(BaseModel):
    strategy: str
    scorers: Optional[str] = None
    picker: Optional[str] = None


class ToolConfig(BaseModel):
    scheduler: Optional[str] = None
    admission_policy: Optional[str] = None
    preemption_policy: Optional[str] = None
    max_concurrency: Optional[int] = None
    vidur_scheduler_type: Optional[str] = None


class Results(BaseModel):
    max_throughput_tok_s: float
    max_throughput_qps: float
    ttft_mean_ms: float
    ttft_p50_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_mean_ms: Optional[float] = None
    meets_slo: bool
    cost_per_hour: float
    cost_per_1k_tokens: Optional[float] = None


class Metadata(BaseModel):
    status: str = "ok"
    tool_version: Optional[str] = None
    wall_clock_seconds: Optional[float] = None
    num_rate_probes: Optional[int] = None
    config_hash: Optional[str] = None
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class ConfigResult(BaseModel):
    tool: str
    workload: WorkloadInfo
    vllm_args: VllmArgs
    routing_config: Optional[RoutingConfig] = None
    tool_config: Optional[ToolConfig] = None
    results: Optional[Results] = None
    metadata: Metadata = Field(default_factory=Metadata)


def compute_config_hash(
    vllm_args: VllmArgs,
    routing_config: Optional[RoutingConfig] = None,
    tool_config: Optional[ToolConfig] = None,
) -> str:
    blob = json.dumps(
        {
            "vllm_args": vllm_args.model_dump(exclude_none=True),
            "routing_config": routing_config.model_dump(exclude_none=True) if routing_config else None,
            "tool_config": tool_config.model_dump(exclude_none=True) if tool_config else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:8]
