from dataclasses import dataclass, field
from typing import Callable
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PruneConfig:
    max_num_seqs: int = 128
    max_batched_tokens: int = 4096
    rate: float = 5.0
    slo_ttft_ms: float = 300.0
    threshold_multiplier: float = 1.5


@dataclass
class PruneResult:
    topology: dict
    ttft_ms: float | None
    status: str
    passes: bool


def prune_topologies(
    topologies: list,
    evaluate_fn: Callable[[dict, PruneConfig], tuple[float | None, str]],
    config: PruneConfig | None = None,
) -> tuple[list, list[PruneResult]]:
    if config is None:
        config = PruneConfig()

    threshold = config.slo_ttft_ms * config.threshold_multiplier
    results: list[PruneResult] = []
    passing: list = []

    for topo in topologies:
        topo_dict = _topology_to_dict(topo)
        ttft_ms, status = evaluate_fn(topo_dict, config)

        if status != "ok" or ttft_ms is None:
            passes = False
            logger.warning(
                "Pruning topology %s: status=%s", topo_dict, status
            )
        elif ttft_ms > threshold:
            passes = False
            logger.info(
                "Pruning topology %s: TTFT %.1fms > threshold %.1fms",
                topo_dict, ttft_ms, threshold,
            )
        else:
            passes = True

        results.append(PruneResult(
            topology=topo_dict,
            ttft_ms=ttft_ms,
            status=status,
            passes=passes,
        ))
        if passes:
            passing.append(topo)

    logger.info(
        "Pruning: %d/%d topologies passed (threshold=%.0fms)",
        len(passing), len(topologies), threshold,
    )
    return passing, results


def save_prune_results(results: list[PruneResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "topology": r.topology,
            "ttft_ms": r.ttft_ms,
            "status": r.status,
            "passes": r.passes,
        }
        for r in results
    ]
    output_path.write_text(json.dumps(data, indent=2))


def load_pruned_topologies(prune_file: Path) -> list[dict]:
    data = json.loads(prune_file.read_text())
    return [r["topology"] for r in data if r["passes"]]


def _topology_to_dict(topo) -> dict:
    if hasattr(topo, "pp") and hasattr(topo, "dp"):
        return {"tp": topo.tp, "pp": topo.pp, "dp": topo.dp}
    elif hasattr(topo, "pp"):
        return {"tp": topo.tp, "pp": topo.pp, "replicas": topo.replicas}
    else:
        return {"tp": topo.tp, "replicas": topo.replicas}
