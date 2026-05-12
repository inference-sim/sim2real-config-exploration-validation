import json
import tempfile
from pathlib import Path

from experiments.analysis.chart1 import plot_chart1, load_tool_results, compute_pareto_indices


def _write_mock_results(results_dir: Path, tool_name: str, configs: list[dict]):
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / f"{tool_name}.jsonl"
    with open(jsonl_path, "w") as f:
        for cfg in configs:
            f.write(json.dumps(cfg) + "\n")


def _make_result(tool, ttft, throughput, cost, meets_slo=True, config_hash="abc123"):
    return {
        "tool": tool,
        "workload": {"model": "test"},
        "vllm_args": {"tensor_parallel_size": 1},
        "results": {
            "ttft_mean_ms": ttft,
            "max_throughput_tok_s": throughput,
            "cost_per_hour": cost,
            "meets_slo": meets_slo,
            "max_throughput_qps": throughput / 256,
        },
        "metadata": {"status": "ok", "config_hash": config_hash},
    }


def test_plot_chart1_generates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "raw"
        _write_mock_results(results_dir, "blis", [
            _make_result("inference-sim", 200, 500, 6.40, config_hash="h1"),
            _make_result("inference-sim", 250, 800, 12.80, config_hash="h2"),
            _make_result("inference-sim", 280, 1000, 19.20, config_hash="h3"),
        ])
        _write_mock_results(results_dir, "vidur", [
            _make_result("Vidur", 180, 400, 6.40, config_hash="v1"),
            _make_result("Vidur", 220, 700, 12.80, config_hash="v2"),
        ])

        output = Path(tmpdir) / "figures" / "chart1"
        png = plot_chart1(results_dir, output)
        assert png.exists()
        assert output.with_suffix(".pdf").exists()


def test_plot_chart1_with_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "raw"
        _write_mock_results(results_dir, "blis", [
            _make_result("inference-sim", 200, 500, 6.40, config_hash="h1"),
        ])

        val_dir = Path(tmpdir) / "validated"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_file = val_dir / "validation.jsonl"
        val_file.write_text(json.dumps({
            "config_hash": "h1",
            "actual_ttft_mean_ms": 350.0,
            "actual_throughput_tok_s": 420.0,
        }) + "\n")

        output = Path(tmpdir) / "figures" / "chart1"
        png = plot_chart1(results_dir, output, validation_file=val_file)
        assert png.exists()


def test_plot_chart1_empty_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "raw"
        results_dir.mkdir(parents=True, exist_ok=True)
        output = Path(tmpdir) / "figures" / "chart1"
        png = plot_chart1(results_dir, output)
        assert png.exists()


def test_compute_pareto_indices():
    costs = [10, 10, 20, 20, 30]
    throughputs = [500, 800, 1000, 600, 900]
    indices = compute_pareto_indices(costs, throughputs)
    assert 1 in indices  # (10, 800)
    assert 2 in indices  # (20, 1000)
    assert 0 not in indices  # dominated by (10, 800)


def test_compute_pareto_indices_empty():
    assert compute_pareto_indices([], []) == []
