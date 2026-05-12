import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from experiments.workloads.generate import (
    generate_canonical_trace,
    convert_to_llmservingsim,
    convert_to_vidur,
    convert_to_llm_optimizer,
)


def test_canonical_trace_calls_blis():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "canonical.yaml"
        with patch("experiments.workloads.generate.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="yaml content")
            generate_canonical_trace(output, blis_binary="./blis")
            args = mock_run.call_args[0][0]
            assert "./blis" in args[0]
            assert "convert" in args
            assert "preset" in args


def test_convert_to_llmservingsim():
    requests = [
        {"arrival_time": 0.0, "input_tokens": 512, "output_tokens": 256},
        {"arrival_time": 0.1, "input_tokens": 128, "output_tokens": 64},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "trace.jsonl"
        convert_to_llmservingsim(requests, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "input_toks" in record
        assert "output_toks" in record
        assert "arrival_time" in record


def test_convert_to_vidur():
    requests = [
        {"arrival_time": 0.0, "input_tokens": 512, "output_tokens": 256},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "trace.csv"
        convert_to_vidur(requests, output)
        content = output.read_text()
        assert "request_id" in content
        assert "arrival_time" in content
        assert "prefill_tokens" in content
        assert "decode_tokens" in content


def test_convert_to_llm_optimizer():
    requests = [
        {"arrival_time": 0.0, "input_tokens": 512, "output_tokens": 256},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "dataset.json"
        convert_to_llm_optimizer(requests, output)
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert "input_len" in data[0]
