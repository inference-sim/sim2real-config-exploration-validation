import csv
import json
import subprocess
from pathlib import Path

CANONICAL_NUM_REQUESTS = 10000
CANONICAL_SEED = 42


def generate_canonical_trace(
    output_path: Path,
    blis_binary: str = "./estimators/inference-sim/blis",
    defaults_filepath: str = "./estimators/inference-sim/defaults.yaml",
    preset: str = "chatbot",
    num_requests: int = CANONICAL_NUM_REQUESTS,
    rate: float = 10.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            blis_binary, "convert", "preset",
            "--defaults-filepath", defaults_filepath,
            "--name", preset,
            "--num-requests", str(num_requests),
            "--rate", str(rate),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    output_path.write_text(result.stdout)


def parse_canonical_trace(trace_path: Path) -> list[dict]:
    import yaml
    with open(trace_path) as f:
        spec = yaml.safe_load(f)
    requests = []
    if "requests" in spec:
        for r in spec["requests"]:
            requests.append({
                "arrival_time": r.get("arrival_time", 0.0),
                "input_tokens": r.get("input_tokens", r.get("isl", 512)),
                "output_tokens": r.get("output_tokens", r.get("osl", 256)),
            })
    return requests


def convert_to_llmservingsim(requests: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in requests:
            record = {
                "input_toks": r["input_tokens"],
                "output_toks": r["output_tokens"],
                "arrival_time": r["arrival_time"],
            }
            f.write(json.dumps(record) + "\n")


def convert_to_vidur(requests: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "arrival_time", "prefill_tokens", "decode_tokens"])
        for i, r in enumerate(requests):
            writer.writerow([i, r["arrival_time"], r["input_tokens"], r["output_tokens"]])


def convert_to_llm_optimizer(requests: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = []
    for r in requests:
        dataset.append({
            "input_len": r["input_tokens"],
            "output_len": r["output_tokens"],
        })
    output_path.write_text(json.dumps(dataset, indent=2))


def generate_all_workloads(
    output_dir: Path,
    blis_binary: str = "./estimators/inference-sim/blis",
    defaults_filepath: str = "./estimators/inference-sim/defaults.yaml",
    preset: str = "chatbot",
    num_requests: int = CANONICAL_NUM_REQUESTS,
    rate: float = 10.0,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = output_dir / f"canonical_{preset}.yaml"
    generate_canonical_trace(
        canonical,
        blis_binary=blis_binary,
        defaults_filepath=defaults_filepath,
        preset=preset,
        num_requests=num_requests,
        rate=rate,
    )

    requests = parse_canonical_trace(canonical)

    paths = {"blis": canonical}

    llmservingsim_path = output_dir / "llmservingsim_trace.jsonl"
    convert_to_llmservingsim(requests, llmservingsim_path)
    paths["llmservingsim"] = llmservingsim_path

    vidur_path = output_dir / "vidur_trace.csv"
    convert_to_vidur(requests, vidur_path)
    paths["vidur"] = vidur_path

    llm_optimizer_path = output_dir / "llm_optimizer_dataset.json"
    convert_to_llm_optimizer(requests, llm_optimizer_path)
    paths["llm_optimizer"] = llm_optimizer_path

    return paths
