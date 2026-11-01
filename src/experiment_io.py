"""Reproducible experiment artefact input/output helpers."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd


TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "nltk",
    "scikit-learn",
    "seqeval",
    "sklearn-crfsuite",
    "torch",
    "pytorch-crf",
    "matplotlib",
]


def package_versions():
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_record(seed: int, device: str):
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "requested_device": device,
        "packages": package_versions(),
    }
    try:
        import torch

        record["torch_device_available"] = "cuda" if torch.cuda.is_available() else "cpu"
        record["cuda_version"] = torch.version.cuda
        record["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        record["torch_device_available"] = None
        record["cuda_version"] = None
        record["gpu_name"] = None
    return record


def write_json(value, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def write_predictions(model_name, data, predictions, output_dir):
    if len(data) != len(predictions):
        raise ValueError("Prediction count does not match dataset size.")
    output = Path(output_dir) / "predictions" / f"{model_name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for item, predicted_tags in zip(data, predictions):
            if len(item["words"]) != len(predicted_tags):
                raise ValueError(f"Prediction length mismatch for review {item['id']}.")
            record = {
                "id": item["id"],
                "words": item["words"],
                "pos_tags": item["pos_tags"],
                "gold_tags": item["bio_tags"],
                "pred_tags": predicted_tags,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def read_predictions(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_metrics(rows, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output

