"""Configurable end-to-end HMM, CRF, and BiLSTM-CRF experiment runner."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classical_models import HMMSequenceTagger, predict_crf, train_crf
from data_validation import validate_dataframe, write_validation_outputs
from evaluation import sequence_metrics
from experiment_io import environment_record, write_json, write_metrics, write_predictions
from preprocessing import build_bio_dataset


TAG_VOCABULARY = [
    "O",
    "B-ASP_POS", "I-ASP_POS",
    "B-ASP_NEG", "I-ASP_NEG",
    "B-ASP_NEU", "I-ASP_NEU",
    "B-OPINION", "I-OPINION",
]


def load_config(path: str | Path, overrides: dict | None = None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return config


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def split_dataset(data, test_size: float, validation_size: float, seed: int):
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    train_all, test = train_test_split(data, test_size=test_size, random_state=seed)
    train, validation = train_test_split(
        train_all, test_size=validation_size, random_state=seed
    )
    return train, validation, test


def choose_device(requested: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("BiLSTM-CRF training requires PyTorch.") from exc
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_vocabularies(train_data):
    word_frequency = Counter(word.lower() for item in train_data for word in item["words"])
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_frequency.items():
        if count >= 2:
            word2idx[word] = len(word2idx)
    pos2idx = {"<PAD>": 0, "<UNK>": 1}
    for item in train_data:
        for pos in item["pos_tags"]:
            pos2idx.setdefault(pos, len(pos2idx))
    tag2idx = {tag: index for index, tag in enumerate(TAG_VOCABULARY)}
    return word2idx, pos2idx, tag2idx


def train_bilstm_crf(train_data, validation_data, config, output_dir):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from models import BiLSTM_CRF, SequenceTaggingDataset, pad_collate_fn

    device = choose_device(config["device"])
    word2idx, pos2idx, tag2idx = build_vocabularies(train_data)
    idx2tag = {index: tag for tag, index in tag2idx.items()}

    def loader(dataset, shuffle):
        return DataLoader(
            SequenceTaggingDataset(dataset, word2idx, pos2idx, tag2idx),
            batch_size=config["batch_size"],
            shuffle=shuffle,
            collate_fn=pad_collate_fn,
        )

    train_loader = loader(train_data, True)
    validation_loader = loader(validation_data, False)
    model = BiLSTM_CRF(
        len(word2idx), len(pos2idx), len(tag2idx), hidden_dim=config["hidden_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    def predict(dataset):
        model.eval()
        predictions = []
        with torch.no_grad():
            for words, pos, _, mask in loader(dataset, False):
                decoded = model.decode(words.to(device), pos.to(device), mask.to(device))
                predictions.extend([[idx2tag[index] for index in sequence] for sequence in decoded])
        return predictions

    gold_validation = [item["bio_tags"] for item in validation_data]
    best_f1, best_state, bad_epochs = -1.0, None, 0
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for words, pos, tags, mask in train_loader:
            words, pos, tags, mask = (
                words.to(device), pos.to(device), tags.to(device), mask.to(device)
            )
            dropout_mask = (
                torch.rand_like(words, dtype=torch.float) < config["word_dropout"]
            ) & mask
            words = words.masked_fill(dropout_mask, word2idx["<UNK>"])
            optimizer.zero_grad()
            loss = model.forward_loss(words, pos, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total_loss += loss.item() * words.size(0)
            total_tokens += int(mask.sum())

        validation_predictions = predict(validation_data)
        validation_f1 = sequence_metrics(
            gold_validation, validation_predictions, target="combined"
        )["f1"]
        scheduler.step(validation_f1)
        history.append(
            {
                "epoch": epoch,
                "loss_per_token": total_loss / total_tokens,
                "validation_f1": validation_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_f1 > best_f1:
            best_f1, bad_epochs = validation_f1, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("BiLSTM-CRF training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "bilstm_crf_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "word2idx": word2idx,
            "pos2idx": pos2idx,
            "tag2idx": tag2idx,
            "config": config,
            "best_validation_f1": best_f1,
        },
        checkpoint_path,
    )
    write_metrics(history, Path(output_dir) / "metrics" / "training_history.csv")
    return model, predict, checkpoint_path


def extract_triplets(data, predictions):
    rows = []
    for item, tags in zip(data, predictions):
        aspects, opinions = [], []
        index = 0
        while index < len(tags):
            tag = tags[index]
            if tag.startswith("B-ASP_"):
                start = index
                polarity = tag.rsplit("_", 1)[-1]
                tokens = [item["words"][index]]
                index += 1
                while index < len(tags) and tags[index].startswith("I-ASP_"):
                    tokens.append(item["words"][index])
                    index += 1
                aspects.append((" ".join(tokens), polarity, start))
            elif tag == "B-OPINION":
                start = index
                tokens = [item["words"][index]]
                index += 1
                while index < len(tags) and tags[index] == "I-OPINION":
                    tokens.append(item["words"][index])
                    index += 1
                opinions.append((" ".join(tokens), start))
            else:
                index += 1
        for aspect, polarity, position in aspects:
            opinion = (
                min(opinions, key=lambda candidate: abs(candidate[1] - position))[0]
                if opinions
                else "implicit_opinion"
            )
            rows.append(
                {
                    "review_id": item["id"],
                    "aspect": aspect,
                    "opinion": opinion,
                    "polarity": polarity,
                    "review": " ".join(item["words"]),
                }
            )
    return rows


def run_experiment(data_path, output_dir, config):
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    set_random_seeds(config["seed"])
    write_json(config, output_dir / "effective_config.json")
    write_json(
        environment_record(config["seed"], config["device"]),
        metrics_dir / "environment.json",
    )

    dataframe = pd.read_csv(data_path)
    validation_report, validation_issues = validate_dataframe(dataframe)
    write_validation_outputs(validation_report, validation_issues, metrics_dir)
    if config["sample_size"] and config["sample_size"] < len(dataframe):
        dataframe = dataframe.sample(n=config["sample_size"], random_state=config["seed"])
    dataset, preprocessing_issues = build_bio_dataset(
        dataframe,
        extract_opinions=config["extract_opinions"],
        return_issues=True,
    )
    preprocessing_issues.to_csv(metrics_dir / "preprocessing_issues.csv", index=False)
    train_data, validation_data, test_data = split_dataset(
        dataset, config["test_size"], config["validation_size"], config["seed"]
    )
    write_json(
        {
            "train_ids": [item["id"] for item in train_data],
            "validation_ids": [item["id"] for item in validation_data],
            "test_ids": [item["id"] for item in test_data],
        },
        output_dir / "split_ids.json",
    )

    predictions = {}
    if "hmm" in config["models"]:
        hmm = HMMSequenceTagger(
            config["hmm_unk_threshold"], config["hmm_smoothing"], use_pos=True
        ).fit(train_data)
        predictions["hmm"] = hmm.predict(test_data)
    if "crf" in config["models"]:
        crf = train_crf(
            train_data,
            c1=config["crf_c1"],
            c2=config["crf_c2"],
            max_iterations=config["crf_max_iterations"],
        )
        predictions["crf"] = predict_crf(crf, test_data)
    if "bilstm_crf" in config["models"]:
        _, neural_predict, _ = train_bilstm_crf(
            train_data, validation_data, config, output_dir
        )
        predictions["bilstm_crf"] = neural_predict(test_data)

    gold = [item["bio_tags"] for item in test_data]
    metric_rows = []
    for model_name, model_predictions in predictions.items():
        write_predictions(model_name, test_data, model_predictions, output_dir)
        for target in ("combined", "aspect", "opinion"):
            metric_rows.append(
                {
                    "model": model_name,
                    "target": target,
                    **sequence_metrics(gold, model_predictions, target=target),
                }
            )
        triplet_path = output_dir / "predictions" / f"{model_name}_triplets.csv"
        pd.DataFrame(extract_triplets(test_data, model_predictions)).to_csv(
            triplet_path, index=False
        )
    write_metrics(metric_rows, metrics_dir / "model_metrics.csv")
    return metric_rows


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "REVIEWS_semeval.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "default.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--models", nargs="+", choices=["hmm", "crf", "bilstm_crf"])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "models": args.models,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    config = load_config(args.config, overrides)
    rows = run_experiment(args.data, args.output_dir, config)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

