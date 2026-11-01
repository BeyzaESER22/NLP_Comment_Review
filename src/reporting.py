"""Advanced evaluation, error exports, and figures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation import sequence_metrics
from experiment_io import read_predictions


NEGATION_WORDS = {"not", "n't", "no", "never", "nothing", "hardly", "without"}
POLARITY_TYPES = ["ASP_POS", "ASP_NEG", "ASP_NEU"]


def extract_entities(tags):
    entities = []
    current_type, start = None, None
    for index, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-"):
            if current_type is not None:
                entities.append((current_type, start, index - 1))
            current_type, start = tag[2:], index
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue
        else:
            if current_type is not None:
                entities.append((current_type, start, index - 1))
                current_type, start = None, None
            if tag.startswith("I-"):
                current_type, start = tag[2:], index
    return entities


def categorize_record_errors(record, model_name):
    gold = {(start, end): entity_type for entity_type, start, end in extract_entities(record["gold_tags"])}
    predicted = {
        (start, end): entity_type
        for entity_type, start, end in extract_entities(record["pred_tags"])
    }
    rows = []
    text = " ".join(record["words"])
    for span, gold_type in gold.items():
        if span in predicted:
            predicted_type = predicted[span]
            if predicted_type != gold_type:
                error_type = (
                    "polarity"
                    if gold_type.startswith("ASP_") and predicted_type.startswith("ASP_")
                    else "type"
                )
                rows.append(_error_row(record, model_name, span, error_type, gold_type, predicted_type, text))
        elif any(_overlaps(span, predicted_span) for predicted_span in predicted):
            rows.append(_error_row(record, model_name, span, "boundary", gold_type, "overlap", text))
        else:
            rows.append(_error_row(record, model_name, span, "missed", gold_type, "-", text))
    for span, predicted_type in predicted.items():
        if span not in gold and not any(_overlaps(span, gold_span) for gold_span in gold):
            rows.append(_error_row(record, model_name, span, "spurious", "-", predicted_type, text))
    return rows


def _overlaps(first, second):
    return first[0] <= second[1] and second[0] <= first[1]


def _error_row(record, model, span, error_type, gold, predicted, text):
    return {
        "model": model,
        "review_id": record["id"],
        "start_token": span[0],
        "end_token": span[1],
        "error_type": error_type,
        "gold": gold,
        "predicted": predicted,
        "review": text,
    }


def polarity_confusion_rows(records, model_name):
    counts = Counter()
    for record in records:
        gold = {
            (start, end): entity_type
            for entity_type, start, end in extract_entities(record["gold_tags"])
            if entity_type in POLARITY_TYPES
        }
        predicted = {
            (start, end): entity_type
            for entity_type, start, end in extract_entities(record["pred_tags"])
            if entity_type in POLARITY_TYPES
        }
        for span in gold.keys() & predicted.keys():
            counts[(gold[span], predicted[span])] += 1
    return [
        {"model": model_name, "gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in sorted(counts.items())
    ]


def pattern_flags(record):
    words = [word.lower() for word in record["words"]]
    return {
        "negation": any(word in NEGATION_WORDS or word.endswith("n't") for word in words),
        "contrast": "but" in words or "however" in words,
        "multi_aspect": sum(tag.startswith("B-ASP_") for tag in record["gold_tags"]) >= 2,
    }


def hard_pattern_rows(records, model_name):
    rows = []
    for pattern in ("negation", "contrast", "multi_aspect"):
        selected = [record for record in records if pattern_flags(record)[pattern]]
        others = [record for record in records if not pattern_flags(record)[pattern]]
        for group_name, group in (("pattern", selected), ("other", others)):
            wrong = sum(record["gold_tags"] != record["pred_tags"] for record in group)
            rows.append(
                {
                    "model": model_name,
                    "pattern": pattern,
                    "group": group_name,
                    "sentence_count": len(group),
                    "error_rate": wrong / len(group) if group else None,
                }
            )
    return rows


def metric_rows(records, model_name):
    gold = [record["gold_tags"] for record in records]
    predicted = [record["pred_tags"] for record in records]
    rows = []
    for target in ("combined", "aspect", "opinion"):
        rows.append({"model": model_name, "target": target, **sequence_metrics(gold, predicted, target)})
    for polarity_type in POLARITY_TYPES:
        gold_projected = [
            [tag if tag.endswith(polarity_type) else "O" for tag in sequence] for sequence in gold
        ]
        predicted_projected = [
            [tag if tag.endswith(polarity_type) else "O" for tag in sequence]
            for sequence in predicted
        ]
        rows.append(
            {
                "model": model_name,
                "target": polarity_type.lower(),
                **sequence_metrics(gold_projected, predicted_projected, "combined"),
            }
        )
    return rows


def create_figures(metrics, errors, confusion, figures_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Figure generation requires matplotlib.") from exc
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined = metrics[metrics["target"] == "combined"]
    if not combined.empty:
        chart = combined.set_index("model")[["precision", "recall", "f1"]].plot(
            kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0
        )
        chart.set_title("Model Performance Comparison")
        chart.set_ylabel("Score")
        chart.figure.tight_layout()
        chart.figure.savefig(figures_dir / "model_metrics.png", dpi=180)
        plt.close(chart.figure)

    if not errors.empty:
        counts = errors.groupby(["model", "error_type"]).size().unstack(fill_value=0)
        chart = counts.plot(kind="bar", figsize=(11, 6), rot=0)
        chart.set_title("Error Categories by Model")
        chart.set_ylabel("Entity count")
        chart.figure.tight_layout()
        chart.figure.savefig(figures_dir / "error_categories.png", dpi=180)
        plt.close(chart.figure)

    if not confusion.empty:
        for model_name, model_rows in confusion.groupby("model"):
            matrix = model_rows.pivot_table(
                index="gold", columns="predicted", values="count", fill_value=0
            )
            figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(matrix.values, cmap="Blues")
            axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
            axis.set_yticks(range(len(matrix.index)), matrix.index)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Gold")
            axis.set_title(f"Polarity Confusion — {model_name}")
            for row in range(len(matrix.index)):
                for column in range(len(matrix.columns)):
                    axis.text(column, row, int(matrix.iloc[row, column]), ha="center", va="center")
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            figure.savefig(figures_dir / f"polarity_confusion_{model_name}.png", dpi=180)
            plt.close(figure)


def generate_reports(predictions_dir, output_dir, figures_dir):
    predictions_dir, output_dir = Path(predictions_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, all_errors, all_confusion, all_patterns = [], [], [], []
    prediction_paths = sorted(
        path for path in predictions_dir.glob("*.jsonl") if not path.name.endswith("_triplets.jsonl")
    )
    if not prediction_paths:
        raise ValueError(f"No prediction JSONL files found in {predictions_dir}.")
    for path in prediction_paths:
        model_name = path.stem
        records = read_predictions(path)
        all_metrics.extend(metric_rows(records, model_name))
        for record in records:
            all_errors.extend(categorize_record_errors(record, model_name))
        all_confusion.extend(polarity_confusion_rows(records, model_name))
        all_patterns.extend(hard_pattern_rows(records, model_name))

    metrics = pd.DataFrame(all_metrics)
    errors = pd.DataFrame(all_errors)
    confusion = pd.DataFrame(all_confusion)
    patterns = pd.DataFrame(all_patterns)
    metrics.to_csv(output_dir / "detailed_model_metrics.csv", index=False)
    errors.to_csv(output_dir / "error_examples.csv", index=False)
    if not errors.empty:
        errors.groupby(["model", "error_type"]).size().reset_index(name="count").to_csv(
            output_dir / "error_summary_by_model.csv", index=False
        )
    confusion.to_csv(output_dir / "polarity_confusion_matrix.csv", index=False)
    patterns.to_csv(output_dir / "hard_pattern_error_rates.csv", index=False)

    create_figures(metrics, errors, confusion, figures_dir)
    return metrics, errors, confusion, patterns


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=project_root / "outputs" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "metrics")
    parser.add_argument("--figures-dir", type=Path, default=project_root / "outputs" / "figures")
    args = parser.parse_args()
    generate_reports(args.predictions_dir, args.output_dir, args.figures_dir)
    print(f"Reports written to {args.output_dir}")
    print(f"Figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()

