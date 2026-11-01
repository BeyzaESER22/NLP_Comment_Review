"""Dataset integrity checks for the aspect–opinion project."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"text", "aspect_terms", "term_polarities"}
SUPPORTED_POLARITIES = {"positive", "negative", "neutral", "conflict"}


def split_pipe(value: Any) -> list[str]:
    """Split a pipe-delimited cell while preserving blank items."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|")]


def find_all_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    """Return all case-insensitive, possibly repeated, literal term offsets."""
    if not term:
        return []
    text_lower, term_lower = text.lower(), term.lower()
    offsets: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text_lower.find(term_lower, start)
        if index < 0:
            break
        offsets.append((index, index + len(term)))
        start = index + max(1, len(term))
    return offsets


def validate_dataframe(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return an aggregate report and one row per detected integrity issue."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    issues: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    usable_pairs = 0

    for row_index, row in df.iterrows():
        text = "" if pd.isna(row["text"]) else str(row["text"])
        aspects = split_pipe(row["aspect_terms"])
        polarities = split_pipe(row["term_polarities"])

        if not text.strip():
            counters["missing_text_rows"] += 1
            issues.append({"row_index": row_index, "issue": "missing_text", "value": ""})
            continue

        if not aspects and not polarities:
            counters["unannotated_rows"] += 1
            continue
        if aspects and not polarities:
            counters["aspect_without_polarity_rows"] += 1
            issues.append({"row_index": row_index, "issue": "aspect_without_polarity", "value": row["aspect_terms"]})
            continue
        if polarities and not aspects:
            counters["polarity_without_aspect_rows"] += 1
            issues.append({"row_index": row_index, "issue": "polarity_without_aspect", "value": row["term_polarities"]})
            continue
        if len(aspects) != len(polarities):
            counters["length_mismatch_rows"] += 1
            issues.append({
                "row_index": row_index,
                "issue": "aspect_polarity_length_mismatch",
                "value": f"{len(aspects)} aspects / {len(polarities)} polarities",
            })

        for pair_index in range(max(len(aspects), len(polarities))):
            aspect = aspects[pair_index] if pair_index < len(aspects) else ""
            polarity = polarities[pair_index].lower() if pair_index < len(polarities) else ""
            if not aspect:
                counters["blank_aspect_items"] += 1
                issues.append({"row_index": row_index, "issue": "blank_aspect", "value": pair_index})
                continue
            if not polarity:
                counters["blank_polarity_items"] += 1
                issues.append({"row_index": row_index, "issue": "blank_polarity", "value": aspect})
                continue
            if polarity not in SUPPORTED_POLARITIES:
                counters["unsupported_polarity_items"] += 1
                issues.append({"row_index": row_index, "issue": "unsupported_polarity", "value": polarity})
                continue

            occurrences = find_all_occurrences(text, aspect)
            if not occurrences:
                counters["unaligned_aspect_items"] += 1
                issues.append({"row_index": row_index, "issue": "aspect_not_found", "value": aspect})
                continue
            if len(occurrences) > 1:
                counters["ambiguous_repeated_aspect_items"] += 1
                issues.append({
                    "row_index": row_index,
                    "issue": "repeated_aspect_ambiguous",
                    "value": f"{aspect} ({len(occurrences)} occurrences)",
                })
            usable_pairs += 1

    report: dict[str, Any] = {
        "total_rows": int(len(df)),
        "usable_aspect_polarity_pairs": int(usable_pairs),
        **{key: int(value) for key, value in sorted(counters.items())},
        "total_issues": int(len(issues)),
    }
    return report, pd.DataFrame(issues, columns=["row_index", "issue", "value"])


def write_validation_outputs(
    report: dict[str, Any], issues: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "data_validation_report.json"
    issues_path = output_path / "data_validation_issues.csv"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    issues.to_csv(issues_path, index=False)
    return report_path, issues_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate REVIEWS_semeval.csv annotations.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    dataframe = pd.read_csv(args.csv_path)
    report, issues = validate_dataframe(dataframe)
    report_path, issues_path = write_validation_outputs(report, issues, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    print(f"Issues: {issues_path}")


if __name__ == "__main__":
    main()

