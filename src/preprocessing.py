"""Validated BIO conversion and occurrence-aware span alignment."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data_validation import find_all_occurrences, split_pipe


POLARITY_MAP = {
    "positive": "POS",
    "negative": "NEG",
    "neutral": "NEU",
    "conflict": "NEU",
}


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def align_aspect_annotations(
    text: str, aspects_raw: Any, polarities_raw: Any
) -> tuple[list[tuple[int, int, str, str]], list[dict[str, Any]]]:
    """Align valid aspect/polarity pairs without silently inventing labels.

    Repeated annotations of the same term consume successive text occurrences.
    When one annotation has several possible occurrences, the first unoccupied
    occurrence is selected and the ambiguity is reported.
    """
    aspects = split_pipe(aspects_raw)
    polarities = split_pipe(polarities_raw)
    aligned: list[tuple[int, int, str, str]] = []
    issues: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    next_occurrence: defaultdict[str, int] = defaultdict(int)

    if not aspects and not polarities:
        return aligned, issues
    if len(aspects) != len(polarities):
        issues.append({
            "issue": "aspect_polarity_length_mismatch",
            "value": f"{len(aspects)} aspects / {len(polarities)} polarities",
        })

    for pair_index in range(max(len(aspects), len(polarities))):
        aspect = aspects[pair_index] if pair_index < len(aspects) else ""
        polarity = polarities[pair_index].lower() if pair_index < len(polarities) else ""
        if not aspect:
            issues.append({"issue": "blank_aspect", "value": pair_index})
            continue
        if not polarity:
            issues.append({"issue": "blank_polarity", "value": aspect})
            continue
        if polarity not in POLARITY_MAP:
            issues.append({"issue": "unsupported_polarity", "value": polarity})
            continue

        occurrences = find_all_occurrences(text, aspect)
        if not occurrences:
            issues.append({"issue": "aspect_not_found", "value": aspect})
            continue
        if len(occurrences) > 1:
            issues.append({
                "issue": "repeated_aspect_ambiguous",
                "value": f"{aspect} ({len(occurrences)} occurrences)",
            })

        term_key = aspect.casefold()
        start_at = next_occurrence[term_key]
        candidates = occurrences[start_at:] + occurrences[:start_at]
        chosen = next((span for span in candidates if not _overlaps(span, occupied)), None)
        if chosen is None:
            issues.append({"issue": "overlapping_or_exhausted_aspect", "value": aspect})
            continue
        chosen_index = occurrences.index(chosen)
        next_occurrence[term_key] = chosen_index + 1
        occupied.append(chosen)
        aligned.append((chosen[0], chosen[1], aspect, POLARITY_MAP[polarity]))

    return aligned, issues


def build_bio_dataset(
    df: pd.DataFrame,
    extract_opinions: bool = True,
    return_issues: bool = False,
    pos_tagger: Callable[[list[str]], list[tuple[str, str]]] | None = None,
):
    """Convert reviews to sentiment-aware BIO sequences.

    Blank and unsupported polarity values are skipped and reported instead of
    being converted to neutral labels.
    """
    if "text" not in df.columns:
        raise ValueError("The dataframe must contain a 'text' column.")
    if pos_tagger is None:
        try:
            import nltk
        except ImportError as exc:
            raise ImportError(
                "NLTK is required for default POS tagging. Install requirements.txt "
                "or pass a custom pos_tagger."
            ) from exc
        tagger = nltk.pos_tag
    else:
        tagger = pos_tagger
    sentences_data: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []

    for row_index, row in df.dropna(subset=["text"]).iterrows():
        raw_text = str(row["text"])
        token_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", raw_text)]
        if not token_spans:
            continue

        words = [token for token, _, _ in token_spans]
        pos_tags = [pos for _, pos in tagger(words)]
        bio_tags = ["O"] * len(words)
        aligned, issues = align_aspect_annotations(
            raw_text, row.get("aspect_terms"), row.get("term_polarities")
        )
        all_issues.extend({"row_index": row_index, **issue} for issue in issues)

        for start_char, end_char, _, polarity in aligned:
            token_indices = [
                index
                for index, (_, token_start, token_end) in enumerate(token_spans)
                if token_start < end_char and token_end > start_char
            ]
            for offset, token_index in enumerate(token_indices):
                if bio_tags[token_index] != "O":
                    all_issues.append({
                        "row_index": row_index,
                        "issue": "token_overlap",
                        "value": words[token_index],
                    })
                    continue
                prefix = "B" if offset == 0 else "I"
                bio_tags[token_index] = f"{prefix}-ASP_{polarity}"

        if extract_opinions:
            for index, (word, pos) in enumerate(zip(words, pos_tags)):
                clean_word = re.sub(r"[^\w]", "", word)
                if bio_tags[index] == "O" and pos.startswith(("JJ", "RB")) and len(clean_word) > 2:
                    previous_is_opinion = index > 0 and bio_tags[index - 1] in {
                        "B-OPINION",
                        "I-OPINION",
                    }
                    bio_tags[index] = "I-OPINION" if previous_is_opinion else "B-OPINION"

        sentences_data.append({
            "id": row_index,
            "words": words,
            "pos_tags": pos_tags,
            "bio_tags": bio_tags,
        })

    if return_issues:
        return sentences_data, pd.DataFrame(all_issues, columns=["row_index", "issue", "value"])
    return sentences_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert REVIEWS_semeval.csv into model-ready BIO-labelled JSONL."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-opinions", action="store_true")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.csv_path).dropna(subset=["text"])
    if args.sample_size > 0 and args.sample_size < len(dataframe):
        dataframe = dataframe.sample(n=args.sample_size, random_state=args.seed)
    records, issues = build_bio_dataset(
        dataframe,
        extract_opinions=not args.no_opinions,
        return_issues=True,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    issues_path = args.output_path.with_suffix(".issues.csv")
    issues.to_csv(issues_path, index=False)
    print(f"Processed reviews: {len(records)}")
    print(f"BIO output: {args.output_path}")
    print(f"Preprocessing issues: {issues_path}")


if __name__ == "__main__":
    main()

