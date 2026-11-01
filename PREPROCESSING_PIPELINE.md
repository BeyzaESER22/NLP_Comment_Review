# Data Preprocessing Pipeline

## Purpose

This document explains how `data/REVIEWS_semeval.csv` is validated and converted into token sequences for HMM, CRF, BiLSTM-CRF, and BERT. The authoritative implementation is `src/preprocessing.py`; `src/NLP2_Aspect_Opinion.py` imports the same function.

## Input Schema

The CSV contains 4,707 rows and the columns `text`, `aspect_terms`, `term_polarities`, `aspect_categories`, `category_polarities`, and `source`. The first three are required. Terms and polarities are pipe-delimited and positionally aligned. Supported polarities are `positive`, `negative`, `neutral`, and `conflict`; `conflict` maps to `NEU`.

## Stage 1 — Validation

`src/data_validation.py` checks required columns, missing text, incomplete or unequal annotation fields, blank or unsupported values, unaligned terms, and repeated literal occurrences.

```bash
python src/data_validation.py data/REVIEWS_semeval.csv --output-dir outputs/metrics
```

The current audit reports 4,707 rows, 7,681 usable pairs, 317 repeated-occurrence warnings, and no other issue types.

## Stage 2 — Selection and Splitting

`sample_size: 0` disables sampling. Seed `42` is used. The configurable runner reserves 20% for testing and 10% of the remaining portion for validation: 3,388 training, 377 validation, and 942 test records. The all-in-one output prints the outer 3,765/942 split before creating the neural validation subset.

## Stage 3 — Tokenisation and POS Tagging

The regular expression `\S+` creates non-whitespace tokens and retains offsets. Punctuation remains attached. NLTK `pos_tag` supplies features and supports weak opinion labelling.

## Stage 4 — Parsing

Aspect and polarity cells are split on `|`. Blank and unsupported items are reported and skipped instead of being silently converted to neutral.

| Source polarity | BIO suffix |
|---|---|
| `positive` | `POS` |
| `negative` | `NEG` |
| `neutral` | `NEU` |
| `conflict` | `NEU` |

## Stage 5 — Occurrence-Aware Alignment

All case-insensitive literal occurrences are located. Repeated annotations consume successive unoccupied occurrences. Overlapping annotations cannot overwrite earlier spans. If several occurrences are possible, the first unoccupied one is selected and a `repeated_aspect_ambiguous` warning is recorded.

## Stage 6 — BIO Conversion

Tokens overlapping an aspect receive `B-ASP_<POLARITY>` or `I-ASP_<POLARITY>`; other tokens initially receive `O`.

```text
Tokens: battery       life,         is  excellent.
Tags:   B-ASP_POS     I-ASP_POS     O   B-OPINION
```

## Stage 7 — Weak Opinion Labels

Tokens outside aspect spans are marked as opinions when their POS tag begins with `JJ` or `RB` and the cleaned token has more than two characters. These are heuristic labels, not human gold annotations.

## Stage 8 — Model-Ready Records

Each record contains `id`, `words`, `pos_tags`, and `bio_tags`. Classical models consume the lists directly. Neural datasets map them to tensors and mask padding.

```bash
python src/preprocessing.py \
  data/REVIEWS_semeval.csv \
  outputs/metrics/preprocessed_reviews.jsonl \
  --sample-size 0 --seed 42
```

Use `--no-opinions` to disable weak opinion labels.

## Testing

```bash
python -m unittest discover -s tests -v
```

Tests cover invalid polarities, multi-token and repeated aspects, punctuation, overlaps, deterministic splits, prediction I/O, separate metrics, and error categorisation.

## Limitations

- Whitespace tokenisation is deliberately simple.
- Original aspect offsets are unavailable.
- `conflict` is collapsed into `NEU`.
- Opinion spans are weakly supervised.
- The reported run uses a random split rather than official SemEval boundaries.
