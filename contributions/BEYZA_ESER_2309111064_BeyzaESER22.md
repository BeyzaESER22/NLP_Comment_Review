# Individual Contribution — Beyza ESER

## Student Information

- **Student ID:** 2309111064
- **GitHub Username:** `BeyzaESER22`
- **Primary Role:** Data-quality investigation, validated preprocessing, modularisation, testing, and technical documentation

## Contribution Overview

I worked on the project after the original all-in-one implementation had been created by Meltem Çelik and Berk Ötün. I identified inconsistencies in the earlier dataset and then improved the reliability, modularity, testing, and documentation of the final project. Meltem subsequently corrected the dataset content that I had flagged.

## Completed Work

### 1. Dataset Inconsistency Investigation

- Examined missing, blank, unsupported, mismatched, and ambiguous aspect–polarity annotations in the earlier dataset.
- Documented why missing polarity values must not be treated as evidence of neutral sentiment.
- Identified repeated aspect strings for which the source CSV did not provide original character offsets.
- Communicated the dataset problems that were subsequently corrected by Meltem Çelik.

### 2. Reusable Validation Pipeline

Implemented `src/data_validation.py` to audit:

- required columns and review text;
- missing or unequal aspect/polarity fields;
- blank and unsupported values;
- terms that do not occur in the review; and
- terms with multiple possible occurrences.

The final corrected dataset audit reports:

- 4,707 reviews;
- 7,681 usable aspect–polarity pairs;
- 317 repeated-occurrence warnings; and
- no unsupported-polarity or unaligned-aspect issue types.

### 3. Validated and Occurrence-Aware Preprocessing

Implemented the modular `src/preprocessing.py` pipeline by refining the original conversion logic:

- valid pipe-field parsing;
- preserved token offsets;
- occurrence-aware repeated-term alignment;
- overlap protection;
- sentiment-aware BIO conversion; and
- auditable issue output.

This separates the final reusable implementation from Berk Ötün's original all-in-one `build_bio_dataset` contribution.

### 4. Neural-Component Modularisation

Moved and completed the reusable neural components in `src/models.py`:

- `SequenceTaggingDataset`;
- `pad_collate_fn`; and
- the modular `BiLSTM_CRF` class.

This work modularised the original neural experiment implemented by Meltem Çelik rather than claiming authorship of the original model concept.

### 5. Evaluation Safeguards and Tests

- Added `src/evaluation.py` for separate combined, aspect, and opinion exact-span metrics.
- Added automated preprocessing and evaluation tests.
- Verified the final package with all 15 project tests.
- Removed the artificial Cohen's Kappa demonstration because it did not use real team annotation data.

### 6. Documentation

- Wrote `PREPROCESSING_PIPELINE.md`.
- Updated `README.md` for the corrected SemEval dataset and final project structure.
- Documented weak opinion labels, repeated-term ambiguity, random splitting, and reproducibility limitations.
- Documented the supplied experiment output without inventing unavailable metrics.

## GitHub Verification

The commits pushed through `BeyzaESER22` should identify the validation, modular preprocessing, modular neural components, evaluation safeguards, tests, and documentation described above. Commit hashes or pull-request links may be added after the final push.
