# Individual Contribution — Meltem ÇELİK

## Student Information

- **Student ID:** 220901591
- **GitHub Username:** `meltemcelik`
- **Primary Role:** Original sequence-model implementation, training, benchmarking, error analysis, and dataset correction

## Contribution Overview

I co-developed the original all-in-one Colab implementation with Berk Ötün. My work focused on the sequence models, training and evaluation workflow, structured information extraction, and correction of the dataset inconsistencies identified during the later validation review.

## Completed Work

### 1. HMM and Viterbi Decoder

- Implemented `HMMSequenceTagger` with start, transition, and emission estimates.
- Added additive smoothing and POS-specific unknown-word handling.
- Implemented log-space Viterbi decoding.
- Added an optional trellis view for inspecting decoding decisions.

### 2. Linear-Chain CRF

- Designed lexical, orthographic, prefix, suffix, POS, and neighbouring-token features.
- Configured the `sklearn-crfsuite` training pipeline and its L1/L2 regularisation.
- Added CRF prediction and entity-level evaluation.

### 3. Neural Models

- Implemented the original BiLSTM-CRF experiment with word and POS embeddings, a bidirectional LSTM, and CRF decoding.
- Implemented the neural training loop with word dropout, gradient clipping, validation monitoring, learning-rate scheduling, early stopping, and best-state restoration.
- Added the BERT token-classification experiment based on `bert-base-uncased`, including subword/word alignment, validation monitoring, and test prediction.

The later extraction of reusable neural components into `src/models.py` is documented separately in Beyza Eser's contribution file.

### 4. Benchmarking and Original Error Analysis

- Produced the original HMM/CRF/BiLSTM-CRF benchmark table and detailed classification report.
- Added side-by-side token predictions for qualitative comparison.
- Implemented the original printed error categorisation for polarity, type, boundary, missed, and spurious errors.
- Added difficult-pattern checks for negation, contrast, and multi-aspect sentences.
- Collected representative examples for the main error categories.

No Cohen's Kappa or inter-annotator agreement result is claimed because the project does not contain a real two-annotator dataset.

### 5. Structured Information Extraction

Implemented `extract_structured_triplets`, which converts predicted BIO spans into aspect, nearest-opinion, polarity, sentence-ID, and review-text records.

### 6. Dataset Correction

After Beyza Eser identified inconsistencies in the earlier dataset, I corrected the dataset content used by the final experiment. The corrected file is `data/REVIEWS_semeval.csv`, containing 4,707 reviews and aligned aspect–polarity annotations.

## Experiment Summary

The completed experiment used all 4,707 reviews with a 3,765/942 outer train/test split. Neural models used part of the outer training set for validation. HMM, CRF, BiLSTM-CRF, and BERT were evaluated with entity-level metrics. These are shared project results rather than sole personal contributions.

## Validation and Limitations

- Compared all models on the same held-out test portion in the original experiment.
- Monitored validation F1 during neural training.
- Inspected classification reports, side-by-side predictions, and error examples.
- Opinion labels are generated from POS tags and therefore are weak labels.
- The random split of merged SemEval sources is not the official benchmark protocol.

## GitHub Verification

The commits pushed through `meltemcelik` should identify the original model, training, evaluation, information-extraction, and dataset-correction changes described above. Commit hashes or pull-request links may be added after the final push.
