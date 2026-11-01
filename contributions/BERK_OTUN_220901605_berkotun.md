# Individual Contribution — Berk ÖTÜN

## Student Information

- **Student ID:** 220901605
- **GitHub Username:** `berkotun`
- **Primary Role:** Original data preparation, BIO conversion, search, and knowledge-graph demonstrations

## Contribution Overview

I co-developed the original all-in-one Colab implementation with Meltem Çelik. My work focused on converting the merged SemEval reviews into model-ready sequences and building the downstream search and visualisation demonstrations. Later modular preprocessing and validation work is documented separately in Beyza Eser's contribution file.

## Completed Work

### 1. Original Dataset Loading and Conversion

- Added local and Google Colab loading support for the original experiment.
- Checked that the required `text`, `aspect_terms`, and `term_polarities` columns were available.
- Implemented the original `build_bio_dataset` workflow in the all-in-one experiment.
- Tokenised reviews while retaining character offsets.
- Projected pipe-delimited aspect terms and polarities onto sentiment-aware BIO labels.
- Mapped `positive`, `negative`, `neutral`, and `conflict` annotations to the project's three polarity suffixes, with `conflict` mapped to `NEU`.
- Added NLTK POS tagging and the adjective/adverb-based weak opinion-span heuristic.
- Added the reproducible random seed, sampling/splitting logic, tag-distribution output, and preprocessing sanity checks.

### 2. Aspect-Based Search

Implemented `search_reviews_by_aspect`, which retrieves structured prediction records using an aspect query and an optional polarity filter. The demonstration shows separate positive and negative search results for the same aspect.

### 3. Knowledge-Graph Demonstration

Implemented `plot_opinion_knowledge_graph`, which creates a directed graph from the extracted records:

- the domain node connects to aspect nodes;
- aspect nodes connect to predicted opinion expressions; and
- edge colours represent positive, negative, or neutral polarity.

The completed Colab experiment displays the generated graph as part of the downstream demonstration.

## Data Covered

The corrected project dataset is `data/REVIEWS_semeval.csv` and contains 4,707 reviews. The completed experiment used a reproducible 3,765/942 outer train/test split. The merged file contains SemEval sources from 2014, 2015, and 2016.

## Main Components

| Component | Contribution |
|---|---|
| Original `build_bio_dataset` | Review text and annotations to BIO/POS sequences |
| Sampling and split | Reproducible experiment preparation |
| `search_reviews_by_aspect` | Aspect- and polarity-filtered review retrieval |
| `plot_opinion_knowledge_graph` | Aspect–opinion relation visualisation |

## Validation and Limitations

- Printed the processed-record counts and BIO-tag distribution as preprocessing checks.
- Inspected example token/POS/BIO sequences before model training.
- The original preprocessing was later consolidated into the occurrence-aware `src/preprocessing.py`; that modular refactor is not claimed here.
- Opinion spans are weak POS-based labels rather than independent human annotations.

## GitHub Verification

The commits pushed through `berkotun` should identify the original data-conversion, search, and graph changes described above. Commit hashes or pull-request links may be added after the final push.
