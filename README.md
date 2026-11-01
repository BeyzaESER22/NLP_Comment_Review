# Aspect–Opinion Extraction and Sentiment-Aware Sequence Labelling

## Project Overview

This project implements an NLP pipeline for extracting sentiment-aware aspect terms and weakly supervised opinion expressions from SemEval reviews. Review-level annotations are aligned to tokens and converted into BIO sequences. Four approaches are examined:

1. Hidden Markov Model (HMM) with Viterbi decoding;
2. feature-engineered linear-chain Conditional Random Field (CRF);
3. BiLSTM-CRF; and
4. BERT token classification in the Colab-oriented experiment.

Predicted spans are also converted into structured aspect–opinion–polarity records for qualitative analysis, aspect-based search, and knowledge-graph visualisation.

## Dataset

The packaged dataset is `data/REVIEWS_semeval.csv`. It contains **4,707 annotated reviews** collected from SemEval laptop, restaurant, and hotel datasets.

| Column | Description |
|---|---|
| `text` | Raw review text. |
| `aspect_terms` | Pipe-delimited explicit aspect terms. |
| `term_polarities` | Pipe-delimited polarity labels aligned with the terms. |
| `aspect_categories` | Optional category annotations. |
| `category_polarities` | Optional category-level polarities. |
| `source` | SemEval source split. |

The current audit found 7,681 usable aspect–polarity pairs and 317 aspect items with more than one possible literal occurrence. It found no unsupported polarities and no unaligned aspect terms.

## Project Structure

```text
project-root/
├── README.md
├── RESULTS.md
├── PREPROCESSING_PIPELINE.md
├── requirements.txt
├── configs/
│   └── default.json
├── data/
│   └── REVIEWS_semeval.csv
├── src/
│   ├── NLP2_Aspect_Opinion.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── classical_models.py
│   ├── evaluation.py
│   ├── experiment_io.py
│   ├── run_experiment.py
│   └── reporting.py
├── tests/
├── contributions/
│   ├── BEYZA_ESER_2309111064_BeyzaESER22.md
│   ├── ASYA_ORAL_2309011045_asyaoral.md
│   ├── NILASU_YILDIZ_2309011057_xyzta.md
│   ├── MELTEM_CELIK_220901591_meltemcelik.md
│   └── BERK_OTUN_220901605_berkotun.md
└── outputs/
    ├── checkpoints/
    ├── figures/
    ├── metrics/
    └── predictions/
```

## File and Folder Descriptions

### Documentation

- `README.md`: project purpose, structure, installation, execution, and limitations.
- `PREPROCESSING_PIPELINE.md`: detailed preprocessing stages and commands.
- `RESULTS.md`: provenance, metrics, error analysis, and interpretation of the completed Colab run.

### Source Code

- `src/data_validation.py`: validates columns, text, aspect–polarity pairs, supported values, alignment, and repeated occurrences.
- `src/preprocessing.py`: authoritative occurrence-aware alignment, POS tagging, and BIO conversion used by both experiment entry points.
- `src/models.py`: PyTorch dataset, padded collation, and BiLSTM-CRF architecture.
- `src/classical_models.py`: HMM, Viterbi decoder, CRF features, training, and prediction.
- `src/evaluation.py`: exact-span combined, aspect, and opinion metrics.
- `src/experiment_io.py`: configuration, environment, metric, prediction, and checkpoint I/O.
- `src/run_experiment.py`: reproducible HMM/CRF/BiLSTM-CRF command-line runner.
- `src/reporting.py`: detailed metrics, error exports, confusion matrices, and figures.
- `src/NLP2_Aspect_Opinion.py`: Colab-oriented experiment including HMM, CRF, BiLSTM-CRF, BERT, search, and graph demonstrations. It imports the shared preprocessing implementation rather than maintaining a second labelling function.

### Other Folders

- `data/`: packaged SemEval CSV.
- `configs/`: version-controlled defaults; `sample_size: 0` means use all records.
- `tests/`: preprocessing, evaluation, runner, and reporting tests.
- `contributions/`: one auditable Markdown file per team member.
- `outputs/metrics/`: validation and reported-result summaries.
- `outputs/figures/`: generated figures, including the supplied graph.
- `outputs/checkpoints/` and `outputs/predictions/`: locations for generated artefacts. Reusable checkpoint and prediction files were not supplied with the pasted run output.

## Preprocessing Summary

1. Required columns are validated.
2. Reviews are tokenised with `\S+` while retaining character offsets.
3. NLTK supplies POS tags.
4. Pipe-delimited annotations are parsed and checked.
5. Repeated aspects consume successive unoccupied occurrences.
6. Aspect spans become polarity-aware BIO labels.
7. Adjectives and adverbs outside aspect spans become weak `B-OPINION`/`I-OPINION` labels.

Opinion labels are heuristic and are not independent human annotations. The project does not report a fabricated inter-annotator agreement score.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import nltk; nltk.download("punkt"); nltk.download("averaged_perceptron_tagger"); nltk.download("averaged_perceptron_tagger_eng")'
```

## Running the Project

Validate the complete dataset:

```bash
python src/data_validation.py data/REVIEWS_semeval.csv --output-dir outputs/metrics
```

Run HMM, CRF, and BiLSTM-CRF on all 4,707 reviews:

```bash
python src/run_experiment.py \
  --data data/REVIEWS_semeval.csv \
  --config configs/default.json \
  --output-dir outputs
```

Generate reports from saved predictions:

```bash
python src/reporting.py \
  --predictions-dir outputs/predictions \
  --output-dir outputs/metrics \
  --figures-dir outputs/figures
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

The all-in-one script can be run from the project root with `python src/NLP2_Aspect_Opinion.py`. BERT requires downloading `bert-base-uncased`, so internet access is required.

## Reported Results

The completed Colab run used all 4,707 reviews and produced a 3,765/942 outer train/test split. BiLSTM-CRF and BERT used 377 records from the training portion for validation.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| HMM + Viterbi | 0.6716 | 0.6204 | 0.6450 |
| Linear-chain CRF | 0.7560 | 0.6976 | 0.7256 |
| BiLSTM-CRF | 0.7570 | 0.7062 | 0.7307 |
| BERT token classification | Not exported | Not exported | **0.7368** |

These values are transcribed from the supplied completed run, not regenerated while preparing this archive. The shared preprocessing function was consolidated after that run, so a final rerun is recommended if exact post-refactor reproducibility is required. See `RESULTS.md`.

## Methodological Limitations

- Opinion spans are POS-derived weak labels; high opinion performance is not evaluation against independent human opinion annotations.
- `conflict` is mapped to `NEU`.
- Original character offsets are unavailable, so repeated identical mentions remain ambiguous.
- Nearest-opinion pairing does not model syntax.
- The run randomly split merged SemEval records. Since `source` contains official train/test names, the scores are not directly comparable with a strict official SemEval benchmark.
- BERT precision and recall were absent from the supplied output and are therefore not invented here.

## Reproducibility and Contributions

Python, NumPy, and PyTorch seeds are set to `42`. Neural results can still vary across hardware and package versions. Contribution files must describe only work the named member actually performed; GitHub commits and diffs are the authoritative evidence. Every member must verify and complete their own file before submission.
