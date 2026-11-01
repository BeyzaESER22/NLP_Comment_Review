# Experiment Results

## Provenance

These values were transcribed from the completed Colab output supplied by the team. The run used `REVIEWS_semeval.csv`, seed `42`, all 4,707 reviews, a 3,765/942 outer train/test split, eight BiLSTM-CRF epochs, and eight BERT epochs. Neural validation used 377 records from the outer training portion.

The all-in-one script was subsequently cleaned to remove notebook-only syntax and the artificial Cohen's Kappa demonstration. It now imports shared occurrence-aware preprocessing. The values below document the supplied run; a rerun is recommended before claiming exact post-refactor reproduction.

## Overall Results

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| HMM + Viterbi | 0.6716 | 0.6204 | 0.6450 |
| Linear-chain CRF | 0.7560 | 0.6976 | 0.7256 |
| BiLSTM-CRF | 0.7570 | 0.7062 | 0.7307 |
| BERT token classification | Not exported | Not exported | **0.7368** |

BERT had the highest reported F1. Its precision and recall were absent from the supplied output and are intentionally unreported.

## BiLSTM-CRF Detailed Report

The detailed report was printed before BERT was added and therefore describes BiLSTM-CRF.

| Entity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ASP_NEG | 0.3898 | 0.2353 | 0.2935 | 391 |
| ASP_NEU | 0.2292 | 0.0445 | 0.0746 | 247 |
| ASP_POS | 0.5078 | 0.5650 | 0.5349 | 862 |
| OPINION | 0.9517 | 0.9722 | 0.9619 | 1,764 |
| Micro average | 0.7570 | 0.7062 | 0.7307 | 3,264 |
| Macro average | 0.5196 | 0.4543 | 0.4662 | 3,264 |
| Weighted average | 0.7125 | 0.7062 | 0.7019 | 3,264 |

Opinion performance is much higher than aspect performance because opinion labels are generated with a POS heuristic. Neutral-aspect recall is the weakest reported category.

## Neural Training History

### BiLSTM-CRF

| Epoch | Loss/token | Validation F1 |
|---:|---:|---:|
| 1 | 0.7428 | 0.6488 |
| 2 | 0.4542 | 0.6621 |
| 3 | 0.3850 | 0.6810 |
| 4 | 0.3425 | 0.6949 |
| 5 | 0.3115 | 0.7055 |
| 6 | 0.2905 | 0.7038 |
| 7 | 0.2682 | 0.6947 |
| 8 | 0.2527 | 0.7125 |

### BERT

| Epoch | Loss | Validation F1 |
|---:|---:|---:|
| 1 | 0.8694 | 0.6339 |
| 2 | 0.3305 | 0.7209 |
| 3 | 0.2231 | 0.7149 |
| 4 | 0.1520 | 0.7185 |
| 5 | 0.1048 | 0.7234 |
| 6 | 0.0743 | 0.7288 |
| 7 | 0.0550 | 0.7261 |
| 8 | 0.0446 | 0.7231 |

The BERT load report showed newly initialised classifier weights and unused pretraining-head weights. This is expected when adapting `bert-base-uncased` to token classification.

## Error Analysis

| Error type | HMM | CRF | BiLSTM-CRF |
|---|---:|---:|---:|
| Missed | 513 | 376 | 353 |
| Boundary | 398 | 300 | 303 |
| Polarity | 309 | 294 | 286 |
| Spurious | 285 | 168 | 173 |
| Type | 19 | 17 | 17 |

| Pattern | Sentences | Pattern error rate | Other sentences |
|---|---:|---:|---:|
| Negation | 206 | 0.796 | 0.689 |
| Contrast | 111 | 0.838 | 0.696 |
| Multiple aspects | 363 | 0.821 | 0.644 |

The pattern analysis was produced for the BiLSTM-CRF before BERT was added. Negation, contrast, and multi-aspect sentences were harder than their comparison groups.

## Knowledge Graph

The supplied graph is stored at `outputs/figures/aspect_opinion_knowledge_graph.png`. Blue nodes represent aspects, light-green nodes predicted opinions, and the yellow node the product/venue root. Green, red, and orange aspect–opinion edges indicate positive, negative, and neutral polarity.

## Interpretation Limits

- Opinion scores measure agreement with heuristic POS labels, not human annotations.
- Nearest-span pairing can be semantically incorrect.
- Randomly splitting merged SemEval sources does not reproduce the official benchmark protocol.
- Checkpoints and prediction files were not supplied, so this is a documented result transcript rather than a regenerated artefact bundle.
