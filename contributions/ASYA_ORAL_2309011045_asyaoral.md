# Individual Contribution — Asya ORAL

## Student Information

- **Student ID:** 2309011045
- **GitHub Username:** `asyaoral`
- **Primary Role:** Configurable experiment execution, reproducibility, artefact management, and presentation preparation

## Contribution Overview

I contributed to the later modular version of the project by helping convert the original hard-coded experiment into a reproducible command-line workflow with explicit configuration and output management.

## Completed Work

### 1. Configurable Experiment Runner

Contributed to `src/run_experiment.py`, which runs HMM, CRF, and BiLSTM-CRF from a shared dataset split and supports configuration or command-line overrides for:

- dataset and output paths;
- random seed and sample size;
- train/validation/test proportions;
- model selection;
- neural training parameters; and
- CPU/GPU device choice.

### 2. Reproducibility and Configuration

- Contributed to `configs/default.json` and effective-configuration handling.
- Helped preserve the seed and record IDs used for experiment splits.
- Contributed to environment, metric, prediction, triplet, training-history, and checkpoint output handling through `src/experiment_io.py` and the runner.
- Helped ensure that the same test set is used by the three configurable-runner models.

### 3. Runner Validation

Contributed tests covering configuration overrides, deterministic and disjoint splitting, prediction round trips, and a lightweight HMM smoke run.

### 4. Presentation

Together with Nilasu Yıldız, I prepared the project presentation in PPTX format.

## Scope Clarification

The original all-in-one models were implemented by Meltem Çelik and Berk Ötün. My contribution concerns the later configurable execution and reproducibility layer. The final Colab measurements are shared project results and should not be interpreted as sole personal output.

## GitHub Verification

The commits pushed through `asyaoral` should identify the runner, configuration, artefact-management, related testing, and presentation work described above. Commit hashes or pull-request links may be added after the final push.
