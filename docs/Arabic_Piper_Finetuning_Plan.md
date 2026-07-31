# Arabic Piper Fine-Tuning Project Plan

## Goal

Fine-tune the Piper Arabic `ar_JO` model using the **Arabic Professional
Voice** dataset to improve Modern Standard Arabic (MSA) pronunciation
while preserving inference speed.

## Design Principles

-   Repository contains **only code, notebooks, configs, and
    documentation**.
-   No datasets or checkpoints are committed.
-   Every Colab session is reproducible from scratch.
-   Training data is downloaded automatically during the session.
-   Dataset preprocessing happens inside Colab.
-   Checkpoints, logs, and outputs are stored on Google Drive.
-   Every experiment is versioned and resumable.

------------------------------------------------------------------------

# Repository Structure

``` text
Arabic-Piper-Finetuning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_environment.ipynb
│   ├── 02_dataset.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_train.ipynb
│   └── 05_evaluate.ipynb
│
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── benchmark.py
│   ├── inference.py
│   ├── evaluate.py
│   └── export_model.py
│
├── configs/
│   ├── experiment001.yaml
│   └── experiment_template.yaml
│
├── benchmark/
│   ├── benchmark_sentences.txt
│   └── expected_notes.md
│
├── docs/
│   ├── training_plan.md
│   └── experiment_log.md
│
└── .github/
```

------------------------------------------------------------------------

# Google Drive Layout

``` text
Arabic-Piper/
├── datasets/
├── processed/
├── checkpoints/
├── tensorboard/
├── logs/
├── outputs/
└── metrics/
```

------------------------------------------------------------------------

# Colab Workflow

1.  Mount Google Drive.
2.  Clone/update this repository.
3.  Install dependencies.
4.  Download pretrained Piper model.
5.  Download Arabic Professional Voice dataset.
6.  Prepare metadata and audio.
7.  Split train/validation.
8.  Run baseline synthesis.
9.  Train.
10. Save checkpoints to Drive.
11. Evaluate every checkpoint.
12. Export final model.

Each session should detect existing checkpoints and resume
automatically.

------------------------------------------------------------------------

# Experiment Lifecycle

## Phase 1

Environment setup and dependency installation.

## Phase 2

Automatic dataset download and preprocessing.

## Phase 3

Baseline evaluation using predefined benchmark sentences.

## Phase 4

Fine-tuning with periodic checkpoints.

## Phase 5

Checkpoint evaluation: - Training loss - Validation loss - WER - CER -
Inference speed - Listening notes

## Phase 6

Final export and benchmarking.

------------------------------------------------------------------------

# Reproducibility

Each experiment folder should contain:

-   configuration
-   checkpoint(s)
-   logs
-   TensorBoard data
-   generated benchmark audio
-   metrics.csv

No manual editing should be required between Colab sessions.

------------------------------------------------------------------------

# Long-Term Goal

The repository should allow launching a complete fine-tuning run from a
fresh Colab runtime with minimal manual steps while keeping experiments
organized, repeatable, and easy to compare.
