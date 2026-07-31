# Arabic Piper TTS Fine-Tuning 🇸🇦⚡

Fine-tuning the **Piper Arabic `ar_JO` (`ar_JO-kareem-medium`)** text-to-speech model using high-quality Modern Standard Arabic (MSA) datasets (e.g. `NightPrince/Arabic-professional-voice`). Optimized for reproducible training in Google Colab with automatic state persistence to Google Drive.

---

## 📌 Project Overview

Piper is a fast, local neural text-to-speech system based on VITS. This repository provides an end-to-end workflow to fine-tune the Arabic model to achieve high-fidelity diacritized speech while maintaining real-time performance.

### Key Features
- 🔄 **Reproducible Colab Sessions**: Automatic drive mounting, dataset fetching, and resume-from-checkpoint capability.
- 📁 **Separation of Code & Data**: No audio files or binary checkpoints are stored in Git.
- 📊 **Comprehensive Evaluation**: Automated benchmark synthesis, Real-Time Factor (RTF) timing, WER/CER tracking, and TensorBoard logging.
- ⚡ **ONNX Export**: Streamlined pipeline to convert PyTorch Lightning checkpoints to optimized ONNX models ready for Piper inference engines.
- 🐣 **[Beginner's Guide](docs/beginner_guide.md)**: New to fine-tuning? Read our step-by-step beginner guide!


---

## 📁 Repository Structure

```text
.
├── README.md                     # Project documentation & execution guide
├── requirements.txt              # Python dependencies for local & Colab environments
├── .gitignore                    # Excludes data, audio, checkpoints, and logs
│
├── configs/                      # Experiment YAML configuration files
│   ├── experiment_template.yaml  # General configuration schema
│   └── experiment001.yaml        # Active fine-tuning experiment config
│
├── scripts/                      # Core python execution scripts
│   ├── download_dataset.py       # Fetch dataset & pretrained checkpoints
│   ├── prepare_dataset.py        # LJSpeech formatting, audio resampling & splits
│   ├── test_local.py             # Test models locally on CPU (before & after training)
│   ├── benchmark.py              # Synthesize benchmark test sentences & measure RTF
│   ├── inference.py              # CLI text-to-speech synthesis
│   ├── evaluate.py               # Objective metrics calculation (WER, CER, RTF)
│   └── export_model.py           # Convert PyTorch Lightning .ckpt to ONNX + json config
│
├── notebooks/                    # Sequential Google Colab notebooks
│   ├── 01_environment.ipynb      # Step 1: Environment & dependency setup
│   ├── 02_dataset.ipynb          # Step 2: Dataset download & pre-processing
│   ├── 03_baseline.ipynb         # Step 3: Baseline model benchmarking
│   ├── 04_train.ipynb            # Step 4: Training & Google Drive check-pointing
│   └── 05_evaluate.ipynb         # Step 5: Export, evaluation & comparison
│
├── benchmark/                    # Benchmark evaluation assets
│   ├── benchmark_sentences.txt   # Diacritized MSA test sentences
│   └── expected_notes.md         # Subjective audio quality evaluation rubric
│
└── docs/                         # Detailed documentation
    ├── beginner_guide.md         # Step-by-step guide for beginners
    ├── local_testing_guide.md    # Local CPU testing guide (before & after training)
    ├── training_plan.md          # Technical architecture & pipeline plan
    └── experiment_log.md         # Matrix tracking experiment metrics


```

---

## ☁️ Google Drive Directory Layout

When running in Google Colab, persistent data is stored under Google Drive (`/content/drive/MyDrive/Arabic-Piper/`):

```text
Arabic-Piper/
├── datasets/        # Raw downloaded datasets
├── processed/       # Resampled audio (22,050 Hz) and metadata.csv
├── checkpoints/     # Training checkpoints (.ckpt files)
├── tensorboard/     # TensorBoard logs for visualization
├── logs/            # Console and execution logs
├── outputs/         # Baseline and experiment synthesis outputs
└── metrics/         # Evaluated metrics CSVs
```

---

## 🚀 Step-by-Step Workflow (Google Colab)

1. **Notebook 01 - Environment**: Mount Google Drive, clone repository, install system dependencies (`espeak-ng`, `piper-train`).
2. **Notebook 02 - Dataset**: Download the `NightPrince/Arabic-professional-voice` dataset and base checkpoint (`ar_JO-kareem-medium`). Format into LJSpeech layout.
3. **Notebook 03 - Baseline**: Synthesize benchmark MSA sentences using the baseline `ar_JO-kareem-medium` model to set quality & RTF baselines.
4. **Notebook 04 - Training**: Launch or resume fine-tuning with `piper_train`. Save checkpoints and TensorBoard logs directly to Google Drive.
5. **Notebook 05 - Evaluation**: Export the best checkpoint to `.onnx` and evaluate objective metrics (WER/CER, RTF) against the baseline.

---

## 🧪 Local CPU Testing

For detailed step-by-step instructions, see the **[Local Testing Guide](docs/local_testing_guide.md)**.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test Baseline model locally on CPU (Before Training)
python3 scripts/test_local.py --mode baseline --text "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ."

# 3. Test Fine-Tuned model locally on CPU (After Training)
python3 scripts/test_local.py --mode finetuned --model outputs/experiment001/ar_JO_finetuned.onnx

# 4. Interactive Terminal Mode
python3 scripts/test_local.py --mode baseline --interactive
```


---

## 📜 License & Acknowledgments

- Base Architecture: [Piper TTS by Rhasspy](https://github.com/rhasspy/piper) (VITS architecture)
- Base Model: `ar_JO-kareem-medium`
- Dataset: [NightPrince/Arabic-professional-voice](https://huggingface.co/datasets/NightPrince/Arabic-professional-voice)
