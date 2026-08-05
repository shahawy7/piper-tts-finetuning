# Arabic Piper TTS Fine-Tuning 🇸🇦⚡

Fine-tuning the **Piper Arabic `ar_JO` (`ar_JO-kareem-medium`)** text-to-speech model using high-quality Modern Standard Arabic (MSA) datasets (e.g. `NightPrince/Arabic-professional-voice`). Optimized for both local GPU workstation execution (e.g. NVIDIA RTX 5090) and Google Colab environments.

---

## 📌 Project Overview

Piper is a fast, local neural text-to-speech system based on VITS. This repository provides an end-to-end workflow to fine-tune the Arabic model to achieve high-fidelity diacritized speech while maintaining real-time performance.

### Key Features
- 💻 **Local Workstation Ready**: Clean local workstation configuration (`local` branch) without Google Colab or Google Drive hardcoded paths. Fully tested on Linux GPUs (RTX 5090 / RTX 4090 / A100).
- 🔄 **Reproducible Training**: Auto-checkpoint detection and seamless resume capability across training sessions.
- 📁 **Separation of Code & Data**: No audio files or binary checkpoints are stored in Git.
- 📊 **Comprehensive Evaluation**: Automated benchmark synthesis, Real-Time Factor (RTF) timing, WER/CER tracking, and TensorBoard logging.
- ⚡ **ONNX Export**: Streamlined pipeline to convert PyTorch Lightning checkpoints to optimized ONNX models ready for Piper inference engines.
- 🎚️ **Speed Control**: Adjust voice speed via `--length-scale` parameter across all scripts.
- 🐣 **[Beginner's Guide](docs/beginner_guide.md)** & **[Local Workstation Guide](docs/LOCAL_SETUP.md)**.

---

## 🚀 Quick Start (Local GPU Workstation)

For detailed local Linux workstation setup (e.g. NVIDIA RTX 5090), see **[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**.

### 1. Run Complete Automated Pipeline via CLI
```bash
python scripts/run_local_pipeline.py --config configs/experiment001.yaml --epochs 50 --batch-size 32
```

### 2. Or Run Interactive Local Notebook
```bash
jupyter lab notebooks/local_piper_finetuning.ipynb
```

---

## ☁️ Quick Start (Google Colab)

1. Open `notebooks/arabic_piper_finetuning.ipynb` in Google Colab.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**.
3. Run all cells top-to-bottom.

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
│   ├── run_local_pipeline.py     # End-to-end local GPU workstation pipeline
│   ├── download_dataset.py       # Fetch dataset & pretrained checkpoints
│   ├── prepare_dataset.py        # LJSpeech formatting, audio resampling & splits
│   ├── test_local.py             # Test models locally on CPU (before & after training)
│   ├── benchmark.py              # Synthesize benchmark test sentences & measure RTF
│   ├── inference.py              # CLI text-to-speech synthesis
│   ├── evaluate.py               # Objective metrics calculation (WER, CER, RTF)
│   └── export_model.py           # Convert PyTorch Lightning .ckpt to ONNX + json config
│
├── notebooks/
│   ├── local_piper_finetuning.ipynb   # Local GPU workstation workflow notebook
│   └── arabic_piper_finetuning.ipynb # Google Colab workflow notebook
│
├── benchmark/                    # Benchmark evaluation assets
│   ├── benchmark_sentences.txt   # Diacritized MSA test sentences
│   └── expected_notes.md         # Subjective audio quality evaluation rubric
│
└── docs/                         # Detailed documentation
    ├── LOCAL_SETUP.md            # Workstation execution guide (RTX 5090 / Linux)
    ├── beginner_guide.md         # Step-by-step guide for beginners
    ├── local_testing_guide.md    # Local CPU testing guide (before & after training)
    ├── training_plan.md          # Technical architecture & pipeline plan
    └── experiment_log.md         # Matrix tracking experiment metrics
```

---

## 📜 License & Acknowledgments

- Base Architecture: [Piper TTS by Rhasspy](https://github.com/rhasspy/piper) (VITS architecture)
- Base Model: `ar_JO-kareem-medium`
- Dataset: [NightPrince/Arabic-professional-voice](https://huggingface.co/datasets/NightPrince/Arabic-professional-voice)
