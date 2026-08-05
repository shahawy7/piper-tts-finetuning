# Arabic Piper TTS Fine-Tuning 🇸🇦⚡

Fine-tuning the **Piper Arabic `ar_JO` (`ar_JO-kareem-medium`)** text-to-speech model using high-quality Modern Standard Arabic (MSA) datasets (e.g. `NightPrince/Arabic-professional-voice`). Optimized for both local GPU workstation execution (e.g. NVIDIA RTX 5090 via SSH background tasks) and Google Colab environments.

---

## 📌 Project Overview

Piper is a fast, local neural text-to-speech system based on VITS. This repository provides an end-to-end workflow to fine-tune the Arabic model to achieve high-fidelity diacritized speech while maintaining real-time performance.

### Key Features
- 💻 **Local Workstation Ready**: Clean local workstation configuration (`local` branch) without Google Colab or Google Drive hardcoded paths. Fully tested on Linux GPUs (RTX 5090 / RTX 4090 / A100).
- 🌐 **SSH Background Execution**: Daemon launcher (`./scripts/start_training.sh`) allows training to run in the background over SSH, safe from disconnection.
- 🔄 **Reproducible Training**: Auto-checkpoint detection and seamless resume capability across training sessions.
- 📁 **Separation of Code & Data**: No audio files or binary checkpoints are stored in Git.
- 📊 **Comprehensive Evaluation**: Automated benchmark synthesis, Real-Time Factor (RTF) timing, WER/CER tracking, and TensorBoard logging.
- ⚡ **ONNX Export**: Streamlined pipeline to convert PyTorch Lightning checkpoints to optimized ONNX models ready for Piper inference engines.
- 🐣 **[Beginner's Guide](docs/beginner_guide.md)** & **[Local Workstation Guide](docs/LOCAL_SETUP.md)**.

---

## 🌐 Quick Start (SSH Remote Workstation / RTX 5090)

For detailed SSH workstation setup, see **[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**.

### 1. Launch Background Fine-Tuning over SSH
```bash
./scripts/start_training.sh --epochs 50 --batch-size 32
```
*You can now safely exit your SSH connection! Training continues in the background.*

### 2. Monitor Progress / GPU Stats
```bash
# Check PID status & GPU utilization
./scripts/status_training.sh

# Live tail logs
tail -f logs/latest.log
```

### 3. Graceful Stop & Final Checkpoint Save
```bash
./scripts/stop_training.sh
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
├── scripts/                      # Core python & bash execution scripts
│   ├── start_training.sh         # Launch background fine-tuning daemon (SSH safe)
│   ├── status_training.sh        # Check background training PID & GPU utilization
│   ├── stop_training.sh          # Gracefully stop background training (save checkpoint)
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
    ├── LOCAL_SETUP.md            # Workstation execution & SSH background guide
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
