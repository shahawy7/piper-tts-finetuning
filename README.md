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
- 🎚️ **Speed Control**: Adjust voice speed via `--length-scale` parameter across all scripts.
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
├── notebooks/
│   └── arabic_piper_finetuning.ipynb  # Complete pipeline in a single Colab notebook
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

## 🚀 Quick Start (Google Colab)

Everything runs from a **single notebook** — no need to juggle multiple files:

1. Push this repo to your GitHub account.
2. Open `notebooks/arabic_piper_finetuning.ipynb` in Google Colab:
   - **From GitHub tab**: Click **File → Open notebook → GitHub**, paste your repo URL, and select the notebook.
   - **Direct URL**: `https://colab.research.google.com/github/YOUR_USERNAME/piper-tts-finetuning/blob/main/notebooks/arabic_piper_finetuning.ipynb`
3. Edit the `REPO_URL` variable in the first cell with your GitHub username.
4. Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**.
5. Run all cells top-to-bottom.

The notebook contains 5 sections:
| Section | What it does |
|---|---|
| **⚙️ Section 1** | Clones repo from GitHub, mounts Drive, installs dependencies |
| **📦 Section 2** | Downloads dataset & prepares audio (22,050 Hz LJSpeech format) |
| **🔊 Section 3** | Baseline benchmark — generates "BEFORE" audio samples |
| **🏋️ Section 4** | Fine-tuning with auto-checkpoint to Drive (resumes on reconnect) |
| **📊 Section 5** | Exports ONNX model & compares "BEFORE" vs "AFTER" audio |

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

# 4. Control voice speed (< 1.0 faster, > 1.0 slower)
python3 scripts/test_local.py --mode baseline --length-scale 0.85 --text "مَرْحَبًا"

# 5. Interactive Terminal Mode
python3 scripts/test_local.py --mode baseline --interactive
```

---

## 📜 License & Acknowledgments

- Base Architecture: [Piper TTS by Rhasspy](https://github.com/rhasspy/piper) (VITS architecture)
- Base Model: `ar_JO-kareem-medium`
- Dataset: [NightPrince/Arabic-professional-voice](https://huggingface.co/datasets/NightPrince/Arabic-professional-voice)
