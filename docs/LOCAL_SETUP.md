# 🖥️ Local Workstation Execution Guide (NVIDIA RTX 5090 / Linux)

This document provides step-by-step instructions for running the **Arabic Piper TTS Fine-Tuning** pipeline on a local Linux workstation equipped with high-performance GPUs (such as the NVIDIA RTX 5090).

---

## 1. Prerequisites & System Requirements

- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS recommended)
- **GPU**: NVIDIA GPU (RTX 5090, RTX 4090, A100, etc.) with CUDA 12.x drivers installed
- **Python**: Python 3.10 – 3.12
- **System Packages**: `espeak-ng`, `libespeak-ng-dev`, `build-essential`, `ffmpeg`

---

## 2. System Installation

Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y espeak-ng libespeak-ng-dev build-essential ffmpeg
```

---

## 3. Repository & Virtual Environment Setup

Clone the repository and switch to the `local` branch:
```bash
git clone https://github.com/YOUR_USERNAME/piper-tts-finetuning.git
cd piper-tts-finetuning
git checkout local
```

Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4. Directory Layout

When running locally, data, checkpoints, and models are stored relative to the project root directory:

```text
piper-tts-finetuning/
├── configs/                  # YAML configurations (experiment001.yaml)
├── datasets/                 # Raw HuggingFace dataset download
│   └── experiment001/
├── processed/                # LJSpeech WAVs, metadata.csv, dataset.jsonl, cached .pt tensors
│   └── experiment001/
├── checkpoints/              # Base model & PyTorch Lightning fine-tuning checkpoints
│   ├── base/
│   └── experiment001/
├── outputs/                  # Exported ONNX models and benchmark audio outputs
│   └── experiment001/
└── scripts/                  # Command line scripts
```

---

## 5. Running via Command Line (Automated Pipeline)

You can launch the complete end-to-end pipeline with one command:

```bash
python scripts/run_local_pipeline.py --config configs/experiment001.yaml --epochs 50 --batch-size 32
```

### Options:
- `--epochs 50`: Number of additional fine-tuning epochs to train.
- `--batch-size 32`: Batch size per GPU (default `32` for 32GB VRAM cards like RTX 5090).
- `--precision 32`: Precision mode (`32`, `16-mixed`, `bf16-mixed`).
- `--devices 1`: Number of GPUs or specific GPU ID.
- `--skip-download`: Skip dataset download if already downloaded.
- `--skip-prepare`: Skip dataset preprocessing if `dataset.jsonl` is ready.

---

## 6. Step-by-Step Command Line Execution

Alternatively, you can run each stage individually:

### 1. Download Dataset & Base Checkpoint
```bash
python scripts/download_dataset.py --config configs/experiment001.yaml --data-root .
```

### 2. Prepare Dataset (Phonemization & STFT Spectrogram Caching)
```bash
python scripts/prepare_dataset.py --config configs/experiment001.yaml --data-root .
```

### 3. Run Fine-Tuning
```bash
python -m piper_train \
    --dataset-dir processed/experiment001 \
    --accelerator gpu \
    --devices 1 \
    --batch-size 32 \
    --validation-split 0.05 \
    --max_epochs 5129 \
    --checkpoint-epochs 5 \
    --default_root_dir checkpoints/experiment001 \
    --resume_from_checkpoint checkpoints/base/ar/ar_JO/kareem/medium/epoch=5079-step=1682020.ckpt
```

### 4. Export Checkpoint to ONNX Model
```bash
python scripts/export_model.py \
    --checkpoint checkpoints/experiment001/lightning_logs/version_0/checkpoints/epoch=5124-step=1684360.ckpt \
    --output-onnx outputs/experiment001/ar_JO_finetuned.onnx \
    --config-json processed/experiment001/config.json
```

### 5. Benchmark Fine-Tuned Model
```bash
python scripts/benchmark.py \
    --model outputs/experiment001/ar_JO_finetuned.onnx \
    --sentences benchmark/benchmark_sentences.txt \
    --output-dir outputs/finetuned_benchmark
```

---

## 7. Running via Jupyter Notebook

Launch JupyterLab or VS Code Notebook:
```bash
jupyter lab notebooks/local_piper_finetuning.ipynb
```
Run cells top-to-bottom for interactive visualization and audio playback!
