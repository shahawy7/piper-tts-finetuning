# Technical Training Plan & Architecture Guide

## 1. Executive Summary

This document describes the technical implementation plan for fine-tuning the **Piper Arabic `ar_JO` (`ar_JO-kareem-medium`)** model using the **Arabic Professional Voice** dataset (`NightPrince/Arabic-professional-voice`).

---

## 2. Technical Architecture

- **Model Architecture**: VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech).
- **Phonemizer**: `espeak-ng` with Arabic language support (`ar`).
- **Input Representation**: Diacritized Arabic text → Phoneme sequences.
- **Audio Output**: 22,050 Hz, 16-bit PCM Mono WAV.
- **Training Framework**: PyTorch Lightning (`piper-train`).
- **Inference Runtime**: ONNX Runtime / C++ `piper` binary.

---

## 3. Data Pipeline & Processing Flow

```text
[Hugging Face Hub: NightPrince/Arabic-professional-voice]
                       │
                       ▼ (scripts/download_dataset.py)
            [Raw Audio & Metadata]
                       │
                       ▼ (scripts/prepare_dataset.py)
   ┌───────────────────┴───────────────────┐
   ▼                                       ▼
[Resampled 22.050kHz WAVs]       [LJSpeech metadata.csv]
(wavs/id.wav)                    (id|speaker|text)
   │                                       │
   └───────────────────┬───────────────────┘
                       ▼
            [Phonemized Data Cache]
                       │
                       ▼ (piper-train preprocess)
            [Processed Tensors (.pt)]
```

---

## 4. Fine-Tuning Protocol

1. **Pretrained Checkpoint Initializer**: Start from `ar_JO-kareem-medium.ckpt` (Base model trained on Jordanian speech corpus).
2. **Transfer Learning Parameters**:
   - Learning Rate: `1.0e-4` with AdamW optimizer.
   - Batch Size: `16` (Adjustable based on GPU VRAM e.g., T4/V100/A100 in Colab).
   - Precision: 32-bit float or 16-bit mixed precision (`fp16`).
   - Discriminator/Generator loss balance following VITS default.
3. **Checkpoint & Saving**:
   - Save top-3 checkpoints based on Validation Loss.
   - Save full PyTorch Lightning `.ckpt` every 5 epochs to `/content/drive/MyDrive/Arabic-Piper/checkpoints/experiment001/`.
   - Automatic resumption: Notebook checks drive for existing `.ckpt` files and picks the highest epoch.

---

## 5. Model Export & Deployment

Once training achieves target loss convergence:
1. Export PyTorch checkpoint to ONNX using `scripts/export_model.py`.
2. Generate matching `.onnx.json` model config file preserving phoneme maps and sample rate settings.
3. Run `scripts/benchmark.py` on fine-tuned model and compare against baseline.
