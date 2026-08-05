#!/usr/bin/env python3
"""
Script: run_local_pipeline.py
Description: End-to-end local workstation execution pipeline for Arabic Piper TTS fine-tuning.
Supports local GPU training (e.g., RTX 5090) and background SSH daemon execution.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import yaml
from pathlib import Path

def setup_logging(log_file_path: Path = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True
    )

def handle_signal(sig, frame):
    logging.info(f"\n⚠️  Received signal {sig}. Interrupted cleanly by user or system.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def run_step(command: list, description: str):
    logging.info(f"\n==========================================")
    logging.info(f"🚀 [STEP] {description}")
    logging.info(f"Running: {' '.join(command)}")
    logging.info(f"==========================================\n")
    res = subprocess.run(command)
    if res.returncode != 0:
        logging.error(f"❌ Failed during step: {description}")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run complete local Arabic Piper fine-tuning pipeline.")
    parser.add_argument("--config", type=str, default="configs/experiment001.yaml", help="Path to YAML experiment config.")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory for storing data/models.")
    parser.add_argument("--epochs", type=int, default=None, help="Additional fine-tuning epochs to train (overrides config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU.")
    parser.add_argument("--precision", type=str, default="32", choices=["32", "16-mixed", "bf16-mixed"], help="PyTorch Lightning precision.")
    parser.add_argument("--devices", type=str, default="1", help="Number of GPUs or GPU IDs to use (e.g. 1 or '0,').")
    parser.add_argument("--log-file", type=str, default=None, help="Path to file for logging execution output.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download step if already present.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation step if already present.")
    args = parser.parse_args()

    if args.log_file:
        setup_logging(Path(args.log_file))
    else:
        setup_logging()

    cfg = load_config(args.config)
    data_root = Path(args.data_root)

    dataset_dir = data_root / cfg.get("paths", {}).get("datasets_dir", "datasets/experiment001")
    processed_dir = data_root / cfg.get("paths", {}).get("processed_dir", "processed/experiment001")
    checkpoints_dir = data_root / cfg.get("paths", {}).get("checkpoints_dir", "checkpoints/experiment001")
    base_ckpt_dir = data_root / "checkpoints" / "base"
    outputs_dir = data_root / cfg.get("paths", {}).get("outputs_dir", "outputs/experiment001")

    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 32)
    fine_tune_epochs = args.epochs or cfg.get("training", {}).get("epochs", 50)
    base_epoch = 5079
    target_max_epochs = base_epoch + fine_tune_epochs

    # Step 1: Download
    if not args.skip_download:
        run_step(
            [sys.executable, "scripts/download_dataset.py", "--config", args.config, "--data-root", str(data_root)],
            "Download Dataset & Base Checkpoint"
        )
    else:
        logging.info("Skipping dataset download step (--skip-download specified).")

    # Step 2: Prepare
    if not args.skip_prepare:
        run_step(
            [sys.executable, "scripts/prepare_dataset.py", "--config", args.config, "--data-root", str(data_root)],
            "Prepare Dataset (Phonemization & Spectrogram Caching)"
        )
    else:
        logging.info("Skipping dataset preparation step (--skip-prepare specified).")

    # Step 3: Checkpoint Detection & Target Epoch Calculation
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    existing_ckpts = sorted(checkpoints_dir.rglob("*.ckpt"))

    if existing_ckpts:
        resume_ckpt = str(existing_ckpts[-1])
        logging.info(f"✅ Resuming fine-tuning from latest checkpoint: '{resume_ckpt}'")
    else:
        candidates = list(base_ckpt_dir.rglob("*.ckpt")) if base_ckpt_dir.exists() else []
        if candidates:
            resume_ckpt = str(candidates[0])
            logging.info(f"🆕 Fine-tuning from base checkpoint: '{resume_ckpt}'")
        else:
            logging.error(f"❌ Base checkpoint not found in '{base_ckpt_dir}'!")
            sys.exit(1)

    logging.info(f"🎯 Target max_epochs: {target_max_epochs} (Base {base_epoch} + Fine-tune {fine_tune_epochs})")

    # Step 4: Fine-Tuning Execution
    train_cmd = [
        sys.executable, "-m", "piper_train",
        "--dataset-dir", str(processed_dir),
        "--accelerator", "gpu",
        "--devices", str(args.devices),
        "--batch-size", str(batch_size),
        "--validation-split", "0.05",
        "--max_epochs", str(target_max_epochs),
        "--precision", str(args.precision),
        "--checkpoint-epochs", str(cfg.get("training", {}).get("checkpoint_every_epochs", 5)),
        "--default_root_dir", str(checkpoints_dir),
        "--resume_from_checkpoint", resume_ckpt,
    ]
    run_step(train_cmd, "Execute Piper Fine-Tuning")

    # Step 5: Export to ONNX
    new_ckpts = sorted(checkpoints_dir.rglob("*.ckpt"))
    if not new_ckpts:
        logging.error("❌ No checkpoint found after training.")
        sys.exit(1)

    best_ckpt = str(new_ckpts[-1])
    onnx_out = outputs_dir / "ar_JO_finetuned.onnx"
    config_json = processed_dir / "config.json"

    export_cmd = [
        sys.executable, "scripts/export_model.py",
        "--checkpoint", best_ckpt,
        "--output-onnx", str(onnx_out),
        "--config-json", str(config_json),
    ]
    run_step(export_cmd, "Export Fine-Tuned Model to ONNX")

    # Step 6: Fine-Tuned Benchmark
    benchmark_out = data_root / "outputs" / "finetuned_benchmark"
    sentences_txt = Path("benchmark/benchmark_sentences.txt")

    if sentences_txt.exists() and onnx_out.exists():
        benchmark_cmd = [
            sys.executable, "scripts/benchmark.py",
            "--model", str(onnx_out),
            "--model-config", str(Path(str(onnx_out) + ".json")),
            "--sentences", str(sentences_txt),
            "--output-dir", str(benchmark_out),
        ]
        run_step(benchmark_cmd, "Benchmark Fine-Tuned Model")

    logging.info(f"\n==========================================")
    logging.info(f"✅ Local Fine-Tuning Pipeline Complete!")
    logging.info(f"Fine-tuned model output: {onnx_out}")
    logging.info(f"==========================================\n")

if __name__ == "__main__":
    main()
