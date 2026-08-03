#!/usr/bin/env python3
"""
Script: export_model.py
Description: Converts PyTorch Lightning (.ckpt) checkpoint into ONNX (.onnx) model with matching JSON config.
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import torch
from pathlib import Path

# Auto-install onnxscript if required by PyTorch 2.6 ONNX exporter
try:
    import onnxscript
except ImportError:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxscript", "onnx"], check=True)
    except Exception:
        pass

# PyTorch 2.6 safe_globals fix
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def export_ckpt_to_onnx(ckpt_path: Path, output_onnx_path: Path, config_json_path: Path = None):
    """Exports PyTorch checkpoint to ONNX format using piper_train.export_onnx module."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' does not exist.")

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Exporting PyTorch checkpoint '{ckpt_path}' to ONNX '{output_onnx_path}'...")

    cmd = [
        sys.executable,
        "-c",
        f"import pathlib, torch; torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath]); "
        f"from piper_train.export_onnx import main; import sys; sys.argv=['export_onnx', '{ckpt_path}', '{output_onnx_path}']; main()"
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"Export output: {res.stdout}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error during ONNX export: {e}")
        if hasattr(e, "stderr") and e.stderr:
            logging.error(f"Error details: {e.stderr}")
        raise RuntimeError(f"ONNX export failed: {e}") from e

    # Locate source config.json if not provided
    if config_json_path is None or not config_json_path.exists():
        candidate = ckpt_path.parent.parent.parent / "processed" / "experiment001" / "config.json"
        if candidate.exists():
            config_json_path = candidate
        else:
            candidate_base = ckpt_path.parent.parent / "base" / "config.json"
            if candidate_base.exists():
                config_json_path = candidate_base

    # Ensure matching .onnx.json metadata config exists
    target_json_path = Path(str(output_onnx_path) + ".json")
    if config_json_path and config_json_path.exists():
        logging.info(f"Using source config JSON: '{config_json_path}'")
        with open(config_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        logging.info("Writing standard fallback ONNX configuration metadata...")
        meta = {
            "dataset": "Arabic-professional-voice",
            "audio": {
                "sample_rate": 22050,
                "quality": "medium"
            },
            "espeak": {
                "voice": "ar"
            },
            "language": {
                "code": "ar_JO",
                "family": "ar",
                "region": "JO"
            },
            "num_speakers": 1,
            "num_symbols": 200,
            "phoneme_type": "espeak"
        }

    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"✅ Export complete. Model: '{output_onnx_path}', Config: '{target_json_path}'")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Lightning checkpoint to ONNX model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch .ckpt checkpoint file.")
    parser.add_argument("--output-onnx", type=str, required=True, help="Path for target .onnx file.")
    parser.add_argument("--config-json", type=str, default=None, help="Path to source config .json file if available.")
    args = parser.parse_args()

    export_ckpt_to_onnx(Path(args.checkpoint), Path(args.output_onnx), Path(args.config_json) if args.config_json else None)

if __name__ == "__main__":
    main()
