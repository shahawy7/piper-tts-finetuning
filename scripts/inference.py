#!/usr/bin/env python3
"""
Script: inference.py
Description: Synthesizes Arabic text to speech using a trained/fine-tuned Piper model.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def synthesize_text(model_path: Path, text: str, output_path: Path, speaker_id: int = 0):
    """Invokes Piper TTS engine to synthesize input Arabic text."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "piper",
        "--model", str(model_path),
        "--output_file", str(output_path),
        "--speaker", str(speaker_id)
    ]

    logging.info(f"Synthesizing text: '{text}'...")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate(input=text)

    if process.returncode != 0:
        logging.error(f"Piper error: {stderr}")
        sys.exit(1)

    logging.info(f"Audio synthesized successfully! Output saved to: '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Piper Arabic TTS CLI Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to Piper ONNX model file (.onnx).")
    parser.add_argument("--text", type=str, required=True, help="Arabic text string to synthesize.")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV filename.")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (default: 0).")
    args = parser.parse_args()

    synthesize_text(Path(args.model), args.text, Path(args.output), speaker_id=args.speaker)

if __name__ == "__main__":
    main()
