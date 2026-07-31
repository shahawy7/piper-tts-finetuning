#!/usr/bin/env python3
"""
Script: inference.py
Description: Synthesizes Arabic text to speech using a trained/fine-tuned Piper model,
with optional speed control (length_scale).
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def synthesize_text(model_path: Path, text: str, output_path: Path, speaker_id: int = 0, length_scale: float = 1.0):
    """Invokes Piper TTS engine to synthesize input Arabic text."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try system piper CLI binary
    piper_exe = shutil.which("piper")
    if piper_exe:
        cmd = [
            piper_exe,
            "--model", str(model_path),
            "--output_file", str(output_path),
            "--speaker", str(speaker_id),
            "--length_scale", str(length_scale)
        ]
        logging.info(f"Synthesizing text with piper binary (Length Scale: {length_scale}): '{text}'...")
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=text)
        if process.returncode == 0 and output_path.exists():
            logging.info(f"Audio synthesized successfully! Output saved to: '{output_path}'")
            return

    # 2. Python PiperVoice fallback
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        config_path = Path(str(model_path) + ".json")
        json_file_arg = str(config_path) if config_path.exists() else None
        voice = PiperVoice.load(str(model_path), config_path=json_file_arg)

        logging.info(f"Synthesizing text with python piper module (Length Scale: {length_scale}): '{text}'...")
        syn_cfg = SynthesisConfig(length_scale=length_scale)
        chunks = list(voice.synthesize(text, syn_config=syn_cfg))
        if chunks:
            with wave.open(str(output_path), "wb") as wav_file:
                chunk = chunks[0]
                wav_file.setnchannels(chunk.sample_channels)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setframerate(chunk.sample_rate)
                for c in chunks:
                    wav_file.writeframes(c.audio_int16_bytes)
            logging.info(f"Audio synthesized successfully! Output saved to: '{output_path}'")
            return
    except Exception as e:
        logging.error(f"Python Piper synthesis error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Piper Arabic TTS CLI Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to Piper ONNX model file (.onnx).")
    parser.add_argument("--text", type=str, required=True, help="Arabic text string to synthesize.")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV filename.")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (default: 0).")
    parser.add_argument("--length-scale", "--length_scale", type=float, default=1.0,
                        help="Phoneme length scale for voice speed (< 1.0 faster, > 1.0 slower, default: 1.0).")
    args = parser.parse_args()

    synthesize_text(Path(args.model), args.text, Path(args.output), speaker_id=args.speaker, length_scale=args.length_scale)

if __name__ == "__main__":
    main()
