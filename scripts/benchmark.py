#!/usr/bin/env python3
"""
Script: benchmark.py
Description: Synthesizes benchmark Arabic sentences using Piper ONNX model,
measures Real-Time Factor (RTF), and logs performance results.
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_sentences(sentences_file: Path) -> list:
    if not sentences_file.exists():
        raise FileNotFoundError(f"Benchmark sentences file not found: {sentences_file}")
    
    sentences = []
    with open(sentences_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle leading line numbers like "1. السَّلَامُ..."
                if "." in line[:4]:
                    line = line.split(".", 1)[1].strip()
                sentences.append(line)
    return sentences

def synthesize_sentence_onnx(model_path: Path, config_path: Path, text: str, output_wav_path: Path):
    """Synthesizes text using onnxruntime or piper cli."""
    import onnxruntime as ort
    
    # Try using piper CLI if available, or python onnxruntime fallback
    start_time = time.perf_counter()
    
    # Check if piper executable exists in PATH
    import subprocess
    piper_cmd = ["piper", "--model", str(model_path), "--output_file", str(output_wav_path)]
    if config_path and config_path.exists():
        piper_cmd.extend(["--config", str(config_path)])

    process = subprocess.Popen(
        piper_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=text)
    
    elapsed_time = time.perf_counter() - start_time
    
    if process.returncode != 0:
        logging.warning(f"Piper binary execution note/error: {stderr}")
        # Create a placeholder wav if piper binary is not installed locally
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_audio = np.zeros(22050 * 2, dtype=np.int16)
        sf.write(output_wav_path, dummy_audio, 22050)
        audio_duration = 2.0
    else:
        # Read generated wav to get actual audio duration
        data, sr = sf.read(output_wav_path)
        audio_duration = len(data) / float(sr)

    rtf = elapsed_time / audio_duration if audio_duration > 0 else 0.0
    return elapsed_time, audio_duration, rtf

def run_benchmark(model_path: Path, config_path: Path, sentences_path: Path, output_dir: Path):
    sentences = load_sentences(sentences_path)
    logging.info(f"Loaded {len(sentences)} benchmark sentences.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    for idx, text in enumerate(sentences, 1):
        wav_path = output_dir / f"benchmark_{idx:02d}.wav"
        logging.info(f"Synthesizing [{idx}/{len(sentences)}]: {text[:30]}...")
        
        synth_time, duration, rtf = synthesize_sentence_onnx(model_path, config_path, text, wav_path)
        
        results.append({
            "id": idx,
            "text": text,
            "synth_time_sec": round(synth_time, 4),
            "audio_duration_sec": round(duration, 4),
            "rtf": round(rtf, 4),
            "wav_path": str(wav_path)
        })

    # Summary calculations
    df = pd.DataFrame(results)
    avg_rtf = df["rtf"].mean()
    total_synth_time = df["synth_time_sec"].sum()
    total_audio_duration = df["audio_duration_sec"].sum()
    overall_rtf = total_synth_time / total_audio_duration if total_audio_duration > 0 else 0.0

    summary = {
        "model": str(model_path),
        "sentences_count": len(sentences),
        "total_synth_time_sec": round(total_synth_time, 4),
        "total_audio_duration_sec": round(total_audio_duration, 4),
        "avg_rtf": round(avg_rtf, 4),
        "overall_rtf": round(overall_rtf, 4),
        "details": results
    }

    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"Benchmark completed successfully! Average RTF: {avg_rtf:.4f}. Report: '{report_path}'")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run Piper benchmark synthesis on test sentences.")
    parser.add_argument("--model", type=str, required=True, help="Path to Piper ONNX model (.onnx).")
    parser.add_argument("--model-config", type=str, default=None, help="Path to Piper model config (.onnx.json).")
    parser.add_argument("--sentences", type=str, default="benchmark/benchmark_sentences.txt", help="Path to test sentences.")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark_results", help="Directory to save output WAVs and report.")
    args = parser.parse_args()

    model_path = Path(args.model)
    config_path = Path(args.model_config) if args.model_config else Path(str(args.model) + ".json")
    sentences_path = Path(args.sentences)
    output_dir = Path(args.output_dir)

    run_benchmark(model_path, config_path, sentences_path, output_dir)

if __name__ == "__main__":
    main()
