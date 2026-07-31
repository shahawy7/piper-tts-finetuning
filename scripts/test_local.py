#!/usr/bin/env python3
"""
Script: test_local.py
Description: CLI tool to test Arabic Piper TTS models locally on CPU before and after training.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASELINE_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx"
BASELINE_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json"

DEFAULT_BASELINE_DIR = Path("checkpoints/base")

def download_file(url: str, target_path: Path):
    """Downloads a remote file if not already present locally."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        logging.info(f"Downloading '{target_path.name}' from {url}...")
        urllib.request.urlretrieve(url, target_path)
        logging.info(f"Successfully downloaded to '{target_path}'.")

def ensure_baseline_model(base_dir: Path = DEFAULT_BASELINE_DIR) -> tuple[Path, Path]:
    """Ensures baseline ar_JO-kareem-medium ONNX model & config are downloaded locally."""
    onnx_path = base_dir / "ar_JO-kareem-medium.onnx"
    json_path = base_dir / "ar_JO-kareem-medium.onnx.json"
    
    download_file(BASELINE_MODEL_URL, onnx_path)
    download_file(BASELINE_CONFIG_URL, json_path)
    
    return onnx_path, json_path

def synthesize_cpu(model_path: Path, config_path: Path, text: str, output_wav: Path) -> dict:
    """Synthesizes text on local CPU using piper executable or Python fallback."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.perf_counter()
    
    # Try calling piper binary if installed
    cmd = ["piper", "--model", str(model_path), "--output_file", str(output_wav)]
    if config_path and config_path.exists():
        cmd.extend(["--config", str(config_path)])

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(input=text)
        elapsed = time.perf_counter() - start_time

        if proc.returncode == 0 and output_wav.exists():
            import soundfile as sf
            data, sr = sf.read(output_wav)
            duration = len(data) / float(sr)
            rtf = elapsed / duration if duration > 0 else 0.0
            return {
                "status": "success",
                "elapsed_sec": round(elapsed, 4),
                "duration_sec": round(duration, 4),
                "rtf": round(rtf, 4),
                "output_wav": str(output_wav)
            }
    except FileNotFoundError:
        pass

    # Python fallback mode using onnxruntime
    logging.info("Piper binary not found in PATH. Running python onnxruntime CPU fallback...")
    try:
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_path), session_options, providers=['CPUExecutionProvider'])
        
        elapsed = time.perf_counter() - start_time
        logging.info(f"ONNX model loaded successfully on CPU execution provider.")
        return {
            "status": "onnxruntime_ready",
            "elapsed_sec": round(elapsed, 4),
            "output_wav": str(output_wav)
        }
    except Exception as e:
        logging.error(f"CPU Synthesis error: {e}")
        return {"status": "error", "error": str(e)}

def run_interactive(model_path: Path, config_path: Path, output_dir: Path):
    """Interactive CLI terminal for typing Arabic text and testing CPU synthesis."""
    logging.info(f"=== Starting Interactive Local CPU Testing ===")
    logging.info(f"Model: {model_path}")
    logging.info("Type Arabic diacritized text (or 'exit' to quit):\n")
    
    count = 1
    while True:
        try:
            text = input("Arabic Input ➔ ").strip()
            if not text or text.lower() in ["exit", "quit", "q"]:
                break
            
            output_file = output_dir / f"interactive_{count:03d}.wav"
            res = synthesize_cpu(model_path, config_path, text, output_file)
            
            if res.get("status") == "success":
                logging.info(f"✓ Audio generated in {res['elapsed_sec']}s (Audio Duration: {res['duration_sec']}s | RTF: {res['rtf']})")
                logging.info(f"Saved to: '{output_file}'\n")
            else:
                logging.info(f"Result: {res}\n")
            count += 1
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description="Test Piper Arabic model locally on CPU before and after training.")
    parser.add_argument("--mode", choices=["baseline", "finetuned", "compare"], default="baseline",
                        help="Testing mode: 'baseline' (before training), 'finetuned' (after training), or 'compare' (both).")
    parser.add_argument("--model", type=str, default=None, help="Path to custom .onnx model file.")
    parser.add_argument("--config-json", type=str, default=None, help="Path to custom .onnx.json model config.")
    parser.add_argument("--text", type=str, default=None, help="Arabic text string to synthesize.")
    parser.add_argument("--sentences-file", type=str, default="benchmark/benchmark_sentences.txt", help="File with benchmark test sentences.")
    parser.add_argument("--output-dir", type=str, default="outputs/local_test", help="Directory to save generated WAV files.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal prompt.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        model_path, config_path = ensure_baseline_model()
        label = "BASELINE (Pre-Training)"
    elif args.mode == "finetuned":
        model_path = Path(args.model) if args.model else Path("outputs/experiment001/ar_JO_finetuned.onnx")
        config_path = Path(args.config_json) if args.config_json else Path(str(model_path) + ".json")
        label = "FINE-TUNED (Post-Training)"
    else:  # compare mode
        base_onnx, base_json = ensure_baseline_model()
        ft_onnx = Path(args.model) if args.model else Path("outputs/experiment001/ar_JO_finetuned.onnx")
        ft_json = Path(args.config_json) if args.config_json else Path(str(ft_onnx) + ".json")
        
        sample_text = args.text or "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ."
        logging.info("=== Comparing Baseline vs Fine-Tuned on CPU ===")
        res_base = synthesize_cpu(base_onnx, base_json, sample_text, output_dir / "baseline_sample.wav")
        logging.info(f"Baseline result: {res_base}")
        
        if ft_onnx.exists():
            res_ft = synthesize_cpu(ft_onnx, ft_json, sample_text, output_dir / "finetuned_sample.wav")
            logging.info(f"Fine-tuned result: {res_ft}")
        else:
            logging.warning(f"Fine-tuned model '{ft_onnx}' not found. Run training on Colab first!")
        return

    if args.interactive:
        run_interactive(model_path, config_path, output_dir)
        return

    sample_text = args.text or "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ."
    output_wav = output_dir / f"{args.mode}_test.wav"
    
    logging.info(f"Testing {label} model on CPU...")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Input Text: '{sample_text}'")
    
    result = synthesize_cpu(model_path, config_path, sample_text, output_wav)
    logging.info(f"Test complete: {result}")

if __name__ == "__main__":
    main()
