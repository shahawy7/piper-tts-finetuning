#!/usr/bin/env python3
"""
Script: test_local.py
Description: CLI tool to test Arabic Piper TTS models locally on CPU before and after training,
with voice speed control via length_scale parameter.
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
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

def play_sound_local(wav_path: Path):
    """Attempts to play WAV audio file on Linux, macOS, or Windows."""
    sys_name = platform.system()
    
    players = []
    if sys_name == "Linux":
        players = [["aplay", str(wav_path)], ["paplay", str(wav_path)], ["ffplay", "-nodisp", "-autoexit", str(wav_path)]]
    elif sys_name == "Darwin":
        players = [["afplay", str(wav_path)]]
    elif sys_name == "Windows":
        players = [["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"]]

    for player in players:
        if shutil.which(player[0]):
            try:
                logging.info(f"Playing audio using '{player[0]}'...")
                subprocess.run(player, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                logging.debug(f"Audio player '{player[0]}' failed: {e}")
                continue
    logging.warning("No audio player binary found to play sound automatically. You can open the generated WAV manually.")
    return False

def synthesize_cpu(model_path: Path, config_path: Path, text: str, output_wav: Path,
                   length_scale: float = 1.0, play_audio: bool = True) -> dict:
    """Synthesizes text on local CPU using piper binary or Python piper package, with speed control."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    # Approach 1: System piper CLI binary if present
    piper_exe = shutil.which("piper")
    if piper_exe:
        cmd = [
            piper_exe,
            "--model", str(model_path),
            "--output_file", str(output_wav),
            "--length_scale", str(length_scale)
        ]
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
                
                if play_audio:
                    play_sound_local(output_wav)
                    
                return {
                    "status": "success",
                    "method": "piper_binary",
                    "length_scale": length_scale,
                    "elapsed_sec": round(elapsed, 4),
                    "duration_sec": round(duration, 4),
                    "rtf": round(rtf, 4),
                    "output_wav": str(output_wav)
                }
        except Exception as e:
            logging.debug(f"System piper CLI failed: {e}")

    # Approach 2: Python piper package (PiperVoice)
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        json_file_arg = str(config_path) if (config_path and config_path.exists()) else None
        voice = PiperVoice.load(str(model_path), config_path=json_file_arg)

        syn_cfg = SynthesisConfig(length_scale=length_scale)
        chunks = list(voice.synthesize(text, syn_config=syn_cfg))
        
        if chunks:
            with wave.open(str(output_wav), "wb") as wav_file:
                chunk = chunks[0]
                wav_file.setnchannels(chunk.sample_channels)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setframerate(chunk.sample_rate)
                for c in chunks:
                    wav_file.writeframes(c.audio_int16_bytes)

            elapsed = time.perf_counter() - start_time
            import soundfile as sf
            data, sr = sf.read(output_wav)
            duration = len(data) / float(sr)
            rtf = elapsed / duration if duration > 0 else 0.0

            if play_audio:
                play_sound_local(output_wav)

            return {
                "status": "success",
                "method": "python_piper",
                "length_scale": length_scale,
                "elapsed_sec": round(elapsed, 4),
                "duration_sec": round(duration, 4),
                "rtf": round(rtf, 4),
                "output_wav": str(output_wav)
            }
    except Exception as e:
        logging.error(f"Python Piper synthesis error: {e}")

    return {"status": "error", "error": "Failed to synthesize audio. Install piper-tts package via `pip install piper-tts`."}

def run_interactive(model_path: Path, config_path: Path, output_dir: Path,
                    length_scale: float = 1.0, play_audio: bool = True):
    """Interactive CLI terminal for typing Arabic text and testing CPU synthesis."""
    logging.info(f"=== Starting Interactive Local CPU Testing (Speed / Length Scale: {length_scale}) ===")
    logging.info(f"Model: {model_path}")
    logging.info("Type Arabic diacritized text (or 'exit' to quit):\n")
    
    count = 1
    while True:
        try:
            text = input("Arabic Input ➔ ").strip()
            if not text or text.lower() in ["exit", "quit", "q"]:
                break
            
            output_file = output_dir / f"interactive_{count:03d}.wav"
            res = synthesize_cpu(model_path, config_path, text, output_file, length_scale=length_scale, play_audio=play_audio)
            
            if res.get("status") == "success":
                logging.info(f"✓ Audio generated in {res['elapsed_sec']}s (Duration: {res['duration_sec']}s | RTF: {res['rtf']} | Speed Scale: {length_scale})")
                logging.info(f"Saved to: '{output_file}'\n")
            else:
                logging.error(f"Synthesis failed: {res}\n")
            count += 1
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description="Test Piper Arabic model locally on CPU with voice speed control.")
    parser.add_argument("--mode", choices=["baseline", "finetuned", "compare"], default="baseline",
                        help="Testing mode: 'baseline' (before training), 'finetuned' (after training), or 'compare' (both).")
    parser.add_argument("--model", type=str, default=None, help="Path to custom .onnx model file.")
    parser.add_argument("--config-json", type=str, default=None, help="Path to custom .onnx.json model config.")
    parser.add_argument("--text", type=str, default=None, help="Arabic text string to synthesize.")
    parser.add_argument("--length-scale", "--length_scale", type=float, default=1.0,
                        help="Phoneme length scale for voice speed control (< 1.0 faster, > 1.0 slower, default: 1.0).")
    parser.add_argument("--sentences-file", type=str, default="benchmark/benchmark_sentences.txt", help="File with benchmark test sentences.")
    parser.add_argument("--output-dir", type=str, default="outputs/local_test", help="Directory to save generated WAV files.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal prompt.")
    parser.add_argument("--no-play", action="store_true", help="Disable automatic audio playback.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    play_audio = not args.no_play
    length_scale = args.length_scale

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
        logging.info(f"=== Comparing Baseline vs Fine-Tuned on CPU (Length Scale: {length_scale}) ===")
        res_base = synthesize_cpu(base_onnx, base_json, sample_text, output_dir / "baseline_sample.wav", length_scale=length_scale, play_audio=play_audio)
        logging.info(f"Baseline result: {res_base}")
        
        if ft_onnx.exists():
            res_ft = synthesize_cpu(ft_onnx, ft_json, sample_text, output_dir / "finetuned_sample.wav", length_scale=length_scale, play_audio=play_audio)
            logging.info(f"Fine-tuned result: {res_ft}")
        else:
            logging.warning(f"Fine-tuned model '{ft_onnx}' not found. Run training on Colab first!")
        return

    if args.interactive:
        run_interactive(model_path, config_path, output_dir, length_scale=length_scale, play_audio=play_audio)
        return

    sample_text = args.text or "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ."
    output_wav = output_dir / f"{args.mode}_test.wav"
    
    logging.info(f"Testing {label} model on CPU (Length Scale: {length_scale})...")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Input Text: '{sample_text}'")
    
    result = synthesize_cpu(model_path, config_path, sample_text, output_wav, length_scale=length_scale, play_audio=play_audio)
    if result.get("status") == "success":
        logging.info(f"✓ Success! Generated WAV: '{result['output_wav']}' (Duration: {result['duration_sec']}s | RTF: {result['rtf']} | Length Scale: {length_scale})")
    else:
        logging.error(f"Test failed: {result}")

if __name__ == "__main__":
    main()
