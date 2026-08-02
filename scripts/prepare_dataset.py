#!/usr/bin/env python3
"""
Script: prepare_dataset.py
Description: Converts raw HuggingFace dataset into LJSpeech structure (wavs/ + metadata.csv),
resamples audio to target rate (22050 Hz), then runs piper_train.preprocess to generate
the config.json and dataset.jsonl required by piper_train.
"""

import argparse
import io
import logging
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def extract_array_from_object(obj):
    """Recursively extracts a 1D/2D numpy array from raw_array / AudioDecoder objects."""
    if obj is None:
        return None

    # Handle 0D numpy scalar wrapping an object
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = obj.item()

    if isinstance(obj, np.ndarray) and obj.ndim > 0:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    # torchcodec.decoders.AudioDecoder / datasets._torchcodec.AudioDecoder
    if hasattr(obj, "get_all_samples"):
        try:
            samples = obj.get_all_samples()
            if hasattr(samples, "data"):
                data_tensor = samples.data
                if isinstance(data_tensor, torch.Tensor):
                    return data_tensor.detach().cpu().numpy()
                return np.asarray(data_tensor)
        except Exception as e:
            logging.debug(f"get_all_samples failed: {e}")

    if hasattr(obj, "to_numpy"):
        try:
            return obj.to_numpy()
        except Exception:
            pass

    if hasattr(obj, "get_array"):
        try:
            return obj.get_array()
        except Exception:
            pass

    if hasattr(obj, "decode"):
        try:
            return obj.decode()
        except Exception:
            pass

    if hasattr(obj, "array"):
        return extract_array_from_object(obj.array)

    try:
        res = np.asarray(obj[:])
        if res.ndim > 0:
            return res
    except Exception:
        pass

    try:
        arr = np.asarray(obj)
        if arr.ndim > 0:
            return arr
    except Exception:
        pass

    return None

def process_and_save_audio(audio_data, target_sr: int, output_wav_path: Path):
    """Resamples audio tensor/array to target sample rate and saves as mono 16-bit WAV."""
    array = None
    orig_sr = target_sr

    if isinstance(audio_data, dict):
        orig_sr = audio_data.get("sampling_rate") or target_sr

        # Priority 1: Try reading raw bytes if available
        if audio_data.get("bytes") is not None:
            try:
                bytes_data = audio_data["bytes"]
                array, orig_sr = sf.read(io.BytesIO(bytes_data))
            except Exception:
                try:
                    waveform_tensor, orig_sr = torchaudio.load(io.BytesIO(bytes_data))
                    array = waveform_tensor.numpy()
                except Exception:
                    array = None

        # Priority 2: Try reading from file path if available
        if array is None and audio_data.get("path") is not None:
            try:
                audio_path = audio_data["path"]
                if os.path.exists(audio_path):
                    array, orig_sr = sf.read(audio_path)
            except Exception:
                array = None

        # Priority 3: Try raw array / AudioDecoder conversion from 'array' key
        if array is None and audio_data.get("array") is not None:
            array = extract_array_from_object(audio_data["array"])

        # Priority 4: Try extracting from any object in dict
        if array is None:
            for val in audio_data.values():
                res = extract_array_from_object(val)
                if res is not None:
                    array = res
                    break

    else:
        array = extract_array_from_object(audio_data)

    if array is None or not hasattr(array, "__len__") or len(array) == 0:
        raise ValueError(f"Could not extract valid audio array from audio data structure: {type(audio_data)}")

    waveform = torch.tensor(array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Save audio file
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav_path, waveform.squeeze(0).numpy(), target_sr, subtype="PCM_16")

def copy_base_config_json(output_dir: Path, drive_root: Path):
    """
    Finds the config.json downloaded from the base checkpoint and copies it
    into the processed dataset directory so piper_train can find it.
    """
    config_target = output_dir / "config.json"
    if config_target.exists():
        logging.info(f"config.json already exists at '{config_target}'.")
        return True

    # Search recursively under checkpoints/base for config.json
    base_ckpt_dir = drive_root / "checkpoints" / "base"
    candidates = list(base_ckpt_dir.rglob("config.json")) if base_ckpt_dir.exists() else []

    if candidates:
        src = candidates[0]
        shutil.copy(src, config_target)
        logging.info(f"Copied base config.json: '{src}' -> '{config_target}'")
        return True

    logging.warning(
        f"No config.json found in '{base_ckpt_dir}'. "
        "piper_train.preprocess will generate one from scratch using espeak-ng phonemization."
    )
    return False

def run_piper_preprocess(output_dir: Path, language: str = "ar", sample_rate: int = 22050):
    """
    Runs piper_train.preprocess to:
      1. Phonemize text from metadata.csv using espeak-ng
      2. Generate dataset.jsonl (training data used by piper_train)
      3. Generate config.json if not already present

    The output_dir must contain:
      - wavs/       (audio files)
      - metadata.csv (LJSpeech format: filename|speaker|text)
    """
    logging.info("Running piper_train.preprocess to generate dataset.jsonl and config.json...")

    cmd = [
        sys.executable, "-m", "piper_train.preprocess",
        "--language", language,
        "--input-dir", str(output_dir),
        "--output-dir", str(output_dir),
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", str(sample_rate),
    ]

    logging.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"piper_train.preprocess failed (exit code {result.returncode}).\n"
            "Make sure piper_train is installed: pip install --no-deps -e /content/piper/src/python"
        )

    # Verify output
    jsonl_path = output_dir / "dataset.jsonl"
    if jsonl_path.exists():
        count = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
        logging.info(f"piper_train.preprocess complete: {count} phonemized entries in dataset.jsonl")
    else:
        logging.warning("dataset.jsonl was not created. Check preprocess output above.")

def prepare_dataset(
    dataset_dir: Path,
    output_dir: Path,
    drive_root: Path,
    target_sr: int = 22050,
    train_ratio: float = 0.95,
    seed: int = 42,
    language: str = "ar",
):
    """Processes dataset files into LJSpeech structure, then runs piper_train.preprocess."""
    from datasets import Audio, load_from_disk

    logging.info(f"Loading dataset from: '{dataset_dir}'")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path {dataset_dir} does not exist. Run download_dataset.py first.")

    dataset = load_from_disk(str(dataset_dir))
    if hasattr(dataset, "keys"):
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
    else:
        data = dataset

    # Force HF datasets to decode audio column into standard numpy arrays if possible
    try:
        if hasattr(data, "cast_column") and "audio" in data.column_names:
            logging.info("Casting audio column with datasets.Audio(decode=True)...")
            data = data.cast_column("audio", Audio(decode=True))
    except Exception as e:
        logging.warning(f"Audio column casting skipped: {e}")

    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_entries = []
    logging.info(f"Processing {len(data)} samples...")

    for idx, item in enumerate(tqdm(data, desc="Processing audio samples")):
        sample_id = f"ar_sample_{idx:05d}"
        wav_path = wavs_dir / f"{sample_id}.wav"

        text = item.get("text", item.get("sentence", item.get("transcription", ""))).strip()
        audio = item.get("audio")

        if not text or audio is None:
            continue

        try:
            process_and_save_audio(audio, target_sr, wav_path)
            # LJSpeech metadata format for Piper: filename|speaker|text
            metadata_entries.append(f"{sample_id}|speaker1|{text}")
        except Exception as e:
            logging.warning(f"Skipping sample {idx} due to error: {e}")

    if not metadata_entries:
        raise RuntimeError("No valid audio samples were processed. Check dataset format and audio extraction.")

    # Write metadata.csv (LJSpeech format)
    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(f"{entry}\n")
    logging.info(f"Metadata saved to '{metadata_csv_path}' with {len(metadata_entries)} valid entries.")

    # Train / Val split (used for manual verification via Cell 2.3)
    np.random.seed(seed)
    indices = np.arange(len(metadata_entries))
    np.random.shuffle(indices)

    split_point = int(len(metadata_entries) * train_ratio)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_file = output_dir / "train.csv"
    val_file = output_dir / "val.csv"

    with open(train_file, "w", encoding="utf-8") as f:
        for idx in train_indices:
            f.write(f"{metadata_entries[idx]}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for idx in val_indices:
            f.write(f"{metadata_entries[idx]}\n")

    logging.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val samples.")

    # ── Step 2: Copy base config.json into output_dir ──────────────────────
    config_copied = copy_base_config_json(output_dir, drive_root)

    # ── Step 3: Run piper_train.preprocess to generate dataset.jsonl ───────
    # If config.json was copied from base, preprocess will reuse it (fine-tuning path).
    # If not found, preprocess generates a fresh config.json from scratch.
    run_piper_preprocess(output_dir, language=language, sample_rate=target_sr)

    logging.info(f"\n✅ Dataset preparation complete. Output directory: '{output_dir}'")
    logging.info(f"   Files created: wavs/, metadata.csv, dataset.jsonl, config.json")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset into LJSpeech + piper_train format.")
    parser.add_argument("--config", type=str, default="configs/experiment001.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Input raw HF dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output processed dataset directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    drive_root = Path(cfg.get("paths", {}).get("drive_root", "."))

    dataset_dir = Path(args.dataset_dir or (drive_root / cfg.get("paths", {}).get("datasets_dir", "datasets")))
    output_dir  = Path(args.output_dir  or (drive_root / cfg.get("paths", {}).get("processed_dir",  "processed")))

    target_sr   = cfg.get("dataset", {}).get("target_sample_rate", 22050)
    train_ratio = cfg.get("dataset", {}).get("train_val_split", 0.95)
    seed        = cfg.get("dataset", {}).get("seed", 42)
    language    = cfg.get("model",   {}).get("phoneme_language", "ar")

    prepare_dataset(
        dataset_dir,
        output_dir,
        drive_root=drive_root,
        target_sr=target_sr,
        train_ratio=train_ratio,
        seed=seed,
        language=language,
    )

if __name__ == "__main__":
    main()
