#!/usr/bin/env python3
"""
Script: prepare_dataset.py
Description: Converts raw dataset into LJSpeech structure (wavs/ + metadata.csv),
resamples audio to target rate (22050 Hz), and splits into train/validation sets.
"""

import argparse
import io
import logging
import os
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

    if hasattr(obj, "to_numpy"):
        return obj.to_numpy()
    if hasattr(obj, "get_array"):
        return obj.get_array()
    if hasattr(obj, "decode"):
        return obj.decode()
    if hasattr(obj, "array"):
        return extract_array_from_object(obj.array)

    try:
        # Try indexing or slicing if object supports it
        return np.asarray(obj[:])
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
        
        # Priority 1: Try reading raw bytes if available (avoids AudioDecoder issues)
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

        # Priority 3: Try raw array / AudioDecoder conversion
        if array is None and audio_data.get("array") is not None:
            array = extract_array_from_object(audio_data["array"])

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

def prepare_dataset(dataset_dir: Path, output_dir: Path, target_sr: int = 22050, train_ratio: float = 0.95, seed: int = 42):
    """Processes dataset files into wavs/ directory and metadata.csv."""
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
            # Metadata format for Piper: filename|speaker|text
            metadata_entries.append(f"{sample_id}|speaker1|{text}")
        except Exception as e:
            logging.warning(f"Skipping sample {idx} due to error: {e}")

    # Write metadata.csv
    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(f"{entry}\n")
    logging.info(f"Metadata saved to '{metadata_csv_path}' with {len(metadata_entries)} valid entries.")

    # Train / Val Split
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

    logging.info(f"Dataset split complete: {len(train_indices)} train samples, {len(val_indices)} val samples.")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset into LJSpeech format for Piper training.")
    parser.add_argument("--config", type=str, default="configs/experiment001.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Input raw dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output processed dataset directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    drive_root = cfg.get("paths", {}).get("drive_root", ".")
    
    dataset_dir = Path(args.dataset_dir or (Path(drive_root) / cfg.get("paths", {}).get("datasets_dir", "datasets")))
    output_dir = Path(args.output_dir or (Path(drive_root) / cfg.get("paths", {}).get("processed_dir", "processed")))
    target_sr = cfg.get("dataset", {}).get("target_sample_rate", 22050)
    train_ratio = cfg.get("dataset", {}).get("train_val_split", 0.95)
    seed = cfg.get("dataset", {}).get("seed", 42)

    prepare_dataset(dataset_dir, output_dir, target_sr=target_sr, train_ratio=train_ratio, seed=seed)

if __name__ == "__main__":
    main()
