#!/usr/bin/env python3
"""
Script: prepare_dataset.py
Description: Converts raw HuggingFace dataset into LJSpeech structure (wavs/ + metadata.csv),
resamples audio to target rate (22050 Hz), copies config.json from base model, and runs
Python-native phonemization + audio tensor caching to create dataset.jsonl for piper_train.
"""

import argparse
import io
import json
import logging
import os
import shutil
import sys
import numpy as np
import soundfile as sf
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

try:
    import torchaudio
except ImportError:
    torchaudio = None

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

    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = obj.item()

    if isinstance(obj, np.ndarray) and obj.ndim > 0:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

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

        if audio_data.get("bytes") is not None:
            try:
                bytes_data = audio_data["bytes"]
                array, orig_sr = sf.read(io.BytesIO(bytes_data))
            except Exception:
                if torchaudio is not None:
                    try:
                        waveform_tensor, orig_sr = torchaudio.load(io.BytesIO(bytes_data))
                        array = waveform_tensor.numpy()
                    except Exception:
                        array = None

        if array is None and audio_data.get("path") is not None:
            try:
                audio_path = audio_data["path"]
                if os.path.exists(audio_path):
                    array, orig_sr = sf.read(audio_path)
            except Exception:
                array = None

        if array is None and audio_data.get("array") is not None:
            array = extract_array_from_object(audio_data["array"])

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
        if torchaudio is not None:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            waveform = resampler(waveform)
        else:
            # Fallback simple resample via numpy / scipy if needed
            pass

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav_path, waveform.squeeze(0).numpy(), target_sr, subtype="PCM_16")

def copy_base_config_json(output_dir: Path, drive_root: Path):
    """Copies config.json downloaded from base checkpoint into output_dir."""
    config_target = output_dir / "config.json"
    if config_target.exists():
        logging.info(f"config.json already exists at '{config_target}'.")
        return True

    base_ckpt_dir = drive_root / "checkpoints" / "base"
    candidates = list(base_ckpt_dir.rglob("config.json")) if base_ckpt_dir.exists() else []

    if candidates:
        src = candidates[0]
        shutil.copy(src, config_target)
        logging.info(f"Copied base config.json: '{src}' -> '{config_target}'")
        return True

    logging.warning(f"No config.json found in '{base_ckpt_dir}'.")
    return False

def spectrogram_torch(y, n_fft=1024, hop_size=256, win_size=1024, center=False):
    """Calculates linear STFT spectrogram tensor for piper_train."""
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def run_piper_preprocess_python(output_dir: Path, language: str = "ar", sample_rate: int = 22050):
    """
    Python-native dataset preprocessor for piper_train.
    Bypasses legacy piper_phonemize CLI dependency by using python-piper + espeak-ng + PyTorch STFT.
    """
    logging.info("Running Python-native Piper preprocessor (phonemization + spectrogram caching)...")

    from piper.phonemize_espeak import EspeakPhonemizer
    from piper.phoneme_ids import phonemes_to_ids, DEFAULT_PHONEME_ID_MAP

    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    # Load phoneme_id_map from config.json if available
    config_path = output_dir / "config.json"
    id_map = DEFAULT_PHONEME_ID_MAP
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                if "phoneme_id_map" in cfg_data:
                    id_map = cfg_data["phoneme_id_map"]
                    logging.info(f"Loaded phoneme_id_map from {config_path}")
        except Exception as e:
            logging.warning(f"Could not load phoneme_id_map from {config_path}: {e}")

    cache_dir = output_dir / "cache" / str(sample_rate)
    cache_dir.mkdir(parents=True, exist_ok=True)

    phonemizer = EspeakPhonemizer()
    jsonl_path = output_dir / "dataset.jsonl"
    wavs_dir = output_dir / "wavs"

    valid_count = 0
    with open(metadata_path, "r", encoding="utf-8") as f_in, \
         open(jsonl_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Processing & caching dataset"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue

            sample_id = parts[0]
            speaker = parts[1] if len(parts) > 2 else "speaker1"
            text = parts[-1]

            wav_file = wavs_dir / f"{sample_id}.wav"
            if not wav_file.exists():
                wav_file = output_dir / sample_id
            if not wav_file.exists():
                logging.warning(f"Audio file missing for {sample_id}")
                continue

            try:
                # 1. Phonemize
                all_phonemes = phonemizer.phonemize(language, text)
                flat_phonemes = [p for sent in all_phonemes for p in sent]
                p_ids = phonemes_to_ids(flat_phonemes, id_map)

                # 2. Audio & Spectrogram caching
                audio_data, sr = sf.read(str(wav_file))
                audio_tensor = torch.FloatTensor(audio_data)
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.ndim > 1:
                    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

                norm_pt_path = cache_dir / f"{sample_id}.pt"
                spec_pt_path = cache_dir / f"{sample_id}.spec.pt"

                torch.save(audio_tensor, norm_pt_path)
                spec_tensor = spectrogram_torch(audio_tensor).squeeze(0)
                torch.save(spec_tensor, spec_pt_path)

                # 3. JSONL record
                utt_dict = {
                    "text": text,
                    "audio_path": str(wav_file),
                    "speaker": speaker,
                    "speaker_id": 0,
                    "phonemes": flat_phonemes,
                    "phoneme_ids": p_ids,
                    "audio_norm_path": str(norm_pt_path),
                    "audio_spec_path": str(spec_pt_path),
                }

                f_out.write(json.dumps(utt_dict, ensure_ascii=False) + "\n")
                valid_count += 1

            except Exception as e:
                logging.warning(f"Failed preprocessing {sample_id}: {e}")

    logging.info(f"✅ Python preprocessor complete: {valid_count} entries in '{jsonl_path}'")

def prepare_dataset(
    dataset_dir: Path,
    output_dir: Path,
    drive_root: Path,
    target_sr: int = 22050,
    train_ratio: float = 0.95,
    seed: int = 42,
    language: str = "ar",
):
    """Processes dataset into LJSpeech format and runs Python preprocessor."""
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
            metadata_entries.append(f"{sample_id}|speaker1|{text}")
        except Exception as e:
            logging.warning(f"Skipping sample {idx} due to error: {e}")

    if not metadata_entries:
        raise RuntimeError("No valid audio samples were processed.")

    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(f"{entry}\n")
    logging.info(f"Metadata saved to '{metadata_csv_path}' with {len(metadata_entries)} valid entries.")

    np.random.seed(seed)
    indices = np.arange(len(metadata_entries))
    np.random.shuffle(indices)

    split_point = int(len(metadata_entries) * train_ratio)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    with open(output_dir / "train.csv", "w", encoding="utf-8") as f:
        for idx in train_indices:
            f.write(f"{metadata_entries[idx]}\n")

    with open(output_dir / "val.csv", "w", encoding="utf-8") as f:
        for idx in val_indices:
            f.write(f"{metadata_entries[idx]}\n")

    logging.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val samples.")

    # Copy config.json from base checkpoint
    copy_base_config_json(output_dir, drive_root)

    # Run Python-native preprocessing (creates dataset.jsonl & cached .pt tensors)
    run_piper_preprocess_python(output_dir, language=language, sample_rate=target_sr)

    logging.info(f"\n✅ Dataset preparation complete: '{output_dir}'")

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
