#!/usr/bin/env python3
"""
Script: download_dataset.py
Description: Downloads the Arabic dataset from Hugging Face and base Piper checkpoint.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

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

def download_dataset(dataset_repo: str, output_dir: Path):
    """Downloads Hugging Face dataset to output directory."""
    from datasets import load_dataset
    
    logging.info(f"Downloading dataset '{dataset_repo}' into '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_repo)
    dataset.save_to_disk(str(output_dir))
    logging.info(f"Dataset successfully saved to '{output_dir}'.")
    return dataset

def download_base_checkpoint(checkpoint_repo: str, filename: str, output_dir: Path):
    """Downloads base pretrained model checkpoint and config from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the .ckpt file
    logging.info(f"Downloading base checkpoint '{filename}' from repo '{checkpoint_repo}'...")
    ckpt_path = hf_hub_download(
        repo_id=checkpoint_repo,
        filename=filename,
        local_dir=str(output_dir),
        repo_type="dataset"
    )
    logging.info(f"Base checkpoint saved at: '{ckpt_path}'")
    
    # Also download config.json from the same directory
    config_filename = str(Path(filename).parent / "config.json")
    try:
        logging.info(f"Downloading training config '{config_filename}'...")
        config_path = hf_hub_download(
            repo_id=checkpoint_repo,
            filename=config_filename,
            local_dir=str(output_dir),
            repo_type="dataset"
        )
        logging.info(f"Training config saved at: '{config_path}'")
    except Exception as e:
        logging.warning(f"Could not download training config: {e}")
    
    return ckpt_path

def main():
    parser = argparse.ArgumentParser(description="Download dataset & base model for Arabic Piper fine-tuning.")
    parser.add_argument("--config", type=str, default="configs/experiment001.yaml", help="Path to experiment config YAML.")
    parser.add_argument("--dataset-repo", type=str, default=None, help="HF dataset repository ID.")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for storing data.")
    parser.add_argument("--drive-root", type=str, default=None, help="Legacy alias for --data-root.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    data_root = args.data_root or args.drive_root or cfg.get("paths", {}).get("data_root") or cfg.get("paths", {}).get("drive_root", ".")
    datasets_subdir = cfg.get("paths", {}).get("datasets_dir", "datasets")
    dataset_repo = args.dataset_repo or cfg.get("dataset", {}).get("hf_repo", "NightPrince/Arabic-professional-voice")
    
    checkpoint_repo = cfg.get("model", {}).get("base_checkpoint_hf_repo", "rhasspy/piper-checkpoints")
    checkpoint_filename = cfg.get("model", {}).get("base_checkpoint_filename", "ar/ar_JO/kareem/medium/epoch=5079-step=1682020.ckpt")
    
    target_dataset_dir = Path(data_root) / datasets_subdir
    target_ckpt_dir = Path(data_root) / "checkpoints" / "base"
    
    # Run downloads
    try:
        download_dataset(dataset_repo, target_dataset_dir)
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        logging.info("Continuing download check...")

    try:
        download_base_checkpoint(checkpoint_repo, checkpoint_filename, target_ckpt_dir)
    except Exception as e:
        logging.error(f"Error downloading checkpoint: {e}")

if __name__ == "__main__":
    main()
