#!/usr/bin/env python3
"""
Script: run_local_pipeline.py
Description: End-to-end local workstation execution pipeline for Arabic Piper TTS fine-tuning.
Supports local GPU training (e.g., RTX 5090) and background SSH daemon execution.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import yaml
from pathlib import Path

def setup_logging(log_file_path: Path = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True
    )

def handle_signal(sig, frame):
    logging.info(f"\n⚠️  Received signal {sig}. Interrupted cleanly by user or system.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def run_command_in_pipeline(cmd: list, cwd: str = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def ensure_piper_train_installed():
    """Checks if piper_train and compiling dependencies are installed. If not, installs/patches them."""
    # Always ensure setuptools (pkg_resources) is installed since PyTorch Lightning requires it
    logging.info("Ensuring setuptools (pkg_resources) is installed in the virtual environment...")
    res = run_command_in_pipeline([sys.executable, "-m", "pip", "install", "setuptools"])
    if res.returncode != 0:
        logging.error(f"Failed to install setuptools: {res.stderr}")
        sys.exit(res.returncode)

    try:
        import pkg_resources
        logging.info(f"✓ pkg_resources is available at: {pkg_resources.__file__}")
    except ImportError as e:
        logging.error(f"❌ Failed to import pkg_resources even after setuptools installation: {e}")
        sys.exit(1)

    try:
        import piper_train
        # Verify compiled Cython extension monotonic_align.core is importable
        from piper_train.vits.monotonic_align import core
        logging.info("✓ piper_train module and monotonic_align compiled extension are ready.")
        return
    except ImportError:
        logging.info("🔍 piper_train or compiled monotonic_align not found. Initiating automatic local installation...")

    project_root = Path(__file__).resolve().parent.parent
    piper_src = project_root / "piper_src"

    if not piper_src.exists():
        logging.info(f"Cloning rhasspy/piper repository into {piper_src}...")
        res = run_command_in_pipeline(["git", "clone", "https://github.com/rhasspy/piper.git", str(piper_src)])
        if res.returncode != 0:
            logging.error(f"Failed to clone piper: {res.stderr}")
            sys.exit(res.returncode)

    # Install in editable mode
    logging.info("Installing piper_train package in editable mode...")
    res = run_command_in_pipeline([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(piper_src / "src" / "python")])
    if res.returncode != 0:
        logging.error(f"Failed to install piper_train: {res.stderr}")
        sys.exit(res.returncode)

    # Patch 1: PyTorch 2.6 weights_only checkpoint unpickling
    main_py = piper_src / "src" / "python" / "piper_train" / "__main__.py"
    if main_py.exists():
        content = main_py.read_text(encoding="utf-8")
        patch = (
            "import pathlib, torch\n"
            "try:\n"
            "    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])\n"
            "except Exception:\n"
            "    pass\n\n"
        )
        if "add_safe_globals" not in content:
            main_py.write_text(patch + content, encoding="utf-8")
            logging.info("Applied PyTorch 2.6 unpickling patch to piper_train/__main__.py")

    # Patch 2: Single-speaker dataset collate assertion in dataset.py
    dataset_py = piper_src / "src" / "python" / "piper_train" / "vits" / "dataset.py"
    if dataset_py.exists():
        ds_content = dataset_py.read_text(encoding="utf-8")
        old_code = "if utt.speaker_id is not None:"
        new_code = "if self.is_multispeaker and (utt.speaker_id is not None):"
        if old_code in ds_content:
            ds_content = ds_content.replace(old_code, new_code)
            dataset_py.write_text(ds_content, encoding="utf-8")
            logging.info("Applied single-speaker patch to piper_train/vits/dataset.py")

    # Patch 3: Dynamic guard assertion in transforms.py for ONNX export
    transforms_py = piper_src / "src" / "python" / "piper_train" / "vits" / "transforms.py"
    if transforms_py.exists():
        tf_content = transforms_py.read_text(encoding="utf-8")
        old_assert = "assert (discriminant >= 0).all(), discriminant"
        new_assert = "discriminant = torch.clamp(discriminant, min=0)"
        if old_assert in tf_content:
            tf_content = tf_content.replace(old_assert, new_assert)
            transforms_py.write_text(tf_content, encoding="utf-8")
            logging.info("Applied spline clamp patch to piper_train/vits/transforms.py")

    # Patch 4: Force legacy TorchScript tracing (dynamo=False) in export_onnx.py
    export_py = piper_src / "src" / "python" / "piper_train" / "export_onnx.py"
    if export_py.exists():
        exp_content = export_py.read_text(encoding="utf-8")
        if "dynamo=" not in exp_content and "torch.onnx.export(" in exp_content:
            exp_content = exp_content.replace("torch.onnx.export(", "torch.onnx.export(dynamo=False, ")
            export_py.write_text(exp_content, encoding="utf-8")
            logging.info("Applied dynamo=False patch to piper_train/export_onnx.py")

    # Compile monotonic_align
    import sysconfig
    mono_dir = piper_src / "src" / "python" / "piper_train" / "vits" / "monotonic_align"
    out_subdir = mono_dir / "monotonic_align"
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Make sure cython and setuptools are installed in this environment
    logging.info("Ensuring cython>=3.0.0 and setuptools are installed...")
    install_res = run_command_in_pipeline([sys.executable, "-m", "pip", "install", "-q", "cython>=3.0.0", "setuptools"])
    if install_res.returncode != 0:
        logging.error(f"Failed to install compilation requirements (cython/setuptools): {install_res.stderr}")
        sys.exit(install_res.returncode)

    logging.info("Compiling cython monotonic_align module (generating Python 3.12 compatible core.c)...")
    res = run_command_in_pipeline([sys.executable, "-m", "cython", "-3", "core.pyx"], cwd=str(mono_dir))
    if res.returncode != 0:
        logging.error(f"Cython compilation failed: {res.stderr}")
        sys.exit(res.returncode)

    py_inc = sysconfig.get_path("include")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    out_so = str(out_subdir / f"core{suffix}")
    logging.info("Building monotonic_align C extension with GCC...")
    res = run_command_in_pipeline(["gcc", "-shared", "-fPIC", "-O2", f"-I{py_inc}", "core.c", "-o", out_so], cwd=str(mono_dir))
    if res.returncode != 0:
        logging.error(f"Failed to compile monotonic_align C extension: {res.stderr}")
        sys.exit(res.returncode)

    logging.info("✅ Automatic installation and setup of piper_train successful!")

def run_step(command: list, description: str):
    logging.info(f"\n==========================================")
    logging.info(f"🚀 [STEP] {description}")
    logging.info(f"Running: {' '.join(command)}")
    logging.info(f"==========================================\n")
    res = subprocess.run(command)
    if res.returncode != 0:
        logging.error(f"❌ Failed during step: {description}")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run complete local Arabic Piper fine-tuning pipeline.")
    parser.add_argument("--config", type=str, default="configs/experiment001.yaml", help="Path to YAML experiment config.")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory for storing data/models.")
    parser.add_argument("--epochs", type=int, default=None, help="Additional fine-tuning epochs to train (overrides config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU.")
    parser.add_argument("--precision", type=str, default="32", choices=["32", "16-mixed", "bf16-mixed"], help="PyTorch Lightning precision.")
    parser.add_argument("--devices", type=str, default="1", help="Number of GPUs or GPU IDs to use (e.g. 1 or '0,').")
    parser.add_argument("--log-file", type=str, default=None, help="Path to file for logging execution output.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download step if already present.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation step if already present.")
    args = parser.parse_args()

    if args.log_file:
        setup_logging(Path(args.log_file))
    else:
        setup_logging()

    # Ensure piper_train training package is installed, compiled, and patched
    ensure_piper_train_installed()

    cfg = load_config(args.config)
    data_root = Path(args.data_root)

    dataset_dir = data_root / cfg.get("paths", {}).get("datasets_dir", "datasets/experiment001")
    processed_dir = data_root / cfg.get("paths", {}).get("processed_dir", "processed/experiment001")
    checkpoints_dir = data_root / cfg.get("paths", {}).get("checkpoints_dir", "checkpoints/experiment001")
    base_ckpt_dir = data_root / "checkpoints" / "base"
    outputs_dir = data_root / cfg.get("paths", {}).get("outputs_dir", "outputs/experiment001")

    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 32)
    fine_tune_epochs = args.epochs or cfg.get("training", {}).get("epochs", 50)
    base_epoch = 5079
    target_max_epochs = base_epoch + fine_tune_epochs

    # Step 1: Download
    if not args.skip_download:
        run_step(
            [sys.executable, "scripts/download_dataset.py", "--config", args.config, "--data-root", str(data_root)],
            "Download Dataset & Base Checkpoint"
        )
    else:
        logging.info("Skipping dataset download step (--skip-download specified).")

    # Step 2: Prepare
    if not args.skip_prepare:
        run_step(
            [sys.executable, "scripts/prepare_dataset.py", "--config", args.config, "--data-root", str(data_root)],
            "Prepare Dataset (Phonemization & Spectrogram Caching)"
        )
    else:
        logging.info("Skipping dataset preparation step (--skip-prepare specified).")

    # Step 3: Checkpoint Detection & Target Epoch Calculation
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    existing_ckpts = sorted(checkpoints_dir.rglob("*.ckpt"))

    if existing_ckpts:
        resume_ckpt = str(existing_ckpts[-1])
        logging.info(f"✅ Resuming fine-tuning from latest checkpoint: '{resume_ckpt}'")
    else:
        candidates = list(base_ckpt_dir.rglob("*.ckpt")) if base_ckpt_dir.exists() else []
        if candidates:
            resume_ckpt = str(candidates[0])
            logging.info(f"🆕 Fine-tuning from base checkpoint: '{resume_ckpt}'")
        else:
            logging.error(f"❌ Base checkpoint not found in '{base_ckpt_dir}'!")
            sys.exit(1)

    logging.info(f"🎯 Target max_epochs: {target_max_epochs} (Base {base_epoch} + Fine-tune {fine_tune_epochs})")

    # Step 4: Fine-Tuning Execution
    train_cmd = [
        sys.executable, "-m", "piper_train",
        "--dataset-dir", str(processed_dir),
        "--accelerator", "gpu",
        "--devices", str(args.devices),
        "--batch-size", str(batch_size),
        "--validation-split", "0.05",
        "--max_epochs", str(target_max_epochs),
        "--precision", str(args.precision),
        "--checkpoint-epochs", str(cfg.get("training", {}).get("checkpoint_every_epochs", 5)),
        "--default_root_dir", str(checkpoints_dir),
        "--resume_from_checkpoint", resume_ckpt,
    ]
    run_step(train_cmd, "Execute Piper Fine-Tuning")

    # Step 5: Export to ONNX
    new_ckpts = sorted(checkpoints_dir.rglob("*.ckpt"))
    if not new_ckpts:
        logging.error("❌ No checkpoint found after training.")
        sys.exit(1)

    best_ckpt = str(new_ckpts[-1])
    onnx_out = outputs_dir / "ar_JO_finetuned.onnx"
    config_json = processed_dir / "config.json"

    export_cmd = [
        sys.executable, "scripts/export_model.py",
        "--checkpoint", best_ckpt,
        "--output-onnx", str(onnx_out),
        "--config-json", str(config_json),
    ]
    run_step(export_cmd, "Export Fine-Tuned Model to ONNX")

    # Step 6: Fine-Tuned Benchmark
    benchmark_out = data_root / "outputs" / "finetuned_benchmark"
    sentences_txt = Path("benchmark/benchmark_sentences.txt")

    if sentences_txt.exists() and onnx_out.exists():
        benchmark_cmd = [
            sys.executable, "scripts/benchmark.py",
            "--model", str(onnx_out),
            "--model-config", str(Path(str(onnx_out) + ".json")),
            "--sentences", str(sentences_txt),
            "--output-dir", str(benchmark_out),
        ]
        run_step(benchmark_cmd, "Benchmark Fine-Tuned Model")

    logging.info(f"\n==========================================")
    logging.info(f"✅ Local Fine-Tuning Pipeline Complete!")
    logging.info(f"Fine-tuned model output: {onnx_out}")
    logging.info(f"==========================================\n")

if __name__ == "__main__":
    main()
