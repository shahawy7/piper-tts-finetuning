#!/usr/bin/env python3
"""
Script: evaluate.py
Description: Evaluates metrics (WER/CER, RTF, audio length) comparing baseline vs fine-tuned models.
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd
import yaml
from pathlib import Path
from jiwer import wer, cer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def evaluate_transcriptions(reference_texts: list, hypothesis_texts: list) -> dict:
    """Calculates Word Error Rate (WER) and Character Error Rate (CER)."""
    if not reference_texts or not hypothesis_texts or len(reference_texts) != len(hypothesis_texts):
        return {"wer": None, "cer": None}

    calculated_wer = wer(reference_texts, hypothesis_texts)
    calculated_cer = cer(reference_texts, hypothesis_texts)

    return {
        "wer": round(float(calculated_wer), 4),
        "cer": round(float(calculated_cer), 4)
    }

def compare_benchmark_reports(baseline_report_path: Path, finetuned_report_path: Path, output_csv: Path):
    """Compares baseline and fine-tuned benchmark reports."""
    with open(baseline_report_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    with open(finetuned_report_path, "r", encoding="utf-8") as f:
        ft_data = json.load(f)

    comparison = [
        {
            "Metric": "Total Audio Duration (sec)",
            "Baseline": base_data.get("total_audio_duration_sec"),
            "Fine-Tuned": ft_data.get("total_audio_duration_sec"),
        },
        {
            "Metric": "Total Synthesis Time (sec)",
            "Baseline": base_data.get("total_synth_time_sec"),
            "Fine-Tuned": ft_data.get("total_synth_time_sec"),
        },
        {
            "Metric": "Average Real-Time Factor (RTF)",
            "Baseline": base_data.get("avg_rtf"),
            "Fine-Tuned": ft_data.get("avg_rtf"),
        },
        {
            "Metric": "Overall Real-Time Factor (RTF)",
            "Baseline": base_data.get("overall_rtf"),
            "Fine-Tuned": ft_data.get("overall_rtf"),
        }
    ]

    df = pd.DataFrame(comparison)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    logging.info("\n=== Benchmark Comparison ===")
    logging.info(f"\n{df.to_string(index=False)}")
    logging.info(f"\nComparison matrix saved to: '{output_csv}'")
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned benchmark reports.")
    parser.add_argument("--baseline-report", type=str, required=True, help="Path to baseline benchmark_report.json.")
    parser.add_argument("--finetuned-report", type=str, required=True, help="Path to fine-tuned benchmark_report.json.")
    parser.add_argument("--output-csv", type=str, default="metrics/experiment_comparison.csv", help="Path for output metrics CSV.")
    args = parser.parse_args()

    compare_benchmark_reports(Path(args.baseline_report), Path(args.finetuned_report), Path(args.output_csv))

if __name__ == "__main__":
    main()
