# Experiment Log & Performance Tracker

This matrix logs all fine-tuning runs, tracking hyperparameters, loss metrics, synthesis speeds, and audio evaluation scores.

---

## 📊 Experiment Tracking Matrix

| Exp ID | Date | Base Model | Dataset | Epochs | Final Val Loss | RTF (avg) | Diacritic Score (1-5) | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 2026-07-31 | `ar_JO-kareem-medium` | N/A (Pretrained) | 0 | N/A | TBD | TBD | Completed | Pretrained baseline reference |
| `exp001` | 2026-07-31 | `ar_JO-kareem-medium` | `Arabic-professional-voice` | 50 | TBD | TBD | TBD | Configured | Initial fine-tuning run |

---

## 🧪 Detailed Log - Experiment 001

- **Experiment Name**: `experiment001`
- **Objective**: Fine-tune baseline Jordanian model on high-fidelity diacritized Modern Standard Arabic dataset.
- **Dataset**: `NightPrince/Arabic-professional-voice`
- **Hardware Target**: Google Colab T4 / V100 GPU
- **Configuration**: `configs/experiment001.yaml`
- **Results Summary**: Pending execution in Colab.
