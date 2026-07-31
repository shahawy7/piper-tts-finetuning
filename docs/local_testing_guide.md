# 💻 Guide: Testing Arabic Piper TTS Locally on CPU (Before & After Training)

This guide explains how to test your Arabic Text-to-Speech (TTS) models locally on your computer's **CPU** without needing a GPU. You can test the **Baseline model (Before Training)** to hear how the original voice sounds, and test your **Fine-Tuned model (After Training)** to verify quality improvements.

---

## ⚙️ 1. Setup Local Environment

Open your terminal or command prompt inside the project directory:

```bash
# 1. Install required Python packages
pip install -r requirements.txt

# 2. (Optional) If you have Piper installed on your system PATH:
# Linux: sudo apt install piper-tts  (or download from github.com/rhasspy/piper)
# macOS: brew install piper
```

---

## 🔊 2. Testing BEFORE Training (Baseline Model)

Before training on Colab, test the baseline `ar_JO-kareem-medium` model locally on CPU to record how the default voice pronounces Arabic words.

### Single Sentence Test:
```bash
python scripts/test_local.py --mode baseline --text "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ."
```

#### What happens automatically:
1. Automatically downloads the baseline `ar_JO-kareem-medium.onnx` and `.onnx.json` model into `checkpoints/base/`.
2. Synthesizes the Arabic text on CPU.
3. Saves the audio output to `outputs/local_test/baseline_test.wav`.

---

## 🎙️ 3. Testing AFTER Training (Your Fine-Tuned Model)

After completing training in Google Colab (Notebook 04/05), download your exported `.onnx` and `.onnx.json` files from Google Drive to your local repository directory (e.g. `outputs/experiment001/ar_JO_finetuned.onnx`).

### Test Your Fine-Tuned Model:
```bash
python scripts/test_local.py --mode finetuned --model outputs/experiment001/ar_JO_finetuned.onnx --text "مَرْحَبًا بِكُمْ فِي مَشْرُوعِ تَطْوِيرِ التَّحْوِيلِ الصَّوْتِيِّ."
```

#### Output:
Saves the fine-tuned speech audio to `outputs/local_test/finetuned_test.wav`.

---

## ⚖️ 4. Side-by-Side Comparison (Baseline vs Fine-Tuned)

Compare the "BEFORE" vs "AFTER" models directly on the exact same Arabic sentence:

```bash
python scripts/test_local.py --mode compare --model outputs/experiment001/ar_JO_finetuned.onnx --text "يَتَمَيَّزُ هَذَا النَّمُوذَجُ بِدِقَّةِ النُّطْقِ وَسُرْعَةِ الْأَدَاءِ."
```

#### Output Files:
- `outputs/local_test/baseline_sample.wav` ➔ Pre-training audio
- `outputs/local_test/finetuned_sample.wav` ➔ Post-training audio

---

## 💬 5. Interactive Terminal Prompt Mode

You can type custom diacritized Arabic text interactively and hear/save audio immediately!

```bash
# Interactive mode with Baseline model
python scripts/test_local.py --mode baseline --interactive

# Interactive mode with Fine-Tuned model
python scripts/test_local.py --mode finetuned --model outputs/experiment001/ar_JO_finetuned.onnx --interactive
```

### Example Session:
```text
=== Starting Interactive Local CPU Testing ===
Model: checkpoints/base/ar_JO-kareem-medium.onnx
Type Arabic diacritized text (or 'exit' to quit):

Arabic Input ➔ هَلْ يُمْكِنُ لِلذَّكَاءِ الِاصْطِنَاعِيِّ أَنْ يُحَسِّنَ نَوْعِيَّةَ الْحَيَاةِ؟
✓ Audio generated in 0.18s (Audio Duration: 3.42s | RTF: 0.0526)
Saved to: 'outputs/local_test/interactive_001.wav'

Arabic Input ➔ exit
```

---

## 🎵 6. How to Play Generated WAV Files

### Linux:
```bash
aplay outputs/local_test/baseline_test.wav
# or using mpv / vlc
mpv outputs/local_test/baseline_test.wav
```

### macOS:
```bash
afplay outputs/local_test/baseline_test.wav
```

### Windows (PowerShell / Command Prompt):
```powershell
start outputs/local_test/baseline_test.wav
```

---

## 📋 Summary of Commands

| Task | Command |
|---|---|
| **Test Baseline CPU** | `python scripts/test_local.py --mode baseline` |
| **Test Fine-Tuned CPU** | `python scripts/test_local.py --mode finetuned --model path/to/model.onnx` |
| **Side-by-Side Compare** | `python scripts/test_local.py --mode compare --model path/to/model.onnx` |
| **Interactive Terminal** | `python scripts/test_local.py --mode baseline --interactive` |
