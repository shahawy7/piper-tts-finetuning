# 🐣 Beginner's Step-by-Step Guide to Arabic Piper TTS Fine-Tuning

Welcome! If you are new to Text-to-Speech (TTS) or AI model fine-tuning, this guide is written specifically for you. You don't need an expensive GPU or advanced machine learning background to follow this project. Everything is designed to run smoothly in **Google Colab** (a free web-based environment provided by Google).

---

## 💡 What is Fine-Tuning? (Simple Explanation)

Imagine you have a native Arabic speaker who speaks with a specific dialect or tone (the **base model**: `ar_JO-kareem-medium`). Fine-tuning is the process of taking this already trained model and giving it extra practice with a new, high-quality studio recording dataset (the **training dataset**: `Arabic-professional-voice`).

By fine-tuning instead of training from scratch:
- **Faster Training**: Takes hours instead of weeks.
- **Better Quality**: Preserves knowledge of Arabic phonemes while adopting the clear pronunciation and diacritics (تَشْكِيل) of the new dataset.
- **Efficient**: Keeps the inference speed super fast!

---

## 🛠️ Prerequisites

All you need is:
1. A **Google Account** (for Google Drive & Google Colab).
2. A web browser (Chrome, Firefox, Edge, etc.).
3. No local installation or powerful computer required!

---

## 🚀 Step 1: Open the Notebook in Google Colab

Everything runs from a **single notebook** — no need to switch between files or sessions.

### Option A: Clone from GitHub (Recommended ✅)

1. Push this repository to your own GitHub account (public or private).
2. Go to [Google Colab](https://colab.research.google.com).
3. Open the notebook using one of these methods:
   - **From GitHub tab**: Click **File ➔ Open notebook ➔ GitHub** tab, paste your repo URL, and select `notebooks/arabic_piper_finetuning.ipynb`.
   - **Direct URL**: Navigate to `https://colab.research.google.com/github/YOUR_USERNAME/piper-tts-finetuning/blob/main/notebooks/arabic_piper_finetuning.ipynb`.
4. The first cell will automatically clone the repository and set up the working directory.

### Option B: Upload to Google Drive (Fallback)

If you prefer not to use GitHub:
1. Download or zip this repository folder.
2. Upload it into your Google Drive under `MyDrive/`.
3. In the Colab notebook, change the working directory to the uploaded folder path.

> [!IMPORTANT]
> **Enable GPU in Google Colab**:
> In Colab, click **Runtime** in the top menu bar ➔ **Change runtime type** ➔ Select **T4 GPU** (or any available GPU) ➔ Click **Save**.

---

## 📖 Step 2: The 5-Section Notebook Pipeline

The notebook contains **5 sections** that you run top-to-bottom inside a single Colab session:

```text
┌────────────────────────────────────────────────────────────┐
│  ⚙️ Section 1: Environment Setup                          │
│  Clones repo from GitHub, mounts Google Drive,             │
│  installs dependencies (espeak-ng, piper-tts, torch, etc.) │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│  📦 Section 2: Dataset Download & Preparation              │
│  Downloads Arabic audio dataset & resamples to 22,050 Hz   │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│  🔊 Section 3: Baseline Benchmark                          │
│  Generates "BEFORE" audio — your quality reference point   │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│  🏋️ Section 4: Fine-Tuning & Checkpointing                │
│  Trains the model & auto-saves checkpoints to Google Drive │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│  📊 Section 5: Export & Evaluation                         │
│  Exports ONNX model & compares "BEFORE" vs "AFTER" audio   │
└────────────────────────────────────────────────────────────┘
```

---

### ⚙️ Section 1: Environment Setup
- **What it does**: Clones the repository from GitHub into Colab, verifies GPU access, mounts Google Drive, and installs required libraries.
- **How to run**: Click the **Play (▶)** button on each code cell from top to bottom.
- **First cell**: Edit the `REPO_URL` variable to point to your GitHub repository. If the repo is already cloned from a previous session, it will automatically `git pull` the latest changes instead.
- **What to look for**: A message asking for permission to mount Google Drive. Click **Connect to Google Drive** and sign in.

---

### 📦 Section 2: Dataset Download & Preparation
- **What it does**: Automatically downloads the `NightPrince/Arabic-professional-voice` dataset from Hugging Face and prepares the audio clips (resampling them to standard 22,050 Hz 16-bit audio).
- **What to look for**: Progress bars showing dataset download and processing. At the end, it will print the number of training and validation audio samples.

---

### 🔊 Section 3: Baseline Benchmark
- **What it does**: Uses the original pre-trained `ar_JO-kareem-medium` model to synthesize 10 benchmark Arabic test sentences.
- **Why is this important?**: This creates your **"BEFORE" baseline**. You will listen to these audio clips to know how much fine-tuning improved the voice!
- **What to look for**: An audio player right inside Colab to listen to the generated baseline speech.

---

### 🏋️ Section 4: Fine-Tuning & Checkpointing
- **What it does**: Starts the fine-tuning training loop!
- **Auto-Save Protection**: Every 5 epochs, a checkpoint (`.ckpt` file) is saved to your Google Drive folder (`/MyDrive/Arabic-Piper/checkpoints/`).
- **What if Colab disconnects?**: Don't worry! Re-run the notebook from Section 1 (setup is fast since everything is cached). Training will automatically resume from the latest checkpoint in Google Drive.
- **TensorBoard**: Includes an inline graph showing training loss decreasing over time.

---

### 📊 Section 5: Export & Evaluation
- **What it does**: Takes your trained checkpoint, converts it to an optimized **ONNX model (`.onnx`)**, and generates "AFTER" audio for the benchmark sentences.
- **Side-by-Side Comparison**: Listen to the Baseline ("BEFORE") vs Fine-Tuned ("AFTER") audio clips directly in Colab to hear the improvement!

---

## ❓ Frequently Asked Questions (FAQ)

### Q1: Why is full diacritization (تَشْكِيل) important for Arabic TTS?
Arabic words change pronunciation and meaning based on short vowels (Fatha َ, Damma ُ, Kasra ِ, Sukun ْ). Without diacritics, a TTS engine has to guess the vowels. Providing fully diacritized text ensures 100% accurate pronunciation!

### Q2: How long does fine-tuning take on Google Colab?
- With a free T4 GPU, 50 epochs typically takes around **30 to 60 minutes**.
- You can stop training early or run for more epochs depending on your loss curve in TensorBoard.

### Q3: How do I use the fine-tuned voice on my local computer?
After completing Section 5, download the generated `.onnx` and `.onnx.json` files from your Google Drive:
```text
/content/drive/MyDrive/Arabic-Piper/outputs/experiment001/ar_JO_finetuned.onnx
/content/drive/MyDrive/Arabic-Piper/outputs/experiment001/ar_JO_finetuned.onnx.json
```
Then test locally using:
```bash
python scripts/test_local.py --mode finetuned --model ar_JO_finetuned.onnx --text "السَّلَامُ عَلَيْكُمْ"
```
See the **[Local Testing Guide](local_testing_guide.md)** for more details.

### Q4: Can I control the speaking speed?
Yes! Use the `--length-scale` parameter:
```bash
# Faster speech (80% duration)
python scripts/test_local.py --mode baseline --length-scale 0.8 --text "مَرْحَبًا"

# Slower, clearer speech (130% duration)
python scripts/test_local.py --mode baseline --length-scale 1.3 --text "مَرْحَبًا"
```

---

## 🎯 Summary Checklist

- [ ] Open `notebooks/arabic_piper_finetuning.ipynb` in Google Colab with **T4 GPU** enabled.
- [ ] Edit `REPO_URL` in the first cell with your GitHub username.
- [ ] Run all cells top-to-bottom through Sections 1–5.
- [ ] Listen to "BEFORE" vs "AFTER" audio comparison in Section 5.
- [ ] Download `.onnx` model from Google Drive for local use.
