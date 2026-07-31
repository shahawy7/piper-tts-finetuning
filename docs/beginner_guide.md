# 🐣 Beginner's Step-by-Step Guide to Arabic Piper TTS Fine-Tuning

Welcome! If you are new to Text-to-Speech (TTS) or AI model fine-tuning, this guide is written specifically for you. You don't need a expensive GPU or advanced machine learning background to follow this project. Everything is designed to run smoothly in **Google Colab** (a free web-based environment provided by Google).

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

## 🚀 Step 1: Prepare Your Workspace in Google Colab

### Option A: Upload to Google Drive
1. Open [Google Drive](https://drive.google.com).
2. Upload the `piper-tts-finetuning` folder into your Google Drive under `MyDrive/`.

### Option B: Open Notebooks directly from GitHub (Recommended)
1. Push this repository to your GitHub account.
2. Go to [Google Colab](https://colab.research.google.com).
3. Click **GitHub** tab, enter your repository URL, and open notebook `01_environment.ipynb`.

> [!IMPORTANT]
> **Enable GPU in Google Colab**:
> In Colab, click **Runtime** in the top menu bar ➔ **Change runtime type** ➔ Select **T4 GPU** (or any available GPU) ➔ Click **Save**.

---

## 📖 Step 2: The 5-Step Notebook Workflow

The repository includes **5 numbered notebooks** inside the `notebooks/` folder. You will run them one by one.

```text
┌──────────────────────────┐
│  01_environment.ipynb    │ ➔ Sets up GPU, mounts Google Drive & installs tools
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│     02_dataset.ipynb     │ ➔ Downloads Arabic audio dataset & base voice model
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│    03_baseline.ipynb     │ ➔ Generates "BEFORE" audio samples to compare quality
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│      04_train.ipynb      │ ➔ Fine-tunes model & auto-saves checkpoints to Drive
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│     05_evaluate.ipynb    │ ➔ Exports final ONNX model & generates "AFTER" audio
└──────────────────────────┘
```

---

### 1️⃣ Notebook 1: Environment Setup (`01_environment.ipynb`)
- **What it does**: Connects Colab to your Google Drive and installs required libraries (`torch`, `espeak-ng`, `piper-phonemize`).
- **How to run**: Click the **Play (▶)** button on each code cell from top to bottom.
- **What to look for**: A message asking for permission to mount Google Drive. Click **Connect to Google Drive** and sign in.
- **Output**: Creates a folder in your Drive named `Arabic-Piper/` where all checkpoints and audio files will be safely stored.

---

### 2️⃣ Notebook 2: Dataset Preparation (`02_dataset.ipynb`)
- **What it does**: Automatically downloads the `NightPrince/Arabic-professional-voice` dataset from Hugging Face and prepares the audio clips (resampling them to standard 22,050 Hz 16-bit audio).
- **How to run**: Run all cells sequentially.
- **What to look for**: Progress bars showing dataset download and processing. At the end, it will print the number of training and validation audio samples ready for training.

---

### 3️⃣ Notebook 3: Baseline Evaluation (`03_baseline.ipynb`)
- **What it does**: Uses the original pre-trained `ar_JO-kareem-medium` model to synthesize 10 benchmark Arabic test sentences (e.g., * السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ*).
- **Why is this important?**: This creates your **"BEFORE" baseline**. You will listen to these audio clips to know how much fine-tuning improved the voice!
- **What to look for**: An audio player right inside Colab to listen to the generated baseline speech.

---

### 4️⃣ Notebook 4: Training & Checkpointing (`04_train.ipynb`)
- **What it does**: Starts the fine-tuning training loop!
- **Auto-Save Protection**: Every 5 epochs, Colab automatically saves a checkpoint (`.ckpt` file) directly into your Google Drive folder (`/MyDrive/Arabic-Piper/checkpoints/`).
- **What if Colab disconnects?**: Don't worry! If your session disconnects or reaches the time limit, simply re-open Notebook 04 and run it again. It will automatically detect your latest saved checkpoint from Google Drive and resume right where it left off!
- **TensorBoard**: Includes an inline graph showing training loss decreasing over time.

---

### 5️⃣ Notebook 5: Model Export & Evaluation (`05_evaluate.ipynb`)
- **What it does**: Takes your trained PyTorch checkpoint, converts it to an optimized **ONNX model (`.onnx`)**, and generates "AFTER" audio for the benchmark sentences.
- **Side-by-Side Comparison**: Listen to the Baseline ("BEFORE") vs Fine-Tuned ("AFTER") audio clips directly in Colab to hear the improvement in Arabic diacritic pronunciation and voice clarity!

---

## ❓ Frequently Asked Questions (FAQ)

### Q1: Why is full diacritization (تَشْكِيل) important for Arabic TTS?
Arabic words change pronunciation and meaning based on short vowels (Fatha َ, Damma ُ, Kasra ِ, Sukun ْ). Without diacritics, a TTS engine has to guess the vowels. Providing fully diacritized text ensures 100% accurate pronunciation!

### Q2: How long does fine-tuning take on Google Colab?
- With a free T4 GPU, 50 epochs typically takes around **30 to 60 minutes**.
- You can stop training early or run for more epochs depending on your loss curve in TensorBoard.

### Q3: How do I use the fine-tuned voice on my local computer or in an application?
After running Notebook 05, download the generated `.onnx` and `.onnx.json` files from your Google Drive folder:
```text
/content/drive/MyDrive/Arabic-Piper/outputs/experiment001/ar_JO_finetuned.onnx
/content/drive/MyDrive/Arabic-Piper/outputs/experiment001/ar_JO_finetuned.onnx.json
```
You can pass these files into the [Piper TTS CLI](https://github.com/rhasspy/piper) or any Python script to generate Arabic speech offline instantly!

```bash
# Example local command using Piper
echo "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ." | piper --model ar_JO_finetuned.onnx --output_file greeting.wav
```

---

## 🎯 Summary Checklist

- [ ] Open Colab with **T4 GPU** enabled.
- [ ] Run `01_environment.ipynb` to mount Google Drive.
- [ ] Run `02_dataset.ipynb` to prepare training data.
- [ ] Run `03_baseline.ipynb` to record "BEFORE" audio.
- [ ] Run `04_train.ipynb` to fine-tune the model.
- [ ] Run `05_evaluate.ipynb` to export `.onnx` model & compare "AFTER" audio!
