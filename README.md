# 🛡️ Gendered Abuse Detection in Hindi, Tamil, and English

This project presents a multilingual gendered abuse detection system targeting social media text in **Hindi**, **Tamil**, and **Indian English**. Using NLP and transformer-based techniques, we aim to identify posts that are abusive, gender-targeted, or contain explicit language—especially toward marginalized genders and sexualities.

---

## 📌 Motivation

Online gender-based abuse is a growing issue that silences voices and reinforces social inequalities. While there's progress in detecting hate speech in English, tools for **Indic languages** are underdeveloped. This project bridges that gap using multilingual NLP techniques and annotated datasets for Hindi, Tamil, and English.

---

## 📚 Dataset Overview

The dataset contains ~23,000 social media posts annotated by experts and activists to identify:
- **Gendered abuse**
- **Explicit/aggressive language**
- **Targeting of marginalized gender/sexuality**

Each language includes:
- ✅ Annotator consensus
- ✅ Binary labels (1 for abusive, 0 for not abusive)
- ✅ NLP-cleaned and transliterated text

---

## 🧠 Tasks and Methodology

We explored **three core tasks** using multiple transformer-based architectures:

### 1️⃣ **Task 1: Baseline Gendered Abuse Detection**
- Models: `DistilRoBERTa`, `mBERT`
- Preprocessing: Cleaning, tokenization, transliteration
- Training: 3 epochs, AdamW optimizer
- Metrics: Accuracy, Weighted F1 Score

### 2️⃣ **Task 2: Transfer Learning**
- Pretrained data from TweetEval (English), HASOC (Hindi), and Tamil Offensive Corpus
- Fine-tuning models on abusive content
- Augmentation: Back-translation, synonym swap

### 3️⃣ **Task 3: Joint Classification**
- Detect **gendered abuse** and **explicit language** using a multitask transformer model
- Architecture: Shared encoder + dual heads
- Also includes a `BiLSTM` variant for comparative analysis

---

## 🚀 Results (Final Model Highlights)

| Language | Gendered Abuse (F1) | Explicit Language (F1) |
|----------|---------------------|-------------------------|
| English  | 0.61                | 0.42                    |
| Hindi    | 0.705               | 0.50                    |
| Tamil    | 0.72                | 0.83                    |

> ✅ Tamil consistently outperformed in explicit content detection  
> ✅ Hindi performed best in gendered abuse detection

---

## 🧰 Tech Stack

- Transformers (HuggingFace)
- PyTorch + Trainer API
- IndicNLP + transliteration
- scikit-learn, pandas, numpy

---

## 🔍 Key Features

- 💡 Custom `Focal Loss` for class imbalance
- 📐 Multilingual fine-tuning with XLM-R, DeBERTa-v3
- ✍ Annotator-based label aggregation
- 🔄 Joint multitask training pipelines

---

## 📂 Folder Structure

```bash
.
├── data/                  # Processed datasets (train/test)
├── models/                # Saved model weights
├── src/                   # Training, evaluation, and preprocessing scripts
├── report.pdf             # Full paper/report
├── README.md              # This file
