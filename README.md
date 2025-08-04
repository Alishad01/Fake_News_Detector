# 📰 Fake News Generator & Detector using Generative AI and NLP

This project showcases the **power and responsibility** of Generative AI. It demonstrates how large language models like **GPT-2** can be used to **generate fake news headlines**, and how transformer-based models like **BERT** can be trained to **detect fake vs real headlines**.

---

## 🚀 Project Overview

| Component        | Model Used | Task                         |
|------------------|------------|------------------------------|
| Fake Headline Generator | GPT-2       | Generate believable fake news |
| Fake News Detector      | BERT        | Classify news as real or fake |

---

## 🎯 Objectives

- Understand and apply **Generative AI (GPT-2)** for content creation
- Use **Natural Language Processing (NLP)** to clean and process text data
- Fine-tune **BERT** for binary classification (real vs fake)
- Promote ethical awareness in AI and misinformation detection

---

## 🧠 Technologies & Tools

- Python 🐍
- Hugging Face 🤗 Transformers
- PyTorch
- Pandas, NumPy, scikit-learn
- NLTK (text cleaning)
- Google Colab
- Gradio for UI
---

## 📂 Dataset Used

- **Fake and Real News Dataset** from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- ~3,000 headlines (cleaned, preprocessed)
- Labels: `1` = Fake, `0` = Real

---

## 🔧 Project Structure

git clone https://github.com/Alishad01/Fake_News_Detector
cd Fake_News_Detector
pip install -r requirements.txt

# For generation
python generator/generate.py

**Accuracy: 0.87**
Generated headline: "Breaking: Government declares national holiday..."
Sample prediction input/output screenshot

## 📈 Results

- **Dataset:** ~3,000 headlines (fake and real)
- **Classifier:** BERT accuracy ≈ 87%, loss ≈ 0.39
- **Generator:** GPT‑2 able to produce realistic headlines like:
- "Breaking: Scientists warn of massive asteroid heading toward Earth"
- "Reports say virus outbreak shuts down international flights"

## 🧰 Quick Start

```
  bash
  git clone <your repo URL>
  cd Fake_News_Detector
  pip install -r requirements.txt
  python generator/generate.py
  python detector/train_and_eval.py
'''
# For detection
python detector/train_and_eval.py
