# 🕵️ Inspectra – Semantic Plagiarism & Paraphrase Detector

**Inspectra** is a powerful web-based tool designed to detect plagiarism, paraphrased content, and semantic similarity between documents. It uses advanced NLP models like Sentence-BERT, BLEU, ROUGE, and keyword-based web scraping to identify both direct and paraphrased overlaps — going beyond traditional copy-paste detection.

---

## 🚀 Features

- 🔍 **Semantic Similarity** using [Sentence-BERT (`paraphrase-MiniLM-L6-v2`)](https://www.sbert.net/)
- 🧠 **BLEU & ROUGE** metrics for sentence-level evaluation
- 📊 **Plagiarism Percentage Estimation** based on similarity matrices
- 🌐 **Web Scraping Detection** – checks if the content exists online using:
  - [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction
  - Google + Wikipedia search scraping
  - Cosine similarity with scraped web content
- ✨ **Highlighting** of matching/paraphrased sentences across documents
- 📎 Supports `.txt` and `.pdf` files

---

## 🧱 Tech Stack

| Layer            | Tool/Library                                   |
|------------------|------------------------------------------------|
| Backend          | Python, Sentence Transformers, NLTK, ROUGE     |
| Web Scraping     | BeautifulSoup, Requests, KeyBERT               |
| Similarity Engine| Cosine Similarity, ROUGE-L                     |
| Frontend         | [Streamlit](https://streamlit.io)              |
| Preprocessing    | Numpy, Regex, TF-IDF                           |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/inspectra.git
cd inspectra
pip install -r requirements.txt
