# 📰 News Summarizer (Streamlit)

A lightweight Streamlit app that turns long news articles into concise summaries using Hugging Face Transformers.

**Live Demo:** https://news-summarizing-project-g2ks35nieplfrzvwpozfzi.streamlit.app/

---

## Features

- **Paste text** and instantly get a **concise summary**
- Choose a model:
  - `sshleifer/distilbart-cnn-12-6` (fast, great for CPU)
  - `facebook/bart-large-cnn`
  - `t5-small`
- Tune **max/min summary length**
- Handles long inputs with smart **chunking**
- Caches model with `st.cache_resource` for faster repeat runs
- **Download** the summary as a `.txt` file

---

## Tech Stack

- **Frontend / App:** Streamlit `1.36.0`
- **NLP:** Transformers `4.46.3`, Tokenizers `0.20.3`
- **Runtime:** Python `3.11` (recommended on Streamlit Cloud)
- **DL backend:** PyTorch `2.5.1` (CPU by default on Cloud)

---

## Project Structure

```
.
├─ app.py                # Streamlit UI + summarization logic
├─ requirements.txt      # Pinned deps with wheels (Cloud-friendly)
└─ runtime.txt           # Python 3.11 hint for Streamlit Cloud
```

---

## Quick Start (Local)

> Requires Python 3.11

```bash
# 1) Create & activate venv
python -m venv .venv
# Windows PowerShell:
. .venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

If you want to install pinned packages without `requirements.txt`:
```bash
pip install ^
  "streamlit==1.36.0" ^
  "transformers==4.46.3" ^
  "tokenizers==0.20.3" ^
  "torch==2.5.1" ^
  "sentencepiece==0.2.1" ^
  "safetensors==0.6.2" ^
  "evaluate==0.4.5" ^
  "reportlab==4.4.3" ^
  "pillow<11"
```

---

## How to Use

1. Open the app.
2. Pick a **model** (DistilBART is fastest on CPU).
3. Paste your **article text**.
4. Adjust **Min/Max length** if needed.
5. Click **Summarize** → read the result → **Download** if you like.

---

## Implementation Highlights

- **Single `st.set_page_config`** call at the **top** of `app.py` (required by Streamlit).
- **Chunking**: large inputs are split into ~2.5k-char chunks and summarized piecewise; if multiple parts, a final pass tightens the output.
- **Device**: uses CPU (`device=-1`) for Streamlit Cloud stability.
- **Caching**: `@st.cache_resource(show_spinner=False)` keeps the pipeline warm.

---

## Deployment Notes (Streamlit Cloud)

- The app is deployed on Streamlit Cloud with **Python 3.11**.
- We pinned **Transformers 4.46.3 + Tokenizers 0.20.x** so Cloud can use **prebuilt wheels** (no Rust build step).
- Keep `requirements.txt` saved as **UTF-8** (invalid encodings can break Cloud’s installer).
- Avoid mixing frameworks (e.g., **remove Flask/Gradio imports** from `app.py`).

**Common log messages and fixes:**

- `set_page_config() can only be called once…`  
  → Ensure there’s **only one** `st.set_page_config(...)` and it’s the **first Streamlit call** in `app.py`.

- `ModuleNotFoundError: flask`  
  → This app doesn’t use Flask; make sure `app.py` does **not** import Flask.

- Tokenizers build failing on Python 3.13  
  → Use Python **3.11** on Cloud (via `runtime.txt`) and **tokenizers 0.20.x**.

- `torch.classes __path__._path` warning  
  → Benign; PyTorch still runs on CPU fine.

---

## Example Code Snippet (core)

```python
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="centered")

@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    return pipeline("summarization", model=model_name, framework="pt", device=-1)

def summarize_long_text(summarizer, text: str, max_len: int, min_len: int):
    chunks = [text[i:i+2500] for i in range(0, len(text.strip()), 2500)]
    parts = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
        parts.append(out[0]["summary_text"].strip())
    if len(parts) > 1:
        joined = " ".join(parts)
        out = summarizer(joined, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
        return out[0]["summary_text"].strip()
    return parts[0] if parts else ""
```

---

## 📝 Roadmap (Nice-to-haves)

- URL fetch + readability extraction
- Abstractive vs. extractive toggle
- Multi-article batch summaries
- Model benchmarking & latency hints

---

## 🔒 Privacy

All text is processed in-app and is not stored. Clear the text area to remove content from your current session.

---

## 📄 License

MIT — feel free to fork, improve, and share.
