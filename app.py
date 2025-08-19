# app.py
import streamlit as st

# Must be the first Streamlit command in the script
st.set_page_config(page_title="News Summarizer", page_icon="📰")

st.title("📰 News Summarizer")
st.caption("Paste an article URL or text → click Summarize → get a concise summary.")

# ...rest of your imports are fine after this, but avoid other st.* before set_page_config
from transformers import T5ForConditionalGeneration, T5Tokenizer
# ...your app logic/UI below

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="centered")

@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    # CPU by default on Streamlit Cloud; device -1 = CPU
    return pipeline("summarization", model=model_name, framework="pt", device=-1)

def summarize_long_text(summarizer, text: str, max_len: int, min_len: int):
    # Simple char-based chunking to stay under model token limits
    chunks = []
    chunk_size = 2500  # ~safe for distilbart on CPU
    text = text.strip()
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    parts = []
    for idx, chunk in enumerate(chunks, 1):
        with st.status(f"Summarizing chunk {idx}/{len(chunks)}...", expanded=False):
            out = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
        parts.append(out[0]["summary_text"].strip())

    # If multiple chunks, do a final pass to tighten it
    if len(parts) > 1:
        joined = " ".join(parts)
        out = summarizer(
            joined,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        return out[0]["summary_text"].strip()
    return parts[0] if parts else ""

st.title("📰 News Summarizer")

model = st.selectbox(
    "Model",
    ("sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn", "t5-small"),
    index=0,
    help="DistilBART is light & fast for CPU on Streamlit Cloud."
)

col1, col2 = st.columns(2)
with col1:
    max_len = st.slider("Max summary length", 64, 512, 180, step=8)
with col2:
    min_len = st.slider("Min summary length", 20, 200, 60, step=5)

text = st.text_area(
    "Paste article text",
    height=240,
    placeholder="Paste a news article here…",
)

if st.button("Summarize", type="primary", disabled=not text.strip()):
    with st.spinner("Loading model… (first run may take a bit)"):
        summarizer = load_summarizer(model)
    with st.spinner("Summarizing…"):
        summary = summarize_long_text(summarizer, text, max_len, min_len)
    st.subheader("Summary")
    st.write(summary)

    st.download_button("Download summary", summary, file_name="summary.txt")

