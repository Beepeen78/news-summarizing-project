# app.py
import streamlit as st
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="centered")

from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    # CPU on Streamlit Cloud; device=-1 forces CPU
    return pipeline("summarization", model=model_name, framework="pt", device=-1)

def summarize_long_text(summarizer, text: str, max_len: int, min_len: int):
    # simple chunking to stay within model limits
    text = text.strip()
    chunk_size = 2500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

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
st.caption("Paste an article text → click Summarize → get a concise summary.")

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
