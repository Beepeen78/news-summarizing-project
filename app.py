import os

# --- Keep RAM low and avoid parallel tokenization ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import streamlit as st

# st.set_page_config MUST be the first Streamlit call
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="centered")

# Heavy libs after page_config
import traceback
import torch
from transformers import pipeline

# Keep CPU threads to 1 to reduce memory spikes
try:
    torch.set_num_threads(1)
except Exception:
    pass

SAFE_DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"
HEAVY_MODEL = "facebook/bart-large-cnn"

st.title("📰 News Summarizer")
st.caption("Paste an article → click Summarize → get a concise summary.")

with st.expander("⚙️ Advanced options", expanded=False):
    allow_heavy = st.checkbox(
        f"Show large model ({HEAVY_MODEL}) — slower & may exceed free-tier RAM",
        value=False,
        help="Enable only if your deployment has enough memory.",
    )

models = [SAFE_DEFAULT_MODEL, "t5-small"]
if allow_heavy:
    models.insert(1, HEAVY_MODEL)

c0, c1, c2 = st.columns([2, 1, 1])
with c0:
    model_name = st.selectbox("Model", models, index=0)
with c1:
    max_len = st.slider("Max length", 64, 512, 180, step=8)
with c2:
    min_len = st.slider("Min length", 16, 200, 60, step=4)

text = st.text_area(
    "Paste article text",
    height=240,
    placeholder="Paste your news article here…",
)

@st.cache_resource(show_spinner=False)
def load_summarizer(name: str):
    """
    Load a summarization pipeline with safer settings on CPU.
    Falls back to SAFE_DEFAULT_MODEL if loading fails.
    """
    model_kwargs = {}
    if name == HEAVY_MODEL:
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float32,
        }

    try:
        summarizer = pipeline(
            task="summarization",
            model=name,
            tokenizer=name,
            framework="pt",
            device=-1,  # CPU
            model_kwargs=model_kwargs,
        )
        return summarizer, None
    except Exception as e:
        primary_err = f"Failed to load '{name}': {type(e).__name__}: {e}"
        traceback.print_exc()
        # Fallback to safe model
        try:
            fb = pipeline(
                task="summarization",
                model=SAFE_DEFAULT_MODEL,
                tokenizer=SAFE_DEFAULT_MODEL,
                framework="pt",
                device=-1,
                model_kwargs={"low_cpu_mem_usage": True},
            )
            return fb, primary_err
        except Exception as e2:
            traceback.print_exc()
            raise RuntimeError(
                f"Could not load any model. Primary: {primary_err} | Fallback: {type(e2).__name__}: {e2}"
            )

def chunk_and_summarize(summarizer, text: str, max_len: int, min_len: int):
    text = (text or "").strip()
    if not text:
        return ""

    # ~2.5k chars per chunk to keep within CPU/token limits
    chunk_size = 2500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = []
    for idx, chunk in enumerate(chunks, 1):
        with st.status(f"Summarizing chunk {idx}/{len(chunks)}…", expanded=False):
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

    return parts[0]

btn = st.button("Summarize", type="primary", disabled=not text.strip())

if btn:
    with st.spinner("Loading model… (first run downloads weights)"):
        summarizer, load_warning = load_summarizer(model_name)

    if load_warning:
        st.warning(
            f"🔁 {load_warning}\n\n"
            f"Using fallback model: **{SAFE_DEFAULT_MODEL}** to keep things running."
        )

    with st.spinner("Summarizing…"):
        try:
            summary = chunk_and_summarize(summarizer, text, max_len, min_len)
            st.subheader("Summary")
            st.write(summary)
            st.download_button("Download summary", summary, file_name="summary.txt")
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error while summarizing: {type(e).__name__}: {e}")
