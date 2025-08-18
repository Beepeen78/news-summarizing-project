# app.py — Streamlit summarizer (T5 / Pegasus / Combined) + TXT upload + PDF export
# Fast + fresh UI
# Deps:
#   streamlit>=1.35  transformers>=4.40  torch  nltk
#   evaluate   (optional, ROUGE)
#   reportlab  (optional, PDF export)

import os
import torch
import nltk
import streamlit as st
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from io import BytesIO
from datetime import datetime

# -------------------- Config --------------------
CACHE_DIR = "hf-cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ensure not in offline mode
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

# NLTK
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
_ensure_nltk()

def _load_rouge():
    try:
        import evaluate  # type: ignore
        return evaluate.load("rouge")
    except Exception:
        return None

# Hardware / dtype
HAS_CUDA = torch.cuda.is_available()
DEVICE = 0 if HAS_CUDA else -1
DTYPE = torch.float16 if HAS_CUDA else torch.float32

# -------------------- Models --------------------
@st.cache_resource(show_spinner=True)
def load_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, pipeline]:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, cache_dir=CACHE_DIR, torch_dtype=DTYPE
    )
    # Pipeline sends tensors to the right device
    pipe = pipeline("summarization", model=mdl, tokenizer=tok, device=DEVICE)
    return tok, mdl, pipe

# -------------------- Summarization helpers --------------------
def summarize(pipe, text: str, *, max_len: int, min_len: int,
              beams: int, ngram_no_repeat: int, len_penalty: float) -> str:
    out = pipe(
        text,
        max_length=max_len,
        min_length=min_len,
        num_beams=beams,
        no_repeat_ngram_size=ngram_no_repeat,
        length_penalty=len_penalty,
    )
    return out[0]["summary_text"].strip()

def combine_by_sentences(t5_text: str, pg_text: str, weight_t5: float) -> str:
    s_t5 = nltk.sent_tokenize(t5_text)
    s_pg = nltk.sent_tokenize(pg_text)
    take_t5 = max(1, round(len(s_t5) * weight_t5))
    take_pg = max(1, round(len(s_pg) * (1.0 - weight_t5)))
    combined = s_t5[:take_t5] + s_pg[:take_pg]
    seen, ordered = set(), []
    for s in combined:
        k = s.strip()
        if k and k not in seen:
            seen.add(k)
            ordered.append(k)
    return " ".join(ordered).strip()

# -------------------- PDF helper --------------------
def build_pdf_bytes(title: str, original: str, summary: str, meta: str):
    """Return bytes for a simple PDF; requires reportlab."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_LEFT

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=LETTER, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        meta_style = ParagraphStyle("meta", parent=styles["Normal"], fontSize=9, leading=11, alignment=TA_LEFT)
        body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=14)

        story = []
        story += [Paragraph(title, h1), Paragraph(meta, meta_style), Spacer(1, 12)]
        story += [Paragraph("Summary", h2)]
        for para in summary.split("\n\n"):
            story += [Paragraph(para.replace("\n", "<br/>"), body), Spacer(1, 6)]
        story += [Spacer(1, 12), Paragraph("Original", h2)]
        for para in original.split("\n\n"):
            story += [Paragraph(para.replace("\n", "<br/>"), body), Spacer(1, 6)]
        doc.build(story)
        return buf.getvalue(), None
    except Exception as e:
        return None, str(e)

# -------------------- UI --------------------
st.set_page_config(page_title="Text Summarizer Demo", page_icon="📝", layout="wide")

# Custom CSS — bright cards + nicer textareas (no dim)
st.markdown(
    """
    <style>
    /* Global tightening */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* White "cards" with soft shadow */
    .app-card {
        background: #ffffff;
        border: 1px solid #e8ecf3;
        border-radius: 16px;
        padding: 16px 16px 8px 16px;
        box-shadow: 0 6px 18px rgba(30, 41, 59, 0.06);
    }

    /* Bright textareas */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        font-size: 1.02rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
    }

    /* Tabs accent */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 2px solid #ef4444;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 Text Summarizer (T5 / Pegasus / Combo)")

# Persisted state
for k, v in {
    "last_input": "",
    "last_summary": "",
    "rouge_scores": None,
    "input_text": "",
}.items():
    st.session_state.setdefault(k, v)

# Sidebar (now with Speed mode and device info)
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model",
        ["T5-small", "Pegasus (xsum)", "Combined (T5 + Pegasus)"],
        index=0,
    )
    speed_mode = st.toggle("⚡ Speed mode (faster, slightly shorter)", value=True)
    max_len = st.slider("Max length", 64, 512, 150, 2)
    min_len = st.slider("Min length", 10, 200, 60, 2)
    beams = st.slider("Beam width", 1, 25, 8)
    ngram_no_repeat = st.slider("No-repeat n-gram size", 0, 5, 2)
    len_penalty = st.slider("Length penalty", 0.5, 2.0, 1.0, 0.1)
    weight_t5 = st.slider("Combined: weight on T5", 0.1, 0.9, 0.4, 0.05)

    st.caption(
        f"Device: **{'GPU' if HAS_CUDA else 'CPU'}** · Dtype: **{'fp16' if HAS_CUDA else 'fp32'}** · Cache: `./hf-cache`"
    )

# dynamic fast params
def apply_speed(max_len, min_len, beams):
    if not speed_mode:
        return max_len, min_len, beams
    # trim lengths, cut beams hard for speed
    return min(max_len, 140), min(min_len, 50), min(beams, 3)

col_left, col_right = st.columns(2)

# ---- Left: input + uploader ----
with col_left:
    st.markdown("### Paste article/text")
    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Or upload a .txt file", type=["txt"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8", errors="ignore")
                st.session_state["input_text"] = content
                st.info(f"Loaded **{uploaded.name}**")
            except Exception as e:
                st.error(f"Couldn't read file: {e}")
        text = st.text_area(
            label="",
            key="input_text",
            height=520,
            placeholder="Paste the text you want summarized…",
        )
        run = st.button("Summarize", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

# ---- Right: summary + original + export ----
with col_right:
    st.markdown("### Summary")
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    t_summary, t_original = st.tabs(["Summary", "Original"])

    with t_summary:
        st.text_area(
            "Generated summary",
            value=st.session_state.last_summary,
            height=520,
            key="summary_display",
            # not disabled (keeps bright style)
        )
    with t_original:
        st.text_area(
            "Original text (kept)",
            value=st.session_state.last_input,
            height=520,
            key="original_display",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download summary (.txt)",
            st.session_state.last_summary or "",
            file_name="summary.txt",
            disabled=not bool(st.session_state.last_summary),
        )
    with c2:
        st.download_button(
            "Download original (.txt)",
            st.session_state.last_input or "",
            file_name="original.txt",
            disabled=not bool(st.session_state.last_input),
        )
    with c3:
        # Build a PDF on demand
        disabled_pdf = not bool(st.session_state.last_summary or st.session_state.last_input)
        title = "Text Summary"
        meta = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
            f"Model: {model_choice}  •  "
            f"Params: max={max_len}, min={min_len}, beams={beams}, ngram={ngram_no_repeat}, len_pen={len_penalty}  •  "
            f"{'GPU fp16' if HAS_CUDA else 'CPU'}"
        )
        if st.button("Generate PDF", disabled=disabled_pdf):
            pdf_bytes, err = build_pdf_bytes(
                title=title,
                original=st.session_state.last_input or "",
                summary=st.session_state.last_summary or "",
                meta=meta,
            )
            if pdf_bytes:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="summary.pdf",
                    mime="application/pdf",
                )
            else:
                st.error(
                    "PDF generation requires `reportlab`. "
                    "Install it with: `pip install reportlab`"
                    + (f"\n\nDetails: {err}" if err else "")
                )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Optional ROUGE ----
with st.expander("Score with ROUGE (optional)"):
    ref = st.text_area(
        "Reference summary (gold)",
        height=140,
        placeholder="Paste a reference (ground-truth) summary to compute ROUGE…",
        key="ref_text",
    )
    if (st.session_state.last_summary and ref):
        rouge = _load_rouge()
        if rouge:
            scores = rouge.compute(
                predictions=[st.session_state.last_summary],
                references=[ref.strip()],
            )
            neat = {
                "ROUGE-1": round(scores.get("rouge1", 0.0), 4),
                "ROUGE-2": round(scores.get("rouge2", 0.0), 4),
                "ROUGE-L": round(scores.get("rougeL", 0.0), 4),
            }
            st.session_state.rouge_scores = neat
            st.json(neat)
        else:
            st.info("Package `evaluate` not installed — install it to compute ROUGE.")

# ---- Run summarization ----
if run:
    if not text or len(text.strip()) < 10:
        st.warning("Please paste some text (or upload a .txt) to summarize.")
        st.stop()

    # apply fast tweaks if enabled
    _max_len, _min_len, _beams = apply_speed(max_len, min_len, beams)

    with st.spinner("Summarizing… (first run may download weights)"):
        try:
            if model_choice == "T5-small":
                _, _, pipe_t5 = load_model("t5-small")
                summary = summarize(
                    pipe_t5, text,
                    max_len=_max_len, min_len=_min_len,
                    beams=_beams, ngram_no_repeat=ngram_no_repeat,
                    len_penalty=len_penalty,
                )
            elif model_choice == "Pegasus (xsum)":
                _, _, pipe_pg = load_model("google/pegasus-xsum")
                summary = summarize(
                    pipe_pg, text,
                    max_len=_max_len, min_len=_min_len,
                    beams=_beams, ngram_no_repeat=ngram_no_repeat,
                    len_penalty=len_penalty,
                )
            else:  # Combined
                _, _, pipe_t5 = load_model("t5-small")
                _, _, pipe_pg = load_model("google/pegasus-xsum")
                t5_sum = summarize(
                    pipe_t5, text,
                    max_len=_max_len, min_len=_min_len,
                    beams=_beams, ngram_no_repeat=ngram_no_repeat,
                    len_penalty=len_penalty,
                )
                pg_sum = summarize(
                    pipe_pg, text,
                    max_len=_max_len, min_len=_min_len,
                    beams=_beams, ngram_no_repeat=ngram_no_repeat,
                    len_penalty=len_penalty,
                )
                summary = combine_by_sentences(t5_sum, pg_sum, weight_t5)

            st.session_state.last_input = text
            st.session_state.last_summary = summary
            st.session_state.rouge_scores = None
            st.rerun()
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            st.stop()

st.caption("Models: t5-small, google/pegasus-xsum • Cache: ./hf-cache (gitignored)")
