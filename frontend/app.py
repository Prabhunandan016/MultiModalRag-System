import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from ingestion.youtube_loader import load_youtube_transcript
from ingestion.pdf_loader import load_pdf
from ingestion.image_loader import load_image
from preprocessing.cleaning import clean_documents
from preprocessing.chunking import chunk_documents
from embedding.embedding_model import EmbeddingModel
from vector_store.vector_db import VectorStore
from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from llm.answer_generator import AnswerGenerator


st.set_page_config(
    page_title="MultiRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Hide default header */
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* Hero */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c63ff, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #8b8fa8;
        font-size: 1.05rem;
    }

    /* Cards */
    .card {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6c63ff;
        margin-bottom: 0.8rem;
    }

    /* Answer box */
    .answer-box {
        background: #1a1f2e;
        border: 1px solid #6c63ff44;
        border-left: 4px solid #6c63ff;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        line-height: 1.8;
        color: #dde1f0;
        font-size: 1rem;
    }

    /* Status badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ready { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade8044; }
    .badge-idle  { background: #2a2a1a; color: #facc15; border: 1px solid #facc1544; }

    /* Sidebar labels */
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6c63ff;
        margin: 1.2rem 0 0.4rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #48cae4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: #1a1f2e !important;
        border: 1px solid #2a2f3e !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1a1f2e;
        border: 1px dashed #2a2f3e;
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* Divider */
    hr { border-color: #2a2f3e; }

    /* Metric row */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric {
        flex: 1;
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6c63ff;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)


# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return EmbeddingModel(), Reranker(), AnswerGenerator()

embedder, reranker, llm = load_models()

for key in ["ready", "dense_retriever", "bm25_retriever", "doc_count", "source_label"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MultiRAG")
    st.markdown("<div style='color:#8b8fa8;font-size:0.85rem'>Multimodal Retrieval-Augmented Generation</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='sidebar-label'>Data Source</div>", unsafe_allow_html=True)
    source_type = st.selectbox("", ["YouTube", "PDF", "Image"], label_visibility="collapsed")

    st.markdown("<div class='sidebar-label'>Load Data</div>", unsafe_allow_html=True)

    documents = []

    if source_type == "YouTube":
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        load_clicked = st.button("▶  Load Video")
        if load_clicked and url:
            try:
                with st.spinner("Fetching transcript..."):
                    documents = load_youtube_transcript(url)
                st.session_state.source_label = f"YouTube · {url[-11:]}"
            except Exception as e:
                st.error(f"Failed: {e}")
                st.stop()

    elif source_type == "PDF":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file:
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())
            documents = load_pdf("temp.pdf")
            st.session_state.source_label = f"PDF · {pdf_file.name}"

    elif source_type == "Image":
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if img_file:
            with open("temp.png", "wb") as f:
                f.write(img_file.read())
            documents = load_image("temp.png")
            st.session_state.source_label = f"Image · {img_file.name}"

    st.markdown("---")

    status_html = (
        "<span class='badge badge-ready'>● Ready</span>"
        if st.session_state.ready else
        "<span class='badge badge-idle'>● Idle</span>"
    )
    st.markdown(f"**Status** &nbsp; {status_html}", unsafe_allow_html=True)

    if st.session_state.source_label:
        st.markdown(f"<div style='color:#8b8fa8;font-size:0.8rem;margin-top:0.5rem'>📄 {st.session_state.source_label}</div>", unsafe_allow_html=True)

    if st.session_state.doc_count:
        st.markdown(f"<div style='color:#8b8fa8;font-size:0.8rem'>🗂 {st.session_state.doc_count} chunks indexed</div>", unsafe_allow_html=True)


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🧠 MultiRAG</h1>
    <p>Ask questions from YouTube videos, PDFs, or Images using AI-powered retrieval</p>
</div>
""", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
if documents:
    st.markdown("<div class='card'><div class='card-title'>⚙ Building Pipeline</div>", unsafe_allow_html=True)
    progress = st.progress(0)
    status_text = st.empty()

    status_text.markdown("<span style='color:#8b8fa8'>Cleaning documents...</span>", unsafe_allow_html=True)
    progress.progress(15)
    documents = clean_documents(documents)

    status_text.markdown("<span style='color:#8b8fa8'>Chunking documents...</span>", unsafe_allow_html=True)
    progress.progress(30)
    documents = chunk_documents(documents)

    status_text.markdown("<span style='color:#8b8fa8'>Generating embeddings...</span>", unsafe_allow_html=True)
    progress.progress(55)
    embeddings = embedder.embed_documents(documents)

    status_text.markdown("<span style='color:#8b8fa8'>Storing in vector database...</span>", unsafe_allow_html=True)
    progress.progress(75)
    vector_db = VectorStore()
    vector_db.add_documents(documents, embeddings)

    status_text.markdown("<span style='color:#8b8fa8'>Setting up retrievers...</span>", unsafe_allow_html=True)
    progress.progress(90)
    st.session_state.dense_retriever = DenseRetriever(vector_db, embedder)
    st.session_state.bm25_retriever = BM25Retriever(documents)
    st.session_state.doc_count = len(documents)
    st.session_state.ready = True

    progress.progress(100)
    status_text.markdown("<span style='color:#4ade80'>✓ Pipeline ready</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Q&A ───────────────────────────────────────────────────────────────────────
if st.session_state.ready:

    if st.session_state.doc_count:
        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric'>
                <div class='metric-value'>{st.session_state.doc_count}</div>
                <div class='metric-label'>Chunks Indexed</div>
            </div>
            <div class='metric'>
                <div class='metric-value'>BGE</div>
                <div class='metric-label'>Embedding Model</div>
            </div>
            <div class='metric'>
                <div class='metric-value'>RRF</div>
                <div class='metric-label'>Fusion Strategy</div>
            </div>
            <div class='metric'>
                <div class='metric-value'>Llama 3</div>
                <div class='metric-label'>LLM</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>💬 Ask a Question</div>", unsafe_allow_html=True)
    query = st.text_input("", placeholder="What is this content about?", label_visibility="collapsed")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_clicked = st.button("🔍  Get Answer")

    st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked and query:
        with st.spinner("Retrieving and generating answer..."):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_dense = executor.submit(st.session_state.dense_retriever.retrieve, query)
                future_bm25 = executor.submit(st.session_state.bm25_retriever.retrieve, query)
                dense_results = future_dense.result()
                bm25_results = future_bm25.result()

            fused_results = reciprocal_rank_fusion([dense_results, bm25_results])
            final_results = reranker.rerank(query, fused_results)
            answer = llm.generate_answer(query, final_results)

        st.markdown("<div class='card-title' style='margin-top:1rem'>✦ Answer</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, doc in enumerate(final_results, 1):
                src = doc.metadata.get("source", "unknown")
                extra = doc.metadata.get("page", doc.metadata.get("video_id", ""))
                st.markdown(f"""
                <div class='card' style='margin-bottom:0.6rem'>
                    <div class='card-title'>Source {i} · {src.upper()} {f"· {extra}" if extra else ""}</div>
                    <div style='color:#c0c4d6;font-size:0.88rem;line-height:1.6'>{doc.content[:400]}{"..." if len(doc.content) > 400 else ""}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='card' style='text-align:center;padding:3rem'>
        <div style='font-size:3rem'>📂</div>
        <div style='color:#8b8fa8;margin-top:0.8rem;font-size:1rem'>Load a data source from the sidebar to get started</div>
        <div style='color:#6c63ff;margin-top:0.4rem;font-size:0.85rem'>YouTube · PDF · Image</div>
    </div>
    """, unsafe_allow_html=True)
