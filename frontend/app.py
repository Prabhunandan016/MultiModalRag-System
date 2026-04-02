import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import tempfile
from concurrent.futures import ThreadPoolExecutor

from auth.history import sign_in, sign_up, sign_out, save_history, fetch_history
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


st.set_page_config(page_title="MultiRAG", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { background: #0a0c12; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0f1219; border-right: 1px solid #1e2433; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── Auth page ── */
.auth-wrap {
    max-width: 420px; margin: 6rem auto; padding: 2.5rem;
    background: #0f1219; border: 1px solid #1e2433; border-radius: 16px;
}
.auth-logo { text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem; }
.auth-title { text-align: center; font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.3rem; }
.auth-sub { text-align: center; font-size: 0.82rem; color: #475569; margin-bottom: 1.8rem; }

/* ── Brand ── */
.brand { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0 0.8rem; }
.brand-name { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
.brand-sub  { font-size: 0.68rem; color: #475569; }

/* ── Section label ── */
.section-label {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #334155; margin: 1.2rem 0 0.4rem;
}

/* ── History item ── */
.hist-item {
    background: #0a0c12; border: 1px solid #1e2433; border-radius: 8px;
    padding: 0.65rem 0.8rem; margin-bottom: 0.4rem; cursor: pointer;
    transition: border-color 0.15s;
}
.hist-item:hover { border-color: #2d3a52; }
.hist-q { font-size: 0.78rem; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-src { font-size: 0.65rem; color: #334155; margin-top: 0.2rem; }

/* ── Status pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600;
}
.status-ready { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-idle  { background: #1c1917; color: #64748b; border: 1px solid #292524; }
.status-dot   { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

/* ── Buttons ── */
.stButton > button {
    background: #1d4ed8; color: #fff; border: none;
    border-radius: 8px; padding: 0.55rem 1.2rem;
    font-size: 0.85rem; font-weight: 600; width: 100%;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #2563eb; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: #0f1219 !important; border: 1px solid #1e2433 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-size: 0.92rem !important; padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important; box-shadow: 0 0 0 3px #3b82f618 !important;
}
.stSelectbox > div > div {
    background: #0f1219 !important; border: 1px solid #1e2433 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
[data-testid="stFileUploader"] {
    background: #0f1219; border: 1px dashed #1e2433; border-radius: 10px; padding: 0.4rem;
}

/* ── Hero ── */
.hero { text-align: center; padding: 2.5rem 1rem 1.8rem; border-bottom: 1px solid #1e2433; margin-bottom: 1.8rem; }
.hero-badge {
    display: inline-block; background: #1e2d4a; color: #60a5fa;
    border: 1px solid #1d4ed840; border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    padding: 0.28rem 0.85rem; margin-bottom: 0.9rem; text-transform: uppercase;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem;
}
.hero p { color: #475569; font-size: 0.92rem; max-width: 460px; margin: 0 auto; }

/* ── Stats row ── */
.stats-row { display: flex; gap: 1px; background: #1e2433; border-radius: 12px; overflow: hidden; margin-bottom: 1.8rem; }
.stat { flex: 1; background: #0f1219; padding: 0.9rem; text-align: center; transition: background 0.2s; }
.stat:hover { background: #131929; }
.stat-value { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; }
.stat-label { font-size: 0.62rem; color: #334155; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.15rem; }

/* ── Q&A panel ── */
.qa-panel { background: #0f1219; border: 1px solid #1e2433; border-radius: 14px; padding: 1.4rem; margin-bottom: 1.4rem; }
.qa-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #334155; margin-bottom: 0.7rem; }

/* ── Answer ── */
.answer-header {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 0.7rem;
}
.answer-header-dot { width: 5px; height: 5px; border-radius: 50%; background: #3b82f6; }
.answer-box {
    background: #0f1219; border: 1px solid #1e2433; border-radius: 12px;
    padding: 1.4rem 1.6rem; line-height: 1.85; color: #cbd5e1; font-size: 0.93rem; position: relative;
}
.answer-box::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%; background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 3px 0 0 3px;
}

/* ── Source cards ── */
.sources-header { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #334155; margin: 1.4rem 0 0.7rem; }
.source-card {
    background: #0f1219; border: 1px solid #1e2433; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; transition: border-color 0.2s;
}
.source-card:hover { border-color: #2d3748; }
.source-meta { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.45rem; }
.source-num {
    width: 18px; height: 18px; border-radius: 50%; background: #1e2d4a; color: #60a5fa;
    font-size: 0.6rem; font-weight: 700; display: flex; align-items: center; justify-content: center;
}
.source-type {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.12rem 0.45rem; border-radius: 4px;
}
.type-youtube { background: #3f1212; color: #f87171; }
.type-pdf     { background: #1c1f12; color: #a3e635; }
.type-image   { background: #12203f; color: #60a5fa; }
.type-unknown { background: #1e2433; color: #94a3b8; }
.source-text  { font-size: 0.81rem; color: #475569; line-height: 1.6; }

/* ── Summary box ── */
.summary-box {
    background: #0f1219; border: 1px solid #1e2433; border-radius: 12px;
    padding: 1.4rem 1.6rem; line-height: 1.85; color: #cbd5e1; font-size: 0.93rem;
    position: relative; margin-top: 0.8rem;
}
.summary-box::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%; background: linear-gradient(180deg, #8b5cf6, #ec4899);
    border-radius: 3px 0 0 3px;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 5rem 2rem; border: 1px dashed #1e2433; border-radius: 16px; }
.empty-icon  { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #334155; margin-bottom: 0.3rem; }
.empty-sub   { font-size: 0.82rem; color: #1e2d3d; }
.empty-chips { display: flex; justify-content: center; gap: 0.5rem; margin-top: 0.9rem; }
.chip { background: #0f1219; border: 1px solid #1e2433; border-radius: 999px; padding: 0.22rem 0.7rem; font-size: 0.72rem; color: #334155; }

/* ── Progress ── */
.stProgress > div > div { background: #1d4ed8 !important; border-radius: 4px; }
.stProgress > div { background: #1e2433 !important; border-radius: 4px; }

hr { border-color: #1e2433; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Session defaults ──────────────────────────────────────────────────────────
for key, val in {
    "user": None, "session": None,
    "ready": None, "dense_retriever": None, "bm25_retriever": None,
    "doc_count": None, "source_label": None, "source_type_loaded": None,
    "history": [], "last_answer": None, "last_query": None, "last_sources": None,
    "last_summary": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Restore session from Supabase token ──────────────────────────────────────
if not st.session_state.user:
    try:
        stored = st.session_state.get("_token")
        if stored:
            sb = __import__("auth.supabase_client", fromlist=["get_supabase"]).get_supabase()
            res = sb.auth.get_user(stored)
            if res.user:
                st.session_state.user = res.user
    except Exception:
        pass


# ── Models (cached) ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return EmbeddingModel(), Reranker(), AnswerGenerator()

embedder, reranker, llm = load_models()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.user:
    st.markdown("<div class='auth-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-logo'>🧠</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-title'>MultiRAG</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-sub'>Sign in to your account to continue</div>", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

    with tab_login:
        email    = st.text_input("Email", key="li_email", placeholder="you@example.com")
        password = st.text_input("Password", key="li_pass", type="password", placeholder="••••••••")
        if st.button("Sign In", key="btn_login"):
            if email and password:
                user, session = sign_in(email, password)
                if user:
                    st.session_state.user    = user
                    st.session_state.session = session
                    st.session_state.history = fetch_history(user.id, session.access_token)
                    st.rerun()
                else:
                    st.error(f"Login failed: {session}")
            else:
                st.warning("Enter email and password.")

    with tab_signup:
        email2 = st.text_input("Email", key="su_email", placeholder="you@example.com")
        pass2  = st.text_input("Password", key="su_pass", type="password", placeholder="Min 6 characters")
        if st.button("Create Account", key="btn_signup"):
            if email2 and pass2:
                ok, msg = sign_up(email2, pass2)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("Fill in all fields.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP  (only reached when logged in)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class='brand'>
        <div style='font-size:1.5rem'>🧠</div>
        <div>
            <div class='brand-name'>MultiRAG</div>
            <div class='brand-sub'>{st.session_state.user.email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Source loader
    st.markdown("<div class='section-label'>Source Type</div>", unsafe_allow_html=True)
    source_type = st.selectbox("Source Type", ["YouTube", "PDF", "Image"], label_visibility="collapsed")
    st.markdown("<div class='section-label'>Load Content</div>", unsafe_allow_html=True)

    documents = []

    if source_type == "YouTube":
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
        if st.button("▶  Load Video"):
            if url:
                try:
                    with st.spinner("Fetching transcript..."):
                        documents = load_youtube_transcript(url)
                    st.session_state.source_label = f"YouTube · {url[-11:]}"
                    st.session_state.source_type_loaded = "youtube"
                    st.session_state.last_answer = None
                    st.session_state.last_summary = None
                except Exception as e:
                    st.error(f"Failed: {e}")
            else:
                st.warning("Enter a YouTube URL first.")

    elif source_type == "PDF":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(pdf_file.read()); tmp_path = f.name
            documents = load_pdf(tmp_path)
            st.session_state.source_label = f"PDF · {pdf_file.name}"
            st.session_state.source_type_loaded = "pdf"
            st.session_state.last_answer = None
            st.session_state.last_summary = None

    elif source_type == "Image":
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if img_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                f.write(img_file.read()); tmp_path = f.name
            documents = load_image(tmp_path)
            st.session_state.source_label = f"Image · {img_file.name}"
            st.session_state.source_type_loaded = "image"
            st.session_state.last_answer = None
            st.session_state.last_summary = None

    st.markdown("---")

    # Session status
    st.markdown("<div class='section-label'>Session</div>", unsafe_allow_html=True)
    status_html = (
        "<span class='status-pill status-ready'><span class='status-dot'></span> Ready</span>"
        if st.session_state.ready else
        "<span class='status-pill status-idle'><span class='status-dot'></span> Idle</span>"
    )
    st.markdown(status_html, unsafe_allow_html=True)
    if st.session_state.source_label:
        st.markdown(f"<div style='color:#334155;font-size:0.76rem;margin-top:0.5rem'>📄 {st.session_state.source_label}</div>", unsafe_allow_html=True)
    if st.session_state.doc_count:
        st.markdown(f"<div style='color:#334155;font-size:0.76rem;margin-top:0.2rem'>🗂 {st.session_state.doc_count} chunks</div>", unsafe_allow_html=True)

    st.markdown("---")

    # History
    st.markdown("<div class='section-label'>Chat History</div>", unsafe_allow_html=True)
    if st.session_state.history:
        for item in st.session_state.history[:15]:
            q   = item.get("question", "")
            src = item.get("source_label", "")
            st.markdown(f"""
            <div class='hist-item'>
                <div class='hist-q'>💬 {q[:55]}{"…" if len(q) > 55 else ""}</div>
                <div class='hist-src'>{src}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#1e2d3d;font-size:0.78rem'>No history yet</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Sign Out"):
        sign_out()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>Multimodal RAG</div>
    <h1>MultiRAG</h1>
    <p>Ask questions from YouTube videos, PDFs, and Images using hybrid AI-powered retrieval</p>
</div>
""", unsafe_allow_html=True)


# ── Quick Summary ─────────────────────────────────────────────────────────────
if documents:
    st.markdown("<div class='qa-label'>⚡ Quick Summary</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#334155;font-size:0.8rem;margin-bottom:0.7rem'>Instant overview before building the full pipeline</div>", unsafe_allow_html=True)
    if st.button("📋  Generate Summary"):
        with st.spinner("Generating summary..."):
            full_text = " ".join(doc.content for doc in documents)[:6000]
            st.session_state.last_summary = llm.client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": f"Provide a detailed and comprehensive summary of the following content:\n\n{full_text}"}],
                max_tokens=1024
            ).choices[0].message.content
    if st.session_state.last_summary:
        st.markdown(f"""
        <div class='answer-header'><div class='answer-header-dot'></div> Summary</div>
        <div class='summary-box'>{st.session_state.last_summary}</div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='margin:1.4rem 0'>", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
if documents:
    st.markdown("<div class='qa-label'>⚙ Indexing Pipeline</div>", unsafe_allow_html=True)
    progress    = st.progress(0)
    status_text = st.empty()

    for msg, pct, fn in [
        ("Cleaning documents...",       15, lambda: clean_documents(documents)),
        ("Chunking into windows...",    30, None),
        ("Generating embeddings...",    55, None),
        ("Storing in vector database...", 75, None),
        ("Initialising retrievers...", 90, None),
    ]:
        status_text.markdown(f"<span style='color:#334155;font-size:0.83rem'>{msg}</span>", unsafe_allow_html=True)
        progress.progress(pct)

    documents  = clean_documents(documents)
    documents  = chunk_documents(documents)
    embeddings = embedder.embed_documents(documents)
    vector_db  = VectorStore()
    vector_db.add_documents(documents, embeddings)
    st.session_state.dense_retriever = DenseRetriever(vector_db, embedder)
    st.session_state.bm25_retriever  = BM25Retriever(documents)
    st.session_state.doc_count       = len(documents)
    st.session_state.ready           = True

    progress.progress(100)
    status_text.markdown("<span style='color:#4ade80;font-size:0.83rem'>✓ Pipeline ready</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:1.4rem 0'>", unsafe_allow_html=True)


# ── Q&A ───────────────────────────────────────────────────────────────────────
if st.session_state.ready:

    if st.session_state.doc_count:
        st.markdown(f"""
        <div class='stats-row'>
            <div class='stat'><div class='stat-value'>{st.session_state.doc_count}</div><div class='stat-label'>Chunks</div></div>
            <div class='stat'><div class='stat-value'>BGE</div><div class='stat-label'>Embeddings</div></div>
            <div class='stat'><div class='stat-value'>BM25</div><div class='stat-label'>Sparse</div></div>
            <div class='stat'><div class='stat-value'>RRF</div><div class='stat-label'>Fusion</div></div>
            <div class='stat'><div class='stat-value'>Llama 3</div><div class='stat-label'>Generator</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='qa-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='qa-label'>💬 Ask a Question</div>", unsafe_allow_html=True)
    query = st.text_input("Question", placeholder="What is this content about?", label_visibility="collapsed")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_clicked = st.button("🔍  Get Answer")
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked and query:
        with st.spinner("Retrieving context and generating answer..."):
            with ThreadPoolExecutor(max_workers=2) as ex:
                fd = ex.submit(st.session_state.dense_retriever.retrieve, query)
                fb = ex.submit(st.session_state.bm25_retriever.retrieve, query)
                dense_results = fd.result()
                bm25_results  = fb.result()

            fused_results = reciprocal_rank_fusion([dense_results, bm25_results])
            final_results = reranker.rerank(query, fused_results)
            answer        = llm.generate_answer(query, final_results)

        st.session_state.last_query   = query
        st.session_state.last_answer  = answer
        st.session_state.last_sources = final_results
        try:
            save_history(st.session_state.user.id, query, answer, st.session_state.source_label, st.session_state.session.access_token)
            st.session_state.history = fetch_history(st.session_state.user.id, st.session_state.session.access_token)
        except Exception:
            pass

    # Display last answer
    if st.session_state.last_answer:
        st.markdown(f"""
        <div class='answer-header'><div class='answer-header-dot'></div> Answer</div>
        <div class='answer-box'>{st.session_state.last_answer}</div>
        """, unsafe_allow_html=True)

        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, doc in enumerate(st.session_state.last_sources, 1):
                src   = doc.metadata.get("source", "unknown").lower()
                extra = doc.metadata.get("page", doc.metadata.get("video_id", ""))
                tc    = {"youtube": "type-youtube", "pdf": "type-pdf", "image": "type-image"}.get(src, "type-unknown")
                ex_html = f"<span style='color:#334155;font-size:0.7rem'>· {extra}</span>" if extra else ""
                st.markdown(f"""
                <div class='source-card'>
                    <div class='source-meta'>
                        <div class='source-num'>{i}</div>
                        <span class='source-type {tc}'>{src.upper()}</span>
                        {ex_html}
                    </div>
                    <div class='source-text'>{doc.content[:350]}{"..." if len(doc.content) > 350 else ""}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-icon'>🧠</div>
        <div class='empty-title'>No content loaded yet</div>
        <div class='empty-sub'>Load a data source from the sidebar to get started</div>
        <div class='empty-chips'>
            <span class='chip'>🎥 YouTube</span>
            <span class='chip'>📄 PDF</span>
            <span class='chip'>🖼 Image</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
