# """
# RAG Pipeline Demo — Streamlit application.

# Wires together every src/ component into an end-to-end pipeline:
#   Upload → Ingest → Chunk → Embed → Index → Ask → Retrieve → Rerank → Generate

# Compatible with HuggingFace Spaces (all file I/O uses /tmp via tempfile,
# API key entered in the sidebar UI — no .env required on Spaces).
# """

# from __future__ import annotations

# import os
# import tempfile
# from pathlib import Path

# import streamlit as st

# from src.utils import source_basename

# # ---------------------------------------------------------------------------
# # Page config (must be first Streamlit call)
# # ---------------------------------------------------------------------------
# st.set_page_config(
#     page_title="RAG Pipeline Demo",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------------------------------------------------------------------------
# # Lazy imports — only import ML libraries after page config so Streamlit
# # can render the UI immediately while models load in the background.
# # ---------------------------------------------------------------------------


# @st.cache_resource(show_spinner="Loading embedding model (~90MB)…")
# def load_embedding_model():
#     from src.embeddings import EmbeddingModel

#     return EmbeddingModel()


# @st.cache_resource(show_spinner="Loading cross-encoder (~85MB)…")
# def load_reranker():
#     from src.reranker import CrossEncoderReranker

#     return CrossEncoderReranker()


# # ---------------------------------------------------------------------------
# # Session state initialization
# # ---------------------------------------------------------------------------


# def init_session_state() -> None:
#     defaults = {
#         "vectorstore": None,
#         "bm25": None,
#         "retriever": None,
#         "indexed_filename": None,
#         "chunk_count": 0,
#         "chat_history": [],
#     }
#     for key, val in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = val


# init_session_state()

# # ---------------------------------------------------------------------------
# # Sidebar
# # ---------------------------------------------------------------------------

# with st.sidebar:
#     st.title("📚 RAG Pipeline")
#     st.caption("Hand-built — no LangChain, no LlamaIndex")
#     st.divider()

#     # --- Document upload ---
#     st.subheader("1. Upload Document")
#     uploaded_file = st.file_uploader(
#         "PDF or TXT file",
#         type=["pdf", "txt", "md"],
#         help="Upload the document you want to query.",
#     )

#     index_btn = st.button(
#         "⚡ Index Document",
#         disabled=(uploaded_file is None),
#         use_container_width=True,
#     )

#     if index_btn and uploaded_file is not None:
#         with st.spinner("Ingesting and indexing…"):
#             try:
#                 from src.bm25 import BM25
#                 from src.chunker import RecursiveTextChunker
#                 from src.ingestion import load_document
#                 from src.retriever import HybridRetriever
#                 from src.vectorstore import VectorStore

#                 # Save uploaded file to a temp location
#                 suffix = Path(uploaded_file.name).suffix
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                     tmp.write(uploaded_file.read())
#                     tmp_path = tmp.name

#                 # Ingest
#                 doc = load_document(tmp_path)
#                 os.unlink(tmp_path)  # clean up temp file

#                 # Chunk
#                 chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=64)
#                 chunks = chunker.split_text(doc["text"])

#                 # Attach source metadata to every chunk
#                 for chunk in chunks:
#                     chunk["metadata"] = {
#                         "source": doc["metadata"]["source"],
#                         "filename": doc["filename"],
#                         "chunk_index": chunk["chunk_index"],
#                         "file_type": doc["metadata"]["file_type"],
#                     }

#                 # Embed
#                 embedder = load_embedding_model()
#                 texts = [c["text"] for c in chunks]
#                 embeddings = embedder.embed_texts(texts, show_progress=False)

#                 # Index: VectorStore + BM25
#                 vs = VectorStore()
#                 vs.add(embeddings, texts, [c["metadata"] for c in chunks])

#                 bm = BM25()
#                 bm.fit(texts)

#                 # Hybrid retriever
#                 retriever = HybridRetriever(vs, bm, embedder)

#                 # Store in session
#                 st.session_state["vectorstore"] = vs
#                 st.session_state["bm25"] = bm
#                 st.session_state["retriever"] = retriever
#                 st.session_state["indexed_filename"] = uploaded_file.name
#                 st.session_state["chunk_count"] = len(chunks)
#                 st.session_state["chat_history"] = []  # reset on new doc

#                 st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

#             except Exception as exc:
#                 st.error(f"Indexing failed: {exc}")

#     # Index status
#     if st.session_state["indexed_filename"]:
#         st.info(f"**Indexed:** {st.session_state['indexed_filename']}\n\n**Chunks:** {st.session_state['chunk_count']}")

#     st.divider()

#     # --- LLM Settings ---
#     # --- LLM Settings ---
#     st.subheader("2. LLM Settings")

#     api_key = st.text_input(
#     "Groq API Key",
#     type="password",
#     placeholder="gsk_...",
#     )

#     base_url = st.text_input(
#         "API Base URL",
#         value="https://api.groq.com/openai/v1",
#     )

#     model = st.text_input(
#         "Groq Model",
#         value="openai/gpt-oss-120b",
#     )


#     st.divider()

#     # --- Retrieval settings ---
#     st.subheader("3. Retrieval Settings")
#     top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

#     st.divider()
#     st.caption(
#         "Built from scratch · "
#         "[GitHub](https://github.com/Archit-Konde/RAG) · "
#         "[HF Spaces](https://huggingface.co/spaces/architechs/RAG)"
#     )

# # ---------------------------------------------------------------------------
# # Main area
# # ---------------------------------------------------------------------------

# st.title("RAG Pipeline Demo")
# st.caption(
#     "Retrieval-Augmented Generation built from scratch — "
#     "recursive chunking · BM25 + dense hybrid retrieval · "
#     "cross-encoder reranking · raw API call"
# )

# # --- Query input ---
# col1, col2 = st.columns([5, 1])
# with col1:
#     query = st.text_input(
#         "Ask a question about your document",
#         placeholder="What is the main topic of this document?",
#         label_visibility="collapsed",
#     )
# with col2:
#     submit = st.button(
#         "Ask ▶",
#         use_container_width=True,
#         disabled=(st.session_state["retriever"] is None or not api_key.strip() or not query.strip()),
#     )

# if st.session_state["retriever"] is None:
#     st.info("Upload and index a document using the sidebar to get started.")

# if not api_key.strip() and st.session_state["retriever"] is not None:
#     st.warning("Enter your API key in the sidebar to enable answer generation.")

# # --- Run pipeline on submit ---
# if submit and query.strip() and st.session_state["retriever"] is not None:
#     with st.spinner("Retrieving and generating…"):
#         try:
#             from src.generator import LLMGenerator

#             # Retrieve
#             retriever = st.session_state["retriever"]
#             retrieved = retriever.retrieve(query, top_k=top_k)

#             # Rerank
#             reranker = load_reranker()
#             reranked = reranker.rerank(query, retrieved, top_k=top_k)

#             # Generate
#             generator = LLMGenerator(
#                 api_key=api_key.strip(),
#                 base_url=base_url.strip(),
#                 model=model.strip(),
#             )
#             result = generator.generate(query, reranked)
#             result["chunks"] = reranked

#             # Prepend to history so newest is at top
#             st.session_state["chat_history"].insert(
#                 0,
#                 {
#                     "query": query,
#                     "result": result,
#                 },
#             )

#         except Exception as exc:
#             st.error(f"Pipeline error: {exc}")

# # --- Render chat history ---
# for entry in st.session_state["chat_history"]:
#     q = entry["query"]
#     res = entry["result"]

#     st.markdown("---")
#     st.markdown(f"### Q: {q}")

#     # Answer
#     st.markdown(res["answer"])

#     # Retrieved chunks
#     with st.expander(f"📄 Retrieved chunks ({len(res.get('chunks', []))})"):
#         for i, chunk in enumerate(res.get("chunks", []), start=1):
#             rerank_score = chunk.get("rerank_score", chunk.get("score", 0))
#             meta = chunk.get("metadata", {})
#             filename = source_basename(meta)
#             st.markdown(
#                 f"**[Source {i}]** `{filename}` · chunk `{meta.get('chunk_index', '?')}` · "
#                 f"rerank score: `{rerank_score:.4f}`"
#             )
#             text = chunk["text"]
#             preview = text[:600] + "…" if len(text) > 600 else text
#             st.text(preview)
#             st.divider()

#     # Sources table
#     with st.expander("📎 Sources"):
#         if res.get("sources"):
#             import pandas as pd

#             st.dataframe(
#                 pd.DataFrame(res["sources"]),
#                 use_container_width=True,
#                 hide_index=True,
#             )

#     # Token usage
#     with st.expander("📊 Token usage"):
#         col_a, col_b, col_c = st.columns(3)
#         col_a.metric("Prompt tokens", res.get("prompt_tokens", 0))
#         col_b.metric("Completion tokens", res.get("completion_tokens", 0))
#         col_c.metric(
#             "Total tokens",
#             res.get("prompt_tokens", 0) + res.get("completion_tokens", 0),
#         )
#         st.caption(f"Model: `{res.get('model', model)}`")


"""
RAG Pipeline Demo — Streamlit application.

Wires together every src/ component into an end-to-end pipeline:
  Upload → Ingest → Chunk → Embed → Index → Ask → Retrieve → Rerank → Generate

Compatible with HuggingFace Spaces (all file I/O uses /tmp via tempfile,
API key entered in the sidebar UI — no .env required on Spaces).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.utils import source_basename

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* ---------- General ---------- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    #MainMenu, footer {visibility: hidden;}

    /* ---------- Hero header ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #C026D3 100%);
        border-radius: 18px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.75rem;
        color: white;
        box-shadow: 0 10px 30px -12px rgba(79, 70, 229, 0.45);
    }
    .hero-banner h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        color: white;
    }
    .hero-banner p {
        margin: 0;
        font-size: 0.95rem;
        opacity: 0.92;
    }
    .hero-badges {
        margin-top: 0.9rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 10px;
        font-weight: 700;
        height: 2.7rem;
    }
    .sidebar-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }
    .sidebar-step .num {
        background: #7C3AED;
        color: white;
        border-radius: 999px;
        width: 22px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
    }

    /* ---------- Status card ---------- */
    .status-card {
        border: 1px solid rgba(124, 58, 237, 0.35);
        background: rgba(124, 58, 237, 0.08);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .status-card b { color: #C084FC; }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 3.5rem 1.5rem;
        border: 1.5px dashed rgba(120,120,140,0.35);
        border-radius: 16px;
        background: rgba(124, 58, 237, 0.03);
    }
    .empty-state .icon { font-size: 2.4rem; margin-bottom: 0.5rem; }
    .empty-state h3 { margin: 0 0 0.35rem 0; }
    .empty-state p { color: rgba(160,160,180,0.9); margin: 0; }

    /* ---------- Chat entries ---------- */
    .qa-card {
        border-radius: 16px;
        border: 1px solid rgba(120,120,140,0.18);
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.4rem;
        background: rgba(255,255,255,0.015);
    }
    .qa-question {
        display: flex;
        gap: 0.6rem;
        align-items: flex-start;
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 0.85rem;
    }
    .qa-question .bubble {
        background: #4F46E5;
        color: white;
        border-radius: 999px;
        min-width: 26px;
        height: 26px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    .qa-answer {
        padding-left: 2.15rem;
        line-height: 1.65;
    }
    .source-chip {
        display: inline-block;
        background: rgba(192, 132, 252, 0.12);
        border: 1px solid rgba(192, 132, 252, 0.3);
        color: #C084FC;
        border-radius: 8px;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: rgba(124, 58, 237, 0.06);
        border: 1px solid rgba(124, 58, 237, 0.18);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy imports — only import ML libraries after page config so Streamlit
# can render the UI immediately while models load in the background.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model (~90MB)…")
def load_embedding_model():
    from src.embeddings import EmbeddingModel

    return EmbeddingModel()


@st.cache_resource(show_spinner="Loading cross-encoder (~85MB)…")
def load_reranker():
    from src.reranker import CrossEncoderReranker

    return CrossEncoderReranker()


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    defaults = {
        "vectorstore": None,
        "bm25": None,
        "retriever": None,
        "indexed_filename": None,
        "chunk_count": 0,
        "chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.1rem;">
            <span style="font-size:1.6rem;">📚</span>
            <span style="font-size:1.25rem;font-weight:800;">RAG Pipeline</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Hand-built · no LangChain, no LlamaIndex")
    st.divider()

    # --- Document upload ---
    st.markdown('<div class="sidebar-step"><span class="num">1</span> Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF or TXT file",
        type=["pdf", "txt", "md"],
        help="Upload the document you want to query.",
        label_visibility="collapsed",
    )

    index_btn = st.button(
        "⚡ Index Document",
        disabled=(uploaded_file is None),
        use_container_width=True,
        type="primary",
    )

    if index_btn and uploaded_file is not None:
        progress = st.progress(0, text="Starting…")
        try:
            from src.bm25 import BM25
            from src.chunker import RecursiveTextChunker
            from src.ingestion import load_document
            from src.retriever import HybridRetriever
            from src.vectorstore import VectorStore

            # Save uploaded file to a temp location
            progress.progress(10, text="Reading uploaded file…")
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Ingest
            progress.progress(25, text="Parsing document…")
            doc = load_document(tmp_path)
            os.unlink(tmp_path)  # clean up temp file

            # Chunk
            progress.progress(40, text="Chunking text…")
            chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=64)
            chunks = chunker.split_text(doc["text"])

            # Attach source metadata to every chunk
            for chunk in chunks:
                chunk["metadata"] = {
                    "source": doc["metadata"]["source"],
                    "filename": doc["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "file_type": doc["metadata"]["file_type"],
                }

            # Embed
            progress.progress(60, text="Embedding chunks…")
            embedder = load_embedding_model()
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(texts, show_progress=False)

            # Index: VectorStore + BM25
            progress.progress(85, text="Building indexes…")
            vs = VectorStore()
            vs.add(embeddings, texts, [c["metadata"] for c in chunks])

            bm = BM25()
            bm.fit(texts)

            # Hybrid retriever
            retriever = HybridRetriever(vs, bm, embedder)

            # Store in session
            st.session_state["vectorstore"] = vs
            st.session_state["bm25"] = bm
            st.session_state["retriever"] = retriever
            st.session_state["indexed_filename"] = uploaded_file.name
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["chat_history"] = []  # reset on new doc

            progress.progress(100, text="Done!")
            progress.empty()
            st.success(f"Indexed **{len(chunks)}** chunks from **{uploaded_file.name}**", icon="✅")

        except Exception as exc:
            progress.empty()
            st.error(f"Indexing failed: {exc}", icon="🚨")

    # Index status
    if st.session_state["indexed_filename"]:
        st.markdown(
            f"""
            <div class="status-card">
                📄 <b>Indexed:</b> {st.session_state['indexed_filename']}<br/>
                🧩 <b>Chunks:</b> {st.session_state['chunk_count']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No document indexed yet.")

    st.divider()

    # --- LLM Settings ---
    st.markdown('<div class="sidebar-step"><span class="num">2</span> LLM Settings</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Groq API configuration", expanded=not bool(os.environ.get("GROQ_API_KEY"))):
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
        )

        base_url = st.text_input(
            "API Base URL",
            value="https://api.groq.com/openai/v1",
        )

        model = st.text_input(
            "Groq Model",
            value="openai/gpt-oss-120b",
        )

    if api_key.strip():
        st.caption("🔑 API key set for this session")
    else:
        st.caption("🔒 No API key entered yet")

    st.divider()

    # --- Retrieval settings ---
    st.markdown('<div class="sidebar-step"><span class="num">3</span> Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    st.divider()
    st.caption(
        "Built from scratch · "
        "[GitHub](https://github.com/Kshubham0315/Rag_From_Scratch.git) · "
    )

# ---------------------------------------------------------------------------
# Main area — hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>📚 RAG Pipeline Demo</h1>
        <p>Retrieval-Augmented Generation built from scratch — no LangChain, no LlamaIndex.</p>
        <div class="hero-badges">
            <span class="hero-badge">🔀 Recursive chunking</span>
            <span class="hero-badge">🔎 BM25 + dense hybrid retrieval</span>
            <span class="hero-badge">🎯 Cross-encoder reranking</span>
            <span class="hero-badge">⚡ Raw API call</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Pipeline status strip ---
status_cols = st.columns(4)
pipeline_steps = [
    ("📤", "Upload", st.session_state["indexed_filename"] is not None),
    ("✂️", "Chunk", st.session_state["chunk_count"] > 0),
    ("🧠", "Index", st.session_state["retriever"] is not None),
    ("🔑", "API Key", False),  # placeholder, updated below
]

for col, (icon, label, done) in zip(status_cols, pipeline_steps):
    with col:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 0.6rem 0.3rem; border-radius: 10px;
                        background: {'rgba(34,197,94,0.10)' if done else 'rgba(120,120,140,0.06)'};
                        border: 1px solid {'rgba(34,197,94,0.35)' if done else 'rgba(120,120,140,0.15)'};">
                <div style="font-size:1.3rem;">{icon}</div>
                <div style="font-size:0.8rem;font-weight:700;">{label}</div>
                <div style="font-size:0.72rem;opacity:0.75;">{'✅ Ready' if done else '⏳ Pending'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# --- Query input ---
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question about your document",
        placeholder="💬 What is the main topic of this document?",
        label_visibility="collapsed",
    )
with col2:
    submit = st.button(
        "Ask ▶",
        use_container_width=True,
        type="primary",
        disabled=(st.session_state["retriever"] is None or not api_key.strip() or not query.strip()),
    )

if st.session_state["retriever"] is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📥</div>
            <h3>Upload a document to get started</h3>
            <p>Use the sidebar to upload a PDF, TXT, or Markdown file and index it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not api_key.strip() and st.session_state["retriever"] is not None:
    st.warning("Enter your Groq API key in the sidebar to enable answer generation.", icon="🔑")

# --- Run pipeline on submit ---
if submit and query.strip() and st.session_state["retriever"] is not None:
    with st.status("Running RAG pipeline…", expanded=True) as status:
        try:
            from src.generator import LLMGenerator

            st.write("🔎 Retrieving candidate chunks…")
            retriever = st.session_state["retriever"]
            retrieved = retriever.retrieve(query, top_k=top_k)

            st.write("🎯 Reranking with cross-encoder…")
            reranker = load_reranker()
            reranked = reranker.rerank(query, retrieved, top_k=top_k)

            st.write("✍️ Generating answer…")
            generator = LLMGenerator(
                api_key=api_key.strip(),
                base_url=base_url.strip(),
                model=model.strip(),
            )
            result = generator.generate(query, reranked)
            result["chunks"] = reranked

            # Prepend to history so newest is at top
            st.session_state["chat_history"].insert(
                0,
                {
                    "query": query,
                    "result": result,
                },
            )
            status.update(label="Done!", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Pipeline failed", state="error", expanded=True)
            st.error(f"Pipeline error: {exc}", icon="🚨")

# --- Render chat history ---
if st.session_state["chat_history"]:
    st.markdown("### 🗂️ History")

for entry in st.session_state["chat_history"]:
    q = entry["query"]
    res = entry["result"]

    st.markdown(
        f"""
        <div class="qa-card">
            <div class="qa-question"><span class="bubble">Q</span>{q}</div>
            <div class="qa-answer">{res["answer"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_chunks, tab_sources, tab_usage = st.tabs(
        [f"📄 Retrieved chunks ({len(res.get('chunks', []))})", "📎 Sources", "📊 Token usage"]
    )

    with tab_chunks:
        for i, chunk in enumerate(res.get("chunks", []), start=1):
            rerank_score = chunk.get("rerank_score", chunk.get("score", 0))
            meta = chunk.get("metadata", {})
            filename = source_basename(meta)
            st.markdown(
                f'<span class="source-chip">Source {i}</span>'
                f'<span class="source-chip">{filename}</span>'
                f'<span class="source-chip">chunk {meta.get("chunk_index", "?")}</span>'
                f'<span class="source-chip">score {rerank_score:.4f}</span>',
                unsafe_allow_html=True,
            )
            text = chunk["text"]
            preview = text[:600] + "…" if len(text) > 600 else text
            st.text(preview)
            st.divider()

    with tab_sources:
        if res.get("sources"):
            import pandas as pd

            st.dataframe(
                pd.DataFrame(res["sources"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No source metadata available.")

    with tab_usage:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prompt tokens", res.get("prompt_tokens", 0))
        col_b.metric("Completion tokens", res.get("completion_tokens", 0))
        col_c.metric(
            "Total tokens",
            res.get("prompt_tokens", 0) + res.get("completion_tokens", 0),
        )
        st.caption(f"Model: `{res.get('model', model)}`")