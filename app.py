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
    st.title("📚 RAG Pipeline")
    st.caption("Hand-built — no LangChain, no LlamaIndex")
    st.divider()

    # --- Document upload ---
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader(
        "PDF or TXT file",
        type=["pdf", "txt", "md"],
        help="Upload the document you want to query.",
    )

    index_btn = st.button(
        "⚡ Index Document",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

    if index_btn and uploaded_file is not None:
        with st.spinner("Ingesting and indexing…"):
            try:
                from src.bm25 import BM25
                from src.chunker import RecursiveTextChunker
                from src.ingestion import load_document
                from src.retriever import HybridRetriever
                from src.vectorstore import VectorStore

                # Save uploaded file to a temp location
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Ingest
                doc = load_document(tmp_path)
                os.unlink(tmp_path)  # clean up temp file

                # Chunk
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
                embedder = load_embedding_model()
                texts = [c["text"] for c in chunks]
                embeddings = embedder.embed_texts(texts, show_progress=False)

                # Index: VectorStore + BM25
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

                st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    # Index status
    if st.session_state["indexed_filename"]:
        st.info(f"**Indexed:** {st.session_state['indexed_filename']}\n\n**Chunks:** {st.session_state['chunk_count']}")

    st.divider()

    # --- LLM Settings ---
    st.subheader("2. LLM Settings")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-…",
        help="OpenAI key or any compatible provider key.",
    )
    base_url = st.text_input(
        "API Base URL",
        value="https://api.openai.com/v1",
        help="Change to use Groq, Together, Ollama, etc.",
    )
    model = st.text_input(
        "Model",
        value="gpt-4o-mini",
        help="Model identifier for the chosen provider.",
    )

    st.divider()

    # --- Retrieval settings ---
    st.subheader("3. Retrieval Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    st.divider()
    st.caption(
        "Built from scratch · "
        "[GitHub](https://github.com/Archit-Konde/RAG) · "
        "[HF Spaces](https://huggingface.co/spaces/architechs/RAG)"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("RAG Pipeline Demo")
st.caption(
    "Retrieval-Augmented Generation built from scratch — "
    "recursive chunking · BM25 + dense hybrid retrieval · "
    "cross-encoder reranking · raw API call"
)

# --- Query input ---
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question about your document",
        placeholder="What is the main topic of this document?",
        label_visibility="collapsed",
    )
with col2:
    submit = st.button(
        "Ask ▶",
        use_container_width=True,
        disabled=(st.session_state["retriever"] is None or not api_key.strip() or not query.strip()),
    )

if st.session_state["retriever"] is None:
    st.info("Upload and index a document using the sidebar to get started.")

if not api_key.strip() and st.session_state["retriever"] is not None:
    st.warning("Enter your API key in the sidebar to enable answer generation.")

# --- Run pipeline on submit ---
if submit and query.strip() and st.session_state["retriever"] is not None:
    with st.spinner("Retrieving and generating…"):
        try:
            from src.generator import LLMGenerator

            # Retrieve
            retriever = st.session_state["retriever"]
            retrieved = retriever.retrieve(query, top_k=top_k)

            # Rerank
            reranker = load_reranker()
            reranked = reranker.rerank(query, retrieved, top_k=top_k)

            # Generate
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

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")

# --- Render chat history ---
for entry in st.session_state["chat_history"]:
    q = entry["query"]
    res = entry["result"]

    st.markdown("---")
    st.markdown(f"### Q: {q}")

    # Answer
    st.markdown(res["answer"])

    # Retrieved chunks
    with st.expander(f"📄 Retrieved chunks ({len(res.get('chunks', []))})"):
        for i, chunk in enumerate(res.get("chunks", []), start=1):
            rerank_score = chunk.get("rerank_score", chunk.get("score", 0))
            meta = chunk.get("metadata", {})
            filename = source_basename(meta)
            st.markdown(
                f"**[Source {i}]** `{filename}` · chunk `{meta.get('chunk_index', '?')}` · "
                f"rerank score: `{rerank_score:.4f}`"
            )
            text = chunk["text"]
            preview = text[:600] + "…" if len(text) > 600 else text
            st.text(preview)
            st.divider()

    # Sources table
    with st.expander("📎 Sources"):
        if res.get("sources"):
            import pandas as pd

            st.dataframe(
                pd.DataFrame(res["sources"]),
                use_container_width=True,
                hide_index=True,
            )

    # Token usage
    with st.expander("📊 Token usage"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prompt tokens", res.get("prompt_tokens", 0))
        col_b.metric("Completion tokens", res.get("completion_tokens", 0))
        col_c.metric(
            "Total tokens",
            res.get("prompt_tokens", 0) + res.get("completion_tokens", 0),
        )
        st.caption(f"Model: `{res.get('model', model)}`")
