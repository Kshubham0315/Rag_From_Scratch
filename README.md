# RAG From Scratch

A modular Retrieval-Augmented Generation (RAG) system implemented from scratch in Python — without LangChain or LlamaIndex.

The project focuses on understanding the internal working of a modern RAG pipeline by implementing the major retrieval algorithms manually.

## Features

* PDF and TXT document ingestion
* Recursive text chunking
* Overlapping chunks
* Transformer-based text embeddings
* Manual mean pooling
* L2 normalization
* NumPy-based vector store
* Cosine similarity search
* BM25 sparse retrieval
* Hybrid dense + sparse retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Grounded LLM generation
* Source attribution
* Retrieval evaluation
* Streamlit demo
* Unit tests

## Architecture

```text
                    INDEXING
                       │
          ┌────────────┴────────────┐
          │                         │
      PDF / TXT                 Documents
          │                         │
          ▼                         │
      Ingestion                     │
          │                         │
          ▼                         │
       Chunking                     │
          │                         │
          ▼                         │
     ┌────┴────┐                    │
     │         │                    │
     ▼         ▼                    │
 Embeddings   BM25                   │
     │         │                    │
     ▼         ▼                    │
 VectorStore  BM25 Index             │
     │         │                    │
     └────┬────┘                    │
          │
          │
          ▼
                    QUERY
                       │
                       ▼
                  User Query
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        Dense Search        BM25 Search
             │                   │
             └─────────┬─────────┘
                       ▼
                    RRF Fusion
                       │
                       ▼
                    Reranker
                       │
                       ▼
                 Top-K Context
                       │
                       ▼
                    LLM
                       │
                       ▼
                Answer + Sources
```

## Pipeline

The complete pipeline consists of two stages.

### 1. Indexing

```text
Documents
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Index
```

At the same time, the chunks are indexed using BM25 for lexical retrieval.

```text
Chunks
   ↓
BM25
   ↓
Sparse Index
```

### 2. Query

```text
User Query
    │
    ├───────────────┐
    │               │
    ▼               ▼
Embedding         BM25
    │               │
    ▼               ▼
Vector Search    Sparse Search
    │               │
    └───────┬───────┘
            ▼
          RRF
            │
            ▼
        Reranking
            │
            ▼
      Relevant Context
            │
            ▼
           LLM
            │
            ▼
      Answer + Sources
```

## Retrieval Methods

### Dense Retrieval

Dense retrieval converts both documents and queries into vector representations.

The vector store uses L2-normalized embeddings, allowing cosine similarity to be calculated using a dot product.

```text
cosine_similarity(A, B) = A · B
```

when both vectors are L2 normalized.

### BM25

BM25 provides sparse lexical retrieval based on term frequency and inverse document frequency.

It is particularly useful when exact terms, names, identifiers, or technical terminology matter.

### Hybrid Retrieval

Dense and sparse retrieval complement each other.

Dense retrieval handles semantic similarity, while BM25 handles exact keyword matching.

The two result lists are combined using Reciprocal Rank Fusion.

```text
RRF(d) = Σ 1 / (k + rank)
```

where:

```text
k = 60
```

### Reranking

The hybrid retriever produces candidate documents.

A cross-encoder then evaluates the query and candidate document together and assigns a relevance score.

```text
Query + Document
       ↓
Cross Encoder
       ↓
Relevance Score
```

This provides more accurate ranking than using embedding similarity alone.

## Models

The default embedding model can be configured through the project configuration.

Example:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The reranker can use an MS-MARCO MiniLM cross-encoder.

The generator is model/provider independent and communicates through an OpenAI-compatible API.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/rag-from-scratch.git
cd rag-from-scratch
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment file:

```bash
copy .env.example .env
```

For Linux/macOS:

```bash
cp .env.example .env
```

Add your LLM API configuration to `.env`.

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit in your browser.

## Adding Documents

Place PDF or TXT files inside:

```text
data/documents/
```

Then run the indexing pipeline:

```bash
python scripts/ingest.py
python scripts/build_index.py
```

The documents will be processed into chunks, embeddings, and retrieval indexes.

## Running Tests

Run all tests:

```bash
pytest tests/ -v
```

Run tests without downloading transformer models:

```bash
pytest tests/ -v \
  --ignore=tests/test_embeddings.py \
  --ignore=tests/test_reranker.py
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Evaluation

The evaluation module supports common retrieval metrics such as:

* Precision@K
* Recall@K
* Mean Reciprocal Rank (MRR)

Example:

```bash
python scripts/run_benchmark.py
```

## Why Build RAG From Scratch?

Frameworks such as LangChain and LlamaIndex make RAG development faster, but they can hide the internal mechanics of retrieval systems.

This project intentionally avoids those abstractions.

The goal is to understand:

* How text chunking works
* How embeddings are generated
* How cosine similarity works
* How vector search works
* How BM25 works
* How hybrid retrieval works
* How RRF combines rankings
* How cross-encoder reranking works
* How retrieved context is passed to an LLM
* How RAG systems can be evaluated

## Design Philosophy

The project follows a simple principle:

```text
Understand the algorithm
        ↓
Implement the algorithm
        ↓
Test the algorithm
        ↓
Integrate the algorithm
```

No retrieval framework is required for the core pipeline.

## Future Improvements

Planned improvements include:

* Metadata filtering
* Persistent vector indexes
* Better document loaders
* Query rewriting
* HyDE retrieval
* Multi-query retrieval
* Context compression
* Citation verification
* RAGAS-based evaluation
* Streaming generation
* Conversation memory
* GPU acceleration
* Medical-domain RAG support

## Learning Resources

Detailed mathematical explanations and implementation notes are available in:

```text
docs/LEARNING.md
```

Topics include:

* TF-IDF
* BM25
* Cosine similarity
* Embeddings
* Mean pooling
* L2 normalization
* Reciprocal Rank Fusion
* Cross-encoder ranking
* Retrieval evaluation

## License

This project is licensed under the MIT License.
