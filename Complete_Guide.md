# Building RAG From Scratch — A Complete Technical Deep-Dive

> A first-principles implementation of Retrieval-Augmented Generation (RAG) using Python, NumPy, PyTorch, and Hugging Face Transformers — without LangChain or LlamaIndex.

This document explains the complete RAG pipeline from the ground up.

The goal is not simply to build a working RAG system, but to understand **what happens inside each component, why each algorithm is used, and how the components work together**.

---

## Table of Contents

1. [RAG Overview](#1-rag-overview)
2. [Recursive Text Chunking](#2-recursive-text-chunking)
3. [Sentence Embeddings](#3-sentence-embeddings)
4. [Vector Store](#4-vector-store)
5. [BM25 Sparse Retrieval](#5-bm25-sparse-retrieval)
6. [Hybrid Retrieval with RRF](#6-hybrid-retrieval-with-rrf)
7. [Cross-Encoder Reranking](#7-cross-encoder-reranking)
8. [Grounded Generation](#8-grounded-generation)
9. [Evaluation](#9-evaluation)
10. [Implementation Decisions](#10-implementation-decisions)
11. [Complete Pipeline](#11-complete-pipeline)

---

# 1. RAG Overview

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with large language models.

Instead of asking an LLM to answer a question entirely from its internal knowledge, RAG first retrieves relevant information from an external knowledge base and provides that information to the model as context.

The basic idea is:

```text
Documents
    ↓
Index
    ↓
User Query
    ↓
Retrieve Relevant Context
    ↓
LLM
    ↓
Grounded Answer
```

## Why do we need RAG?

LLMs have several limitations:

* Their knowledge can become outdated.
* They may not know private or domain-specific documents.
* They can hallucinate information.
* Updating model weights whenever new documents arrive is impractical.

RAG addresses these problems by keeping knowledge outside the model.

For example:

```text
Company Policy PDF
Research Papers
Medical Documents
Product Documentation
Internal Knowledge Base
          ↓
       RAG Index
          ↓
       User Query
          ↓
    Relevant Context
          ↓
         LLM
```

The LLM does not need to memorize the entire knowledge base. It only needs to understand the relevant retrieved context.

---

## RAG Pipeline

The complete system can be divided into two phases.

### Indexing Phase

This happens when documents are added to the system.

```text
PDF / TXT
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Index
   +
BM25 Index
```

### Query Phase

This happens whenever a user asks a question.

```text
User Query
    ↓
 ┌──┴─────────────┐
 ↓                ↓
Dense Search    BM25 Search
 ↓                ↓
 └───────┬────────┘
         ↓
     RRF Fusion
         ↓
      Reranker
         ↓
   Top Relevant Chunks
         ↓
      Prompt
         ↓
        LLM
         ↓
 Answer + Sources
```

---

# 2. Recursive Text Chunking

## Why do we need chunking?

Documents can contain thousands of words or hundreds of pages.

Embedding an entire document as one vector has two major problems:

1. Transformer models have limited input lengths.
2. A single vector representing an entire document loses fine-grained information.

Instead, documents are divided into smaller passages called **chunks**.

```text
Large Document
      ↓
 ┌────┬────┬────┬────┐
 │ C1 │ C2 │ C3 │ C4 │
 └────┴────┴────┴────┘
```

A good chunk should:

* Fit within the embedding model's context window.
* Preserve semantic meaning.
* Contain enough information to answer questions.
* Avoid unnecessary unrelated content.

---

## Why Recursive Splitting?

A simple splitter might always split on paragraphs:

```text
text.split("\n\n")
```

But a paragraph can itself be very large.

A recursive splitter therefore uses a hierarchy of separators:

```text
1. Paragraph
2. Line
3. Sentence
4. Word
5. Character
```

The algorithm tries the most meaningful separator first.

If the resulting piece is still too large, it recursively falls back to the next separator.

```text
Paragraph
   ↓
Too large?
   ↓
Split by lines
   ↓
Still too large?
   ↓
Split by sentences
   ↓
Still too large?
   ↓
Split by words
```

---

## Recursive Splitting Algorithm

Conceptually:

```python
def recursive_split(text, separators):

    separator = separators[0]
    remaining = separators[1:]

    pieces = text.split(separator)

    for piece in pieces:

        if len(piece) <= chunk_size:
            keep(piece)

        else:
            recursive_split(piece, remaining)
```

The important idea is that **the algorithm only moves to a less meaningful separator when necessary**.

---

## Chunk Overlap

Chunks usually overlap slightly.

For example:

```text
Chunk 1
────────────────────────
        │
        │ overlap
        ▼
            Chunk 2
            ────────────────────────
```

Example:

```text
chunk_size = 512
chunk_overlap = 64
```

The last 64 characters of one chunk may become part of the next chunk.

### Why overlap?

Consider:

```text
Chunk 1:
"The patient has a history of high blood pressure and..."

Chunk 2:
"...requires regular monitoring of blood pressure."
```

Without overlap, important information may be split across two chunks.

Overlap reduces this boundary problem.

---

## Character Position Tracking

Each chunk should retain its position inside the original document.

This is useful for:

* Source attribution
* Debugging
* Highlighting retrieved text
* Citation generation

A forward cursor can be used:

```python
cursor = 0

for chunk in chunks:

    position = original_text.find(
        chunk.text,
        cursor
    )

    chunk.start_char = position
    chunk.end_char = position + len(chunk.text)

    cursor = max(
        cursor,
        position + len(chunk.text) - chunk_overlap
    )
```

The cursor prevents repeatedly searching the entire document.

---

# 3. Sentence Embeddings

## What is an Embedding?

An embedding converts text into a numerical vector.

For example:

```text
"Machine learning is useful"
              ↓
[0.12, -0.42, 0.87, ..., 0.19]
```

A sentence embedding represents the semantic meaning of text in a fixed-dimensional vector space.

Semantically similar sentences should produce vectors that are close to each other.

---

## Model

This implementation can use:

```text
all-MiniLM-L6-v2
```

The model produces:

```text
384-dimensional embeddings
```

The embedding process is:

```text
Text
 ↓
Tokenizer
 ↓
Transformer
 ↓
Token Embeddings
 ↓
Mean Pooling
 ↓
L2 Normalization
 ↓
Sentence Embedding
```

---

# Mean Pooling

The transformer produces a vector for every token.

Suppose:

```text
Input:
"The cat is sleeping"
```

The model may produce:

```text
Token 1 → vector
Token 2 → vector
Token 3 → vector
Token 4 → vector
...
```

We need to convert these token-level representations into one vector.

Mean pooling does this by averaging the representations of the actual tokens.

---

## Tensor Shape

The transformer output has shape:

```text
(B, T, D)
```

where:

* `B` = batch size
* `T` = number of tokens
* `D` = hidden dimension

For MiniLM:

```text
D = 384
```

The attention mask identifies real tokens and padding tokens.

```text
1 = real token
0 = padding
```

---

## Mean Pooling Formula

For token embeddings:

```text
h₁, h₂, ..., hₜ
```

the mean representation is:

```text
embedding = Σ hᵢ / T
```

In practice, padding tokens must be excluded.

```python
mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings)

sum_embeddings = torch.sum(
    token_embeddings * mask,
    dim=1
)

sum_mask = torch.clamp(
    mask.sum(dim=1),
    min=1e-9
)

pooled = sum_embeddings / sum_mask
```

The `unsqueeze(-1)` changes:

```text
(B, T)
```

into:

```text
(B, T, 1)
```

allowing the mask to broadcast across the embedding dimension.

---

# L2 Normalization

After mean pooling, embeddings are normalized.

```python
norm = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

norm = np.clip(
    norm,
    1e-12,
    None
)

normalized = embeddings / norm
```

This converts every vector into a unit vector.

That gives us:

```text
||embedding|| = 1
```

This optimization becomes important when calculating cosine similarity.

---

# 4. Vector Store

## Cosine Similarity

For two vectors `a` and `b`:

```text
cosine(a,b)
=
(a · b)
────────────
||a|| ||b||
```

Cosine similarity measures the angle between two vectors.

```text
1.0  → very similar direction
0.0  → perpendicular
-1.0 → opposite direction
```

For normalized embeddings:

```text
||a|| = 1
||b|| = 1
```

Therefore:

```text
cosine(a,b)
=
(a · b)
```

So cosine similarity becomes a simple dot product.

---

## Vector Search

Suppose we have:

```text
N documents
D-dimensional embeddings
```

Stored embeddings have shape:

```text
(N, D)
```

The query embedding has shape:

```text
(D,)
```

We can calculate all similarities with:

```python
scores = embeddings @ query_embedding
```

Result:

```text
(N, D) × (D,)
       ↓
      (N,)
```

Every document receives one similarity score.

---

# Top-K Search

Suppose we have:

```text
50,000 documents
```

and need:

```text
top 5
```

Sorting all 50,000 elements is unnecessary.

Instead:

```python
partition = np.argpartition(
    scores,
    -top_k
)[-top_k:]

top_indices = partition[
    np.argsort(
        scores[partition]
    )[::-1]
]
```

This performs:

```text
Partial selection → O(N)
Final sorting → O(k log k)
```

Instead of fully sorting:

```text
O(N log N)
```

This is efficient for portfolio-scale datasets.

---

# Vector Store Persistence

The vector store can be saved using:

```text
embeddings.npz
metadata.json
```

The NumPy file stores the embedding matrix.

The JSON file stores metadata such as:

```json
{
    "document": "medical_guide.pdf",
    "chunk_index": 12,
    "start_char": 2450,
    "end_char": 2980
}
```

Keeping binary vectors and human-readable metadata separate makes the system easier to inspect and debug.

---

# 5. BM25 Sparse Retrieval

Dense retrieval is powerful, but semantic embeddings are not always ideal for exact terms.

For example:

```text
"Python 3.12.1"
"HTTP 404"
"RTX 4050"
"ICD-10 E11.9"
```

Exact lexical matching can be extremely important.

This is where **BM25** comes in.

---

# TF-IDF

A traditional TF-IDF score is:

```text
TF-IDF(t,d)
=
TF(t,d) × IDF(t)
```

where:

```text
TF = frequency of term
IDF = inverse document frequency
```

A common IDF formulation is:

```text
IDF(t) = log(N / df(t))
```

However, TF-IDF has some limitations.

### Problem 1 — Unbounded term frequency

If a word appears 100 times instead of 10 times, TF increases proportionally.

### Problem 2 — Document length bias

Longer documents naturally contain more occurrences of many terms.

BM25 addresses both problems.

---

# BM25

BM25 introduces two important concepts:

1. Term-frequency saturation
2. Document-length normalization

The full formula is:

```text
                 tf × (k₁ + 1)
BM25(t,d) = IDF × ─────────────────────────────
                 tf + k₁ × (1 - b + b × |d|/avgdl)
```

Typical values:

```text
k₁ = 1.5
b  = 0.75
```

---

## Term Frequency Saturation

BM25 prevents repeated occurrences of the same word from increasing the score indefinitely.

```text
TF component
     │
     │             ________
     │          __/
     │       __/
     │    __/
     │___/
     └──────────────────────
          Term Frequency
```

The score increases quickly at first and then gradually saturates.

---

# Document Length Normalization

BM25 considers document length:

```text
|d| = document length
avgdl = average document length
```

This prevents long documents from automatically receiving higher scores simply because they contain more words.

The `b` parameter controls how strongly length is normalized.

```text
b = 0
```

means no length normalization.

```text
b = 1
```

means full normalization.

A common default is:

```text
b = 0.75
```

---

# BM25 IDF

The Robertson-Walker IDF formulation is:

```text
IDF(t)
=
log(
    (N - df(t) + 0.5)
    ──────────────────
    (df(t) + 0.5)
    + 1
)
```

The smoothing terms prevent extreme values.

The final `+1` inside the logarithm ensures that extremely common terms do not produce problematic negative infinity values.

---

# Tokenization

This implementation intentionally keeps tokenization simple:

```python
text.lower()
```

followed by punctuation removal and:

```python
.split()
```

Advantages:

* Easy to understand
* No external NLP resources
* Easy to debug
* Language-model independent
* Suitable for a first-principles implementation

For more advanced systems, tokenization can later be replaced with NLTK, spaCy, or a custom tokenizer.

---

# 6. Hybrid Retrieval with RRF

## Why Hybrid Retrieval?

Dense and sparse retrieval solve different problems.

### Dense Retrieval

Good at semantic similarity.

```text
"furry household companion"
```

can retrieve:

```text
"The cat sat on the mat."
```

even though the words are different.

### BM25

Good at exact matching.

```text
"PyTorch 2.1"
```

should strongly prefer documents containing:

```text
"PyTorch 2.1"
```

The best system combines both.

---

# Why Not Average Scores?

A simple approach would be:

```text
final_score =
α × dense_score
+
(1 - α) × bm25_score
```

The problem is that the score distributions are different.

Dense retrieval:

```text
approximately [-1, 1]
```

BM25:

```text
0 → potentially large values
```

Directly combining these scores requires calibration.

---

# Reciprocal Rank Fusion

RRF combines ranked lists instead of raw scores.

The formula is:

```text
RRF(d)
=
Σ 1 / (k + rank(d))
```

where:

```text
k = smoothing constant
rank = position in retrieval list
```

A common value is:

```text
k = 60
```

---

## Example

Suppose:

```text
Dense:

1. A
2. B
3. C

BM25:

1. B
2. D
3. A
```

RRF rewards documents appearing near the top of both lists.

```text
A → strong dense + strong BM25
B → strong dense + strong BM25
C → dense only
D → BM25 only
```

This makes the fusion robust without requiring score normalization.

---

# Why Over-Fetch?

If we need:

```text
top_k = 5
```

we should not necessarily retrieve only 5 documents from each retriever.

Instead:

```text
dense → 3 × top_k
BM25  → 3 × top_k
```

with a minimum candidate count such as:

```text
20
```

This gives RRF a larger candidate pool.

```text
Dense candidates
       +
Sparse candidates
       ↓
     RRF
       ↓
Top candidates
```

---

# 7. Cross-Encoder Reranking

After hybrid retrieval, we have a smaller candidate set.

We can now use a more expensive but more accurate model.

---

## Bi-Encoder

The embedding model works like:

```text
Query
 ↓
Encoder
 ↓
Query Vector


Document
 ↓
Encoder
 ↓
Document Vector
```

Then:

```text
Similarity(Query Vector, Document Vector)
```

The query and document are encoded independently.

This allows document embeddings to be precomputed.

---

# Cross-Encoder

A cross-encoder processes the query and document together:

```text
[CLS] Query [SEP] Document [SEP]
                 ↓
             Transformer
                 ↓
          Relevance Score
```

The tokens from the query and document can directly attend to each other.

This allows the model to capture detailed interactions that independent embeddings may miss.

---

# Why Not Use Cross-Encoder for Everything?

Suppose there are:

```text
1,000,000 documents
```

A cross-encoder would need to evaluate:

```text
Query × 1,000,000 documents
```

for every query.

That is too expensive.

Instead:

```text
1,000,000 documents
        ↓
Dense + BM25
        ↓
20 candidates
        ↓
Cross-Encoder
        ↓
Top 5
```

This gives us a practical balance between speed and accuracy.

---

# Raw Logits

For ranking, we only care about the relative order of scores.

Therefore, raw model logits are sufficient:

```python
logits = model(**encoding).logits
```

Softmax is unnecessary because it converts scores into probabilities and makes them dependent on the other items in the batch.

For ranking:

```text
Higher logit = More relevant
```

is enough.

---

# 8. Grounded Generation

After retrieval and reranking, we have the best context chunks.

The final step is to give these chunks to an LLM.

```text
User Query
    +
Retrieved Context
    ↓
Prompt
    ↓
LLM
    ↓
Answer
```

---

# The RAG Contract

The system prompt should clearly define how the model must use the retrieved context.

Example:

```text
You are a precise question-answering assistant.

Answer using only the provided context.

If the context does not contain enough information,
say that the provided documents do not contain
enough information to answer the question.

Do not speculate or introduce unsupported facts.

Cite relevant sources using [Source N].
```

This establishes three important rules:

1. Use retrieved context.
2. Refuse when context is insufficient.
3. Provide source attribution.

---

# Context Formatting

Retrieved chunks can be formatted as:

```text
[Source 1]
Document: medical_guide.pdf
Chunk: 3

Blood pressure should be monitored regularly...


[Source 2]
Document: clinical_notes.txt
Chunk: 7

Patients with hypertension may require...


Question:
What does the documentation say about hypertension?
```

The numbered source identifiers allow the model to reference the supporting chunks.

---

# Source Attribution

Metadata should travel with every chunk throughout the pipeline.

```text
Document
   ↓
Chunk
   ↓
Embedding
   ↓
Vector Store
   ↓
Retriever
   ↓
Reranker
   ↓
Generator
```

Example metadata:

```json
{
    "source": "medical_guide.pdf",
    "chunk_index": 3,
    "start_char": 1250,
    "end_char": 1800
}
```

The final answer can therefore display:

```text
Sources:
- medical_guide.pdf — Chunk 3
- clinical_notes.txt — Chunk 7
```

---

# OpenAI-Compatible API

The generator does not need to depend on a specific SDK.

A direct HTTP request is sufficient:

```python
requests.post(
    f"{base_url}/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024
    },
    timeout=30
)
```

This architecture can work with any provider exposing a compatible API.

It can also be adapted to local inference servers.

---

# 9. Evaluation

A RAG system should not be evaluated only by looking at whether the final answer "looks good."

There are two separate problems.

## Retrieval Quality

Did the retriever find the correct chunks?

Useful metrics include:

* Precision@K
* Recall@K
* MRR
* Hit Rate
* NDCG

## Generation Quality

Did the LLM produce an answer that is actually supported by the retrieved context?

Useful concepts include:

* Faithfulness
* Answer relevance
* Context relevance
* Citation accuracy

---

# Precision@K

```text
Precision@K
=
Relevant Retrieved Documents
─────────────────────────────
          K
```

Example:

```text
Retrieved = 5
Relevant = 4

Precision@5 = 4 / 5 = 0.8
```

High precision means the retrieved context contains little irrelevant information.

---

# Recall@K

```text
Recall@K
=
Relevant Retrieved Documents
────────────────────────────
Total Relevant Documents
```

Example:

```text
Relevant documents = 4
Retrieved relevant = 3

Recall@K = 3 / 4 = 0.75
```

High recall means the retriever is less likely to miss useful information.

---

# Mean Reciprocal Rank

MRR focuses on how early the first relevant result appears.

```text
MRR
=
1/N × Σ 1/rank(first relevant result)
```

Examples:

```text
Relevant at rank 1 → 1.00
Relevant at rank 2 → 0.50
Relevant at rank 3 → 0.33
Relevant at rank 5 → 0.20
```

For RAG, this is especially useful because highly relevant chunks appearing near the top are more likely to be included in the final context.

---

# Faithfulness

Retrieval quality alone does not guarantee a good answer.

A system may retrieve the correct chunk but still generate unsupported information.

One possible evaluation approach is LLM-as-a-judge:

```text
Context
   +
Generated Answer
   ↓
Judge LLM
   ↓
Faithfulness Score
```

For example:

```python
def evaluate_faithfulness(answer, context):

    prompt = f"""
    Context:
    {context}

    Answer:
    {answer}

    Rate how well the answer is supported
    by the provided context from 0.0 to 1.0.

    Return only the score.
    """

    return float(llm.generate(prompt))
```

LLM-as-a-judge has limitations:

* Additional inference cost
* Judge-model bias
* Possible disagreement with human evaluation
* Sensitivity to answer style

Therefore, it should complement rather than replace human evaluation.

---

# 10. Implementation Decisions

## PyPDF2 vs pdfminer.six

This implementation uses:

```text
PyPDF2
```

because it is lightweight and easy to integrate.

For complex PDFs containing:

* Tables
* Multi-column layouts
* Complex formatting
* Scanned pages

a more advanced extraction pipeline may be required.

Document extraction quality is extremely important in production RAG because **bad extraction leads to bad retrieval**.

---

# NumPy vs FAISS

The vector store uses exact cosine search with NumPy.

Advantages:

* Simple
* Transparent
* Easy to debug
* No additional vector database
* Exact results

For a portfolio-scale corpus, this is often sufficient.

For very large datasets, approximate nearest-neighbor systems such as FAISS become more appropriate.

The important distinction is:

```text
NumPy
→ Exact Search

FAISS / ANN
→ Approximate / Optimized Search
```

The correct choice depends on corpus size and latency requirements.

---

# Simple BM25 Tokenization

The implementation intentionally uses simple tokenization.

```python
text.lower().split()
```

Advantages:

* Minimal dependencies
* Easy to understand
* Easy to debug
* Fully transparent

For production applications, tokenization can be improved with:

* NLTK
* spaCy
* Custom domain-specific tokenization

However, the BM25 scoring logic remains unchanged.

---

# Raw Transformers vs Sentence Transformers

A library such as `sentence-transformers` provides convenient methods such as:

```python
model.encode(text)
```

But internally, the pipeline still involves:

```text
Tokenizer
   ↓
Transformer
   ↓
Pooling
   ↓
Normalization
```

Using raw Hugging Face Transformers in this project makes those steps explicit.

This is useful for learning because it demonstrates what actually happens behind the abstraction.

For production systems, `sentence-transformers` remains a perfectly valid choice.

---

# Generation Temperature

The default generation temperature is:

```text
temperature = 0.2
```

Lower temperature generally makes generation more deterministic.

For a RAG system, we want the model to:

* Follow the retrieved context.
* Avoid unnecessary creativity.
* Produce consistent answers.
* Avoid unsupported speculation.

The exact temperature should still be tuned according to the selected model and application.

---

# 11. Complete Pipeline

Putting everything together:

```text
                    INDEXING PHASE

                    PDF / TXT
                        │
                        ▼
                  Text Extraction
                        │
                        ▼
                     Chunking
                        │
                  ┌─────┴─────┐
                  ▼           ▼
             Embeddings      BM25
                  │           │
                  ▼           ▼
             Vector Store   BM25 Index


                    QUERY PHASE

                    User Query
                        │
               ┌────────┴────────┐
               ▼                 ▼
          Query Embedding       BM25
               │                 │
               ▼                 ▼
          Vector Search      Sparse Search
               │                 │
               └────────┬────────┘
                        ▼
                     RRF Fusion
                        │
                        ▼
                   Candidate Set
                        │
                        ▼
                    Cross-Encoder
                        │
                        ▼
                  Ranked Context
                        │
                        ▼
                     Prompt
                        │
                        ▼
                       LLM
                        │
                        ▼
               Answer + Citations
```

---

# Key Takeaways

This project demonstrates the core building blocks of a modern RAG system without hiding them behind framework abstractions.

### Document Processing

```text
PDF/TXT
  ↓
Chunking
  ↓
Metadata
```

### Semantic Retrieval

```text
Text
 ↓
Transformer
 ↓
Mean Pooling
 ↓
Normalization
 ↓
Vector Search
```

### Lexical Retrieval

```text
Text
 ↓
Tokenization
 ↓
BM25
 ↓
Ranked Results
```

### Hybrid Retrieval

```text
Dense Results
      +
Sparse Results
      ↓
     RRF
```

### Precision Retrieval

```text
RRF Candidates
      ↓
Cross-Encoder
      ↓
Reranked Context
```

### Generation

```text
Context
   +
Query
   ↓
LLM
   ↓
Grounded Answer
```

---

# Final Architecture

The complete philosophy of this project can be summarized as:

```text
Understand
    ↓
Implement
    ↓
Evaluate
    ↓
Optimize
```

Instead of treating RAG as a single library call, this project breaks it into individual algorithms:

```text
Chunking
   +
Embeddings
   +
Vector Search
   +
BM25
   +
RRF
   +
Reranking
   +
Generation
   +
Evaluation
```

Each component can be independently inspected, tested, replaced, and optimized.

That makes the system useful not only as a working RAG implementation, but also as a practical study of **information retrieval, semantic search, ranking systems, and LLM-based generation**.

---

## Repository Mapping

Each major concept in this document corresponds directly to an implementation module:

| Concept            | Implementation       |
| ------------------ | -------------------- |
| Document ingestion | `src/ingestion.py`   |
| Recursive chunking | `src/chunker.py`     |
| Embeddings         | `src/embeddings.py`  |
| Vector search      | `src/vectorstore.py` |
| BM25               | `src/bm25.py`        |
| Hybrid retrieval   | `src/retriever.py`   |
| Reranking          | `src/reranker.py`    |
| Generation         | `src/generator.py`   |
| Evaluation         | `src/evaluation.py`  |

The implementation intentionally keeps the architecture small and transparent.

**No LangChain. No LlamaIndex. No hidden retrieval abstraction.**

Just the core algorithms, implemented and connected from first principles.
