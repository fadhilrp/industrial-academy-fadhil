---
tags: [handout, day3, rag, reference]
session: "Day 3 — Retrieval-Augmented Generation"
program: "Industrial AI & LLM Training Program"
export: "PDF-ready — export via Obsidian or Pandoc"
---

# Day 3 Handout — Retrieval-Augmented Generation (RAG)
*Industrial AI & LLM Training Program*

---

## 1. The RAG Architecture

### Why RAG?

Large Language Models (Day 2) excel at tasks where the answer is in their training data.
But industrial AI faces a fundamental mismatch:

- **LLM training data:** Public internet, books, research papers — cut off at training date
- **Industrial knowledge:** Your plant's SOPs, equipment manuals, incident logs, ERP configs
  — private, updated continuously, never in public training data

Without access to your documents, an LLM asked "What is the LOTO isolation sequence for pump P-101?"
will hallucinate: it invents plausible-sounding valve tags and panel locations that do not
exist in your plant.

**RAG (Retrieval-Augmented Generation)** solves this by giving the LLM the relevant document
*at query time* — before it generates the answer.

### Parametric vs. Non-Parametric Memory

| Memory type | Location | Updated by | Latency |
|---|---|---|---|
| **Parametric** | Model weights | Retraining (costly) | Zero (baked in) |
| **Non-parametric** | External document store | Adding/editing files (free) | Retrieval time |

RAG combines both: LLM provides reasoning and language generation (parametric);
the knowledge base provides domain-specific facts (non-parametric).

### The RAG Pipeline (ASCII)

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: RETRIEVE                                       │
│  Embed query → search vector index → top-k chunks      │
└────────────────────────┬────────────────────────────────┘
                         │  top-3 chunks
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: AUGMENT                                        │
│  Format: System prompt + Context block + User query    │
└────────────────────────┬────────────────────────────────┘
                         │  augmented prompt
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: GENERATE                                       │
│  LLM reads context → produces grounded answer          │
└─────────────────────────────────────────────────────────┘
```

**Three-letter acronym:** RAG = Retrieve → Augment → Generate

---

## 2. Document Preparation

### Loading Strategies

| Source | Tool | Notes |
|---|---|---|
| Plain text / Markdown | Python `open()` | Direct — no conversion |
| PDF | `PyPDFLoader` (LangChain), `pdfplumber` | Layout can be complex |
| Word (.docx) | `python-docx`, `Docx2txtLoader` | Handles tables |
| HTML | `BeautifulSoup`, `UnstructuredHTMLLoader` | Strip navigation |
| SharePoint / Confluence | LangChain connectors | Requires auth setup |
| SAP documents | Export to PDF/CSV → standard loaders | No native SAP connector |

**Cleaning before chunking:**
1. Remove boilerplate: headers, footers, page numbers
2. Normalize whitespace: collapse multiple spaces/newlines
3. Fix encoding: ensure UTF-8 (critical for Indonesian text)
4. Tag metadata: assign `doc_id`, `category`, `version`, `date` per document

### Chunking Strategies

| Strategy | Split boundary | Chunk size | Overlap | Best for |
|---|---|---|---|---|
| **Fixed-size** | Every N words | 100–300 words | 10–20% | Quick prototype |
| **Sentence-aware** | Sentence end (`.!?`) | 3–8 sentences | 1–2 sentences | Readable prose |
| **Semantic** | Embedding distance shift | Variable | N/A | Topic-structured docs |
| **Hierarchical** | Paragraph → section | Multi-level | N/A | Deep Q&A, summaries |

**Fixed-size chunking:**

```python
def chunk_fixed(text, doc_id, chunk_size=200, overlap=40):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "chunk_id": f"{doc_id}::chunk{len(chunks):02d}",
            "text": " ".join(words[start:end]),
        })
        if end == len(words): break
        start += chunk_size - overlap
    return chunks
```

**Sentence-aware chunking:**

```python
import re

def chunk_sentences(text, doc_id, target_chars=500, overlap=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, length, idx = [], [], 0, 0
    for sent in sentences:
        current.append(sent); length += len(sent)
        if length >= target_chars:
            chunks.append({"chunk_id": f"{doc_id}::sent{idx:02d}",
                           "text": " ".join(current)})
            current = current[-overlap:] if overlap else []
            length = sum(len(s) for s in current); idx += 1
    if current:
        chunks.append({"chunk_id": f"{doc_id}::sent{idx:02d}",
                       "text": " ".join(current)})
    return chunks
```

**Overlap visualization:**

```
Chunk 1: [──────────────────────][─── overlap ───]
Chunk 2:                         [─── overlap ───][──────────────────────]
```

Overlap = 10–20% of chunk size prevents losing context at chunk boundaries.

---

## 3. Embedding Models

### What is an embedding?

An embedding converts text into a dense numerical vector capturing semantic meaning:

```
"LOTO tidak dilakukan"      → [0.23, -0.87, 0.14, ..., 0.56]  (384 dims)
"Lockout procedure skipped" → [0.21, -0.85, 0.17, ..., 0.54]  (similar!)
"laporan keuangan Q3"       → [0.67,  0.12, -0.43, ..., 0.23]  (different)
```

Two sentences with similar meaning → vectors pointing in similar directions.
**Language-agnostic:** Indonesian text and English text map to the same space.

### Model Comparison

| Model | Dimensions | Languages | Max tokens | Size | Throughput |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | English only | 256 | 22 MB | Very fast |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ (incl. ID) | 128 | 118 MB | Fast |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ (incl. ID) | 384 | 278 MB | Medium |
| `indobert-base-p2` | 768 | Indonesian | 512 | 540 MB | Slow |
| `text-embedding-3-small` (OpenAI) | 1536 | 100+ | 8191 | API | Fast (API) |

**Recommendation for Indonesian industrial text:**
`paraphrase-multilingual-MiniLM-L12-v2` — handles mixed Indonesian/English,
runs on CPU, 384 dimensions keep index size manageable.

### Using sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings = model.encode(
    [chunk['text'] for chunk in chunks],
    normalize_embeddings=True,   # normalize to unit vectors for cosine
    show_progress_bar=True,
)
# embeddings.shape → (n_chunks, 384)
```

---

## 4. Vector Databases

### Comparison

| Feature | FAISS | ChromaDB | Milvus | Qdrant |
|---|---|---|---|---|
| **Type** | Library (in-process) | Embedded/Server | Distributed server | Server |
| **Persistence** | Manual (save file) | SQLite/DuckDB | Yes | On-disk |
| **Metadata filtering** | No | Yes | Yes | Yes |
| **Python install** | `faiss-cpu` | `chromadb` | `pymilvus` | `qdrant-client` |
| **Best for** | Prototype, Jupyter | Local app, team use | Large scale | Large scale |

### Index Types

| Index | How it works | Recall | Speed |
|---|---|---|---|
| **Flat** (brute-force) | Compare all vectors | 100% | Slow at scale |
| **IVF** (inverted file) | Cluster → search nearby clusters | ~95% | Fast |
| **HNSW** (graph-based) | Navigate small-world graph | ~99% | Very fast |

**Rule of thumb:** Under 100k vectors → Flat (exact). Over 100k → HNSW.

### FAISS Quick Reference

```python
import faiss
import numpy as np

dim = 384
index = faiss.IndexFlatIP(dim)                          # cosine for unit vecs
index.add(embeddings.astype(np.float32))

query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
scores, indices = index.search(query_vec, k=3)          # top-3
```

### ChromaDB Quick Reference

```python
import chromadb

client = chromadb.Client()  # in-memory
# client = chromadb.PersistentClient("./chroma_db")  # persistent

collection = client.create_collection("industrial_kb",
    metadata={"hnsw:space": "cosine"})

collection.add(
    ids=[c["chunk_id"] for c in chunks],
    embeddings=embeddings.tolist(),
    documents=[c["text"] for c in chunks],
    metadatas=[{"doc_id": c["doc_id"], "category": c["category"]}],
)

# Query with metadata filter
results = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=3,
    where={"category": "SOP"},          # metadata filter
)
```

---

## 5. Retrieval Strategies

### Dense Retrieval (Semantic Search)

```python
def cosine_top_k(query_text, k=3):
    q_vec = model.encode([query_text], normalize_embeddings=True)[0]
    sims = np.dot(chunk_embeddings, q_vec)
    top_k_idx = np.argsort(sims)[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in top_k_idx]
```

**Strengths:** Cross-language, paraphrase-robust, finds concepts not matching keywords.
**Weaknesses:** Fails on exact codes (ME21N, SOP-MECH-001), numbers, proper names.

### Sparse Retrieval (BM25)

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1(1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Standard parameters: $k_1 = 1.5$, $b = 0.75$.

```python
from rank_bm25 import BM25Okapi

corpus = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(corpus)

def bm25_top_k(query_text, k=3):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_k_idx]
```

### Hybrid Retrieval (Reciprocal Rank Fusion)

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

Typical $k = 60$. Outperforms either strategy alone.

### Re-ranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, candidates, k=3):
    pairs = [(query, c["text"]) for c, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(cs[0], float(s)) for cs, s in ranked[:k]]
```

### Trade-off Summary

| Strategy | Recall | Latency | Exact match | Cross-language |
|---|---|---|---|---|
| Dense | High | Medium | Poor | Excellent |
| Sparse (BM25) | Medium | Low | Excellent | Poor |
| Hybrid | Highest | Medium | Good | Good |
| Hybrid + Rerank | Best | High | Best | Best |

---

## 6. Context Injection

### Prompt Assembly

```python
RAG_SYSTEM = """You are an industrial AI assistant for a manufacturing facility.
Answer questions ONLY using the context documents below.
If the information is not in the context, say: "I could not find this in the knowledge base."
Always cite the source document: [Source: DOC-ID]"""

def format_context(retrieved_chunks, max_chunks=3):
    parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks[:max_chunks], 1):
        parts.append(f"[DOCUMENT {i}: {chunk['doc_id']}]\n{chunk['text']}")
    return "\n\n".join(parts)

def rag_query(user_query, k=3):
    retrieved = retrieve_top_k(user_query, k=k)
    context = format_context(retrieved)
    user_message = f"Context:\n{context}\n\nQuestion: {user_query}"
    answer = chat(
        messages=[{"role": "user", "content": user_message}],
        system=RAG_SYSTEM,
    )
    return answer, retrieved
```

### Token Budget

| Component | Typical tokens |
|---|---|
| System prompt | ~100 |
| 3 context chunks (200 words each) | ~850 |
| User query | ~50 |
| **Total input** | **~1,000** |
| LLM answer | ~200 |
| **Grand total** | **~1,200** |

**Cost at Haiku ($0.25/M):** $0.0003/query. 10,000 queries/month = $3.00/month.

### Citation Markers

```
System: "Always cite the source document ID: [Source: DOC-ID]"

Answer: "According to SOP-MECH-001, step 5 requires applying a personal lock
at each isolation point before proceeding with maintenance. [Source: SOP-MECH-001]"
```

Benefits: safety officers can verify, auditability for compliance, users know where to read more.

---

## 7. RAG Evaluation

### Evaluation Metrics

| Metric | Definition | Formula |
|---|---|---|
| **Faithfulness** | Answer supported by retrieved context? | Overlap(answer, context) / len(answer) |
| **Groundedness** | Does answer cite a source? | 1 if [Source:...] in answer else 0 |
| **Answer Relevance** | Does answer address the query? | Overlap(query, answer) / len(query) |
| **Context Precision** | Were retrieved chunks actually useful? | Useful chunks / total retrieved |
| **Latency** | End-to-end response time | wall-clock seconds |

### Heuristic Implementation

```python
import re

def faithfulness_score(answer, context_chunks):
    answer_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
    context_tokens = set()
    for chunk in context_chunks:
        context_tokens.update(re.findall(r'\b\w+\b', chunk.lower()))
    stopwords = {'the','a','is','in','of','to','and','for','yang','di','dan'}
    answer_tokens -= stopwords
    if not answer_tokens: return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)

def groundedness_score(answer):
    return 1.0 if re.search(r'\[Source\s*:', answer, re.IGNORECASE) else 0.0

def relevance_score(query, answer):
    q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
    stopwords = {'perlu','tidak','dan','the','a','is','to','and'}
    q_tokens -= stopwords
    if not q_tokens: return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens)
```

### RAGAS Framework (Production)

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=eval_dataset,   # HuggingFace Dataset with question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision],
)
```

**Typical production targets:**

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| Faithfulness | 0.6 | 0.75 | 0.9 |
| Answer Relevance | 0.5 | 0.7 | 0.85 |
| Context Precision | 0.5 | 0.7 | 0.9 |

---

## 8. Essential Libraries

### Installation

```bash
pip install sentence-transformers faiss-cpu chromadb anthropic openai \
            tiktoken requests numpy pandas matplotlib python-dotenv rank-bm25
```

### Library Reference

| Library | Version | Purpose |
|---|---|---|
| `sentence-transformers` | ≥2.3 | Text embeddings (multilingual MiniLM, IndoBERT) |
| `faiss-cpu` | ≥1.7 | Fast in-process vector similarity search |
| `chromadb` | ≥0.4 | Persistent vector database with metadata filtering |
| `anthropic` | ≥0.28 | Claude API client (Haiku, Sonnet, Opus) |
| `openai` | ≥1.0 | OpenAI API client (GPT-3.5, GPT-4o) |
| `tiktoken` | ≥0.6 | BPE tokenizer — count tokens before API call |
| `rank-bm25` | ≥0.2 | BM25 sparse retrieval |
| `ragas` | ≥0.1 | LLM-judge RAG evaluation framework |

---

## 9. Vocabulary / Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — combine retrieval from a knowledge base with LLM generation |
| **Chunk** | A segment of a larger document (~100–300 words), the retrieval unit |
| **Embedding** | Dense numerical vector representing the semantic meaning of text |
| **Vector DB** | Database optimized for storing and searching embedding vectors |
| **Dense retrieval** | Semantic search using embedding cosine similarity |
| **Sparse retrieval** | Keyword-based search using BM25 or TF-IDF scoring |
| **BM25** | Best Match 25 — standard sparse retrieval algorithm |
| **Hybrid retrieval** | Combining dense + sparse retrieval (Reciprocal Rank Fusion) |
| **Re-ranking** | Second scoring step using a cross-encoder to reorder initial retrieval results |
| **Faithfulness** | Fraction of answer content supported by retrieved context |
| **Groundedness** | Whether the answer includes citations to source documents |
| **Answer Relevance** | Whether the answer addresses the user's question |
| **RAGAS** | Open-source framework for automated RAG evaluation using LLM judges |
| **Cosine similarity** | Dot product of two unit vectors — measures angle between embedding vectors |
| **FAISS** | Facebook AI Similarity Search — fast in-process vector search library |
| **ChromaDB** | Open-source embedded vector database with persistence and metadata |
| **HNSW** | Hierarchical Navigable Small World — graph-based ANN index |
| **Parametric memory** | Knowledge stored in LLM weights (baked in during training) |
| **Non-parametric memory** | Knowledge stored in external documents (retrieved at query time) |
| **Context window** | Maximum token count an LLM can process in one call |
| **Token budget** | Allocation of context window across system, context, query, and answer |

---

## 10. Formulas Quick Reference

### Cosine Similarity

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

For L2-normalized unit vectors: $\text{cosine}(A, B) = A \cdot B$ (just the dot product).

### BM25 Scoring

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1+1)}{f(t,d) + k_1(1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Standard parameters: $k_1 = 1.5$, $b = 0.75$.

### Reciprocal Rank Fusion (RRF)

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

Standard: $k = 60$.

### Maximum Marginal Relevance (MMR)

Selects diverse chunks — avoids retrieving three chunks from the same document:

$$\text{MMR}(d) = \lambda \cdot \text{sim}(d, q) - (1-\lambda) \cdot \max_{d' \in S} \text{sim}(d, d')$$

Where $S$ = already selected chunks. $\lambda = 0.5$ balances relevance vs. diversity.

### Heuristic Faithfulness

$$\text{faithfulness} = \frac{|\text{answer\_tokens} \cap \text{context\_tokens}|}{|\text{answer\_tokens}|}$$

(Excluding stopwords from both sets.)

---

## 11. Day 4 Preview

### LangChain & LangGraph

Day 4 moves from manual pipeline code to declarative chains and intelligent agents.

**LangChain RAG chain (replaces tonight's `rag_query()`):**

```python
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa_chain.invoke({"query": "What is the LOTO procedure for P-101?"})
```

**LangChain document loaders:**

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("./sop_documents/", glob="**/*.pdf",
                         loader_cls=PyPDFLoader)
docs = loader.load()  # loads all PDFs, preserves metadata
```

**LangGraph multi-step retrieval agent:**

```python
# Agent decides: retrieve → reason → re-retrieve if needed → answer
from langgraph.graph import StateGraph
# (full implementation in Day 4 notebook)
```

### Day 4 Agenda

| Time | Topic |
|---|---|
| 20:00–20:15 | Day 3 recap + LangChain introduction |
| 20:15–21:00 | LangChain: document loaders, text splitters, RAG chain |
| 21:00–21:15 | Break |
| 21:15–21:45 | LangGraph: stateful agents, multi-step retrieval |
| 21:45–22:10 | RAGAS evaluation: LLM-judge faithfulness |
| 22:10–22:30 | Lab + Q&A |

---
*Day 3 — Retrieval-Augmented Generation | Industrial AI & LLM Training Program*
