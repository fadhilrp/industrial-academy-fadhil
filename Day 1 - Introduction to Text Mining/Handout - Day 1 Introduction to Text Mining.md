---
tags: [handout, day1, text-mining, reference]
session: "Day 1 — Introduction to Text Mining"
program: "Industrial AI & LLM Training Program"
export: "PDF-ready — export via Obsidian or Pandoc"
---

# Day 1 Handout — Introduction to Text Mining
*Industrial AI & LLM Training Program*

---

## 1. The Text Mining Pipeline

Text mining is the discipline of extracting structured, actionable insights from unstructured text. Unlike traditional analytics — which works on spreadsheets, sensor readings, and database records where every value has a defined column and type — text mining unlocks information trapped in free-form language. The maintenance ticket describing "vibration on the north pump," the shift handover note warning that a "bearing sounded different last night," the CAR explaining why the same failure has happened three times: none of this data lives in a number. Text mining gives us the tools to read it at scale, systematically, across thousands of documents at once.

> **Analogy — the manufacturing production line:** Raw ore comes out of the ground as unprocessed material full of impurities. A refinery cleans it, concentrates the valuable minerals, and shapes them into a usable form. Text mining does exactly the same thing. Raw, messy text (full of abbreviations, typos, and mixed languages) is the ore. The pipeline refines it, step by step, into a finished product: structured predictions, similarity scores, and searchable knowledge.

Here is what happens at each stage of the pipeline:

- **Collection & Loading** — You pull text from its source (a CMMS database, a CSV export from SAP, an email archive) into a Python environment using `pandas`. At this stage, practical data quality issues surface: inconsistent encoding (UTF-8 vs. ISO-8859-1), mixed languages within a single document, missing or blank fields, and duplicates. What goes in: raw text in whatever format the source provides. What comes out: a structured DataFrame where each row is one document. Where it can break: encoding errors that corrupt characters, or source systems that mix metadata fields with narrative text in ways you must disentangle.

- **Preprocessing** — Raw text is too noisy and variable for any algorithm to process directly. "Bearing," "BEARING," "bearing," "bearng," and "brg" might all refer to the same component — but a computer treats them as five completely unrelated tokens. Preprocessing cleans, normalizes, and tokenizes the text. What goes in: raw strings. What comes out: lists of meaningful tokens. Where it can break: if abbreviations are not expanded before stopword removal, "PM" (Preventive Maintenance) can be silently dropped because it looks like a two-letter noise word.

- **Feature Extraction** — Machine learning algorithms cannot read words; they only operate on numbers. This stage converts token lists into numerical vectors. Two families exist: sparse vectors (TF-IDF, where most values are zero) and dense vectors (embeddings from Word2Vec or transformers, where every dimension carries signal). What goes in: tokens or raw text. What comes out: a matrix of numbers — one row per document, one column per feature. Where it can break: a vocabulary that is too large creates a memory-intensive sparse matrix; a vocabulary that is too small loses important distinctions.

- **Model / Algorithm** — Once text is numeric, standard machine learning applies: K-Means for clustering similar documents, cosine similarity for semantic search, logistic regression or BERT for classification. The algorithm has no knowledge that it is working with text — it only sees arrays of numbers. What goes in: the numeric feature matrix. What comes out: cluster assignments, similarity scores, or class probabilities.

- **Insight / Application** — Model output is translated into business value: a dashboard showing failure category trends, an automated ticket-routing system, an alert when a new document resembles a historically critical failure, or a knowledge Q&A chatbot (Day 4). This step is where data science meets operations — and where domain experts must validate that the patterns the model found are real.

```
Raw Text
(maintenance logs, SAP tickets, safety reports, shift notes, emails)
    │
    ▼
┌─────────────────────────────────────────────┐
│  COLLECTION & LOADING                       │
│  pandas.read_csv() / read_excel() / DB API  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  PREPROCESSING                              │
│  Lowercase → Remove punctuation             │
│  Expand abbreviations → Tokenize            │
│  Remove stopwords → Stem (optional)         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  FEATURE EXTRACTION                         │
│  TF-IDF  /  Word2Vec  /  Embeddings         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  MODEL / ALGORITHM                          │
│  K-Means / BERT classifier / cosine sim     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  INSIGHT / APPLICATION                      │
│  Dashboard / Alert / Chatbot / Search       │
└─────────────────────────────────────────────┘
    │
    ▼
Business Value
(trend detection, automated tagging, knowledge Q&A)
```

---

## 2. Industrial Text Sources

Industrial organizations generate enormous amounts of text as a byproduct of daily operations — but this text has historically been ignored. Structured data (sensor readings, production counts, KPI dashboards) dominated analytics because it fits neatly into databases. Text does not. As a result, organizations sit on years of institutional knowledge buried in maintenance logs, incident reports, and operator notes: knowledge that could predict failures, identify repeat problems, and preserve expertise when experienced engineers retire.

> **Industrial context:** A single plant with 200 engineers filing one work order per week generates over 10,000 text documents per year. Across five years, that is 50,000 real descriptions of failures, fixes, root causes, and lessons learned — all untouched by any systematic analysis.

**Accessing industrial text data** typically requires one of three paths: (1) a CSV or Excel export from your CMMS or SAP system — most platforms support this from the reporting module; (2) a read-only database query via ODBC or JDBC connection if your IT team grants access; (3) email archives exported in `.eml` or `.mbox` format from your mail server. Always work on a read-only copy of the data, never modify source systems, and obtain explicit IT and data governance approval before connecting to live systems.

| Text Source | Typical Format | Mining Goal |
|---|---|---|
| Maintenance logs / Work Orders | Free text + form fields | Failure pattern detection |
| SAP/ERP tickets & comments | Semi-structured + free text | Issue categorization |
| Safety incident reports | Form + narrative | Risk signal extraction |
| Operator shift handover notes | Informal free text | Anomaly detection |
| Procurement emails | Email threads | Bottleneck / delay detection |
| Technical manuals / SOPs | PDF documents | Knowledge retrieval (Day 4) |
| Quality inspection notes | Form + comment fields | Defect trend analysis |
| Corrective Action Requests (CAR) | Structured workflow | Systemic issue identification |

Each source presents unique NLP challenges. Maintenance logs are short and abbreviation-heavy ("Ganti seal pompa P-101, bocor") — making abbreviation expansion critical before any other step. Safety incident reports mix formal narrative sections written by the HSE team with informal handwritten notes added in the field, often under time pressure. Operator shift handover notes contain plant-specific shorthand that only long-tenure staff understand and that no generic NLP library will handle correctly. Procurement emails contain signature blocks, legal disclaimers, and quoted reply chains that must be stripped before any meaningful analysis. Understanding the quirks of your specific data sources is more valuable than knowing which algorithm to use.

---

## 3. Preprocessing Steps

The fundamental principle of preprocessing is "garbage in, garbage out." A model trained on raw, uncleaned industrial text will learn noise patterns — abbreviations treated as distinct words, the same concept spelled five different ways, stopwords dominating the feature space. Preprocessing removes that noise before it reaches the algorithm. Done well, it is invisible: the model simply works better. Done poorly, it corrupts your results in ways that are hard to diagnose.

> **Key insight:** Preprocessing decisions are largely irreversible in your pipeline. Once you drop a word or collapse a token, it is gone. This is where domain expertise matters most — an NLP engineer cannot know that "PM" means "Preventive Maintenance" at your plant unless a maintenance engineer tells them. The code is generic; the dictionaries and decisions are yours.

### Step-by-Step Explanation

**Step 1 — Lowercase:** Converts all characters to lowercase so that "Bearing," "BEARING," and "bearing" are treated as the same word. Without this, your vocabulary triples unnecessarily and term frequency counts become inaccurate. This step is almost always safe to apply.
- Before: `"GANTI SEAL POMPA Sentrifugal"`
- After: `"ganti seal pompa sentrifugal"`

**Step 2 — Remove punctuation:** Strips characters that carry no semantic meaning for most analysis tasks. The regex `re.sub(r'[^\w\s-]', ' ', text)` keeps word characters, spaces, and hyphens (important for hyphenated terms like "lock-out"). Without this step, `"bearing,"` and `"bearing"` become two different tokens, and every word followed by a period or comma in your corpus creates a spurious vocabulary entry.
- Before: `"pompa P-101: bocor!! segera ganti."`
- After: `"pompa P-101  bocor   segera ganti "`

**Step 3 — Collapse whitespace:** Removing punctuation leaves multiple consecutive spaces. This step collapses them to a single space and strips leading/trailing whitespace. It is trivial but required for clean tokenization downstream.

**Step 4 — Expand abbreviations:** This is the most domain-specific step and the one most frequently skipped by NLP engineers who do not know the domain. In industrial settings, `"PM"` means `"preventive maintenance"`, `"WO"` means `"work order"`, `"K3"` means `"keselamatan kesehatan kerja"`. If you skip this step, your model will never learn that "PM" and "preventive maintenance" are the same concept — they will appear as completely unrelated tokens. Expanding abbreviations before tokenization also ensures the expanded phrases ("preventive maintenance") are available as bigram features.
- Before: `"wo 4521 pm pump p101 blm selesai"`
- After: `"work order 4521 preventive maintenance pump p101 belum selesai"`

**Step 5 — Tokenize:** Splits a continuous string into a list of individual word tokens. `nltk.word_tokenize()` handles edge cases that simple `.split()` misses — it correctly separates punctuation attached to words and manages contractions. The output changes your data type from a string to a list, which is the input format all subsequent steps expect.
- Before: `"ganti bearing pompa sentrifugal"`
- After: `["ganti", "bearing", "pompa", "sentrifugal"]`

**Step 6 — Remove stopwords:** Removes high-frequency words that carry little meaning: prepositions, conjunctions, auxiliary verbs. In Indonesian + English industrial text, you need two stopword lists combined. Without this step, the most common words in your TF-IDF matrix will be `"yang"`, `"di"`, `"dan"`, `"the"`, `"and"` — words that appear everywhere and distinguish nothing. They will dominate your feature space and make every document look similar to every other document.

> **Critical nuance:** Generic stopword lists do not know your domain. The word `"normal"` appears in standard English stopword lists — but `"tidak normal"` (not normal) is a critical failure signal in maintenance text. The word `"critical"` may be too important to remove. Always review and customize your stopword list with a domain expert. Build a small custom list of plant-specific noise words (`"mohon"`, `"harap"`, `"segera"`, `"info"`) and add them to the standard list.

**Step 7 (optional) — Stemming:** Reduces inflected word forms to their root. In Indonesian, PySastrawi handles this — `"penggantian"`, `"mengganti"`, `"terganti"` all reduce to `"ganti"`. Stemming improves recall (finding all variants of a concept) at the cost of some precision. Use it when your corpus has rich morphological variation and your goal is broad topic clustering. Skip it when fine-grained word distinctions matter (e.g., classifying `"perlu diganti"` vs `"sudah diganti"`).

### Stemming vs. Lemmatization

Both approaches normalize word forms to a common root, but they differ in how they do it and what they produce:

| | Stemming | Lemmatization |
|---|---|---|
| **Method** | Chop word endings using rules | Map to dictionary base form |
| **Result** | May not be a real word | Always a valid word |
| **Speed** | Very fast | Slower (requires lexicon lookup) |
| **Indonesian example** | `penggantian` → `ganti` | `penggantian` → `ganti` |
| **English example** | `running` → `run`, `better` → `bet` | `running` → `run`, `better` → `good` |
| **Best for** | Fast pipelines, clustering tasks | Classification tasks needing precision |

For Indonesian industrial NLP, PySastrawi stemming is usually sufficient. The stemmer handles the rich affixation system (prefixes `me-`, `pe-`, `ke-`; suffixes `-an`, `-kan`, `-i`) and produces meaningful roots for technical vocabulary.

### Preprocessing Tradeoffs

Every preprocessing decision involves a tradeoff between signal and noise reduction. Lowercasing loses `URGENT` as a severity indicator. Removing punctuation loses `>80°C` as a temperature threshold. Removing stopwords loses negation: `"tidak bocor"` (not leaking) can become simply `"bocor"` (leaking) after Indonesian stopword removal, since `"tidak"` may appear on your list. Stemming collapses `"penggantian"` (replacement) and `"pengganti"` (replacement part) into the same root even if the distinction matters.

> **Rule of thumb:** Preprocess aggressively for unsupervised tasks (clustering) where you want broad concept grouping. Preprocess conservatively for supervised classification tasks where fine distinctions — urgency, negation, severity levels — carry label information.

### 3.1 Standard Six-Step Pipeline

| Step | Operation | Code |
| ------------ | -------------------- | ----------------------------------- |
| 1 | Lowercase | `text.lower()` |
| 2 | Remove punctuation | `re.sub(r'[^\w\s-]', ' ', text)` |
| 3 | Collapse whitespace | `re.sub(r'\s+', ' ', text).strip()` |
| 4 | Expand abbreviations | Custom dictionary lookup |
| 5 | Tokenize | `nltk.word_tokenize(text)` |
| 6 | Remove stopwords | NLTK + custom Indonesian list |
| 7 (optional) | Stem | PySastrawi `StemmerFactory` |

### 3.2 Common Indonesian Industrial Abbreviations

| Abbreviation | Full Form | Domain |
|---|---|---|
| WO | Work Order | Maintenance |
| PM | Preventive Maintenance | Maintenance |
| CM | Corrective Maintenance | Maintenance |
| MTTR | Mean Time To Repair | Maintenance KPI |
| MTBF | Mean Time Between Failures | Maintenance KPI |
| PO | Purchase Order | Procurement |
| GR | Goods Receipt | ERP/SAP |
| LOTO | Lock-Out Tag-Out | Safety |
| APD | Alat Pelindung Diri (PPE) | Safety |
| K3 | Keselamatan, Kesehatan, Kerja | Safety |
| NCR | Non-Conformance Report | Quality |
| SOP | Standard Operating Procedure | General |
| P&ID | Piping and Instrumentation Diagram | Engineering |
| DCS | Distributed Control System | Instrumentation |
| SCADA | Supervisory Control and Data Acquisition | IT/OT |
| HMI | Human-Machine Interface | IT/OT |

### 3.3 Common Informal Indonesian Normalizations

| Informal | Formal | Informal | Formal |
| -------- | ----------- | -------------- | -------- |
| `udah` | `sudah` | `ga` / `nggak` | `tidak` |
| `blm` | `belum` | `sdh` | `sudah` |
| `tiba2` | `tiba-tiba` | `bs` | `bisa` |
| `gnti` | `ganti` | `utk` | `untuk` |
| `bgt` | `banget` | `krn` | `karena` |
| `sm` | `sama` | `jgn` | `jangan` |

### 3.4 Full Preprocessing Function

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')

ABBREV_MAP = {
    'wo': 'work order',      'pm': 'preventive maintenance',
    'po': 'purchase order',  'gr': 'goods receipt',
    'loto': 'lockout tagout', 'apd': 'alat pelindung diri',
    'ga': 'tidak',           'nggak': 'tidak',
    'udah': 'sudah',         'blm': 'belum',
    'sdh': 'sudah',          'tiba2': 'tiba-tiba',
    'gnti': 'ganti',         'utk': 'untuk',
    'bs': 'bisa',            'krn': 'karena',
}

INDONESIAN_STOPS = {
    'yang', 'di', 'dan', 'ini', 'itu', 'dari', 'ke', 'pada',
    'dengan', 'untuk', 'adalah', 'ada', 'atau', 'karena',
    'oleh', 'dalam', 'saat', 'masih', 'baru', 'sedang',
    'sudah', 'belum', 'akan', 'perlu', 'bisa', 'sangat',
    'juga', 'lagi', 'pun', 'saja', 'maka', 'jika',
}

def preprocess_text(text):
    """Preprocess industrial text for NLP analysis."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [ABBREV_MAP.get(t, t) for t in tokens]
    text = ' '.join(tokens)
    tokens = word_tokenize(text)
    en_stops = set(stopwords.words('english'))
    all_stops = INDONESIAN_STOPS | en_stops
    tokens = [t for t in tokens if t not in all_stops and len(t) > 1]
    return tokens  # Returns list of tokens
```

---

## 4. Feature Extraction

### 4.1 TF-IDF

**Plain-English intuition:** TF-IDF answers one question — *what makes this document unique compared to everything else in the collection?* A word that appears frequently in one document but rarely across the corpus is highly informative for that document. A word that appears in every document is not informative at all. TF-IDF captures this by multiplying two quantities: how often a term appears in the document (TF) by how rare that term is across the corpus (IDF).

**Formula:**

```
TF(t, d)     = count(t in document d) / total words in d
IDF(t)       = log( N / df(t) )
               N = total number of documents
               df(t) = number of documents containing term t
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Worked numerical example:** Suppose you have a corpus of 100 maintenance tickets and you want to score the term `"bearing"` for Ticket #7.

- Ticket #7 has 20 words, and `"bearing"` appears 3 times → **TF = 3/20 = 0.15**
- `"bearing"` appears in 10 of the 100 tickets → **IDF = log(100/10) = log(10) ≈ 2.30**
- **TF-IDF = 0.15 × 2.30 ≈ 0.345**

Now compare with the term `"dan"` (and):
- `"dan"` appears 5 times in Ticket #7 → TF = 5/20 = 0.25
- `"dan"` appears in 95 of 100 tickets → IDF = log(100/95) ≈ 0.05
- **TF-IDF = 0.25 × 0.05 ≈ 0.013**

Despite `"dan"` being more frequent in the document, its near-universal presence in the corpus makes its TF-IDF score negligible. TF-IDF automatically down-weights ubiquitous words without you needing to manually specify them as stopwords.

**Sparse vs. dense vectors:** TF-IDF produces a sparse matrix. If your vocabulary has 500 terms and you have 1,000 documents, your matrix is 1,000 × 500 — but most cells are zero (because most terms don't appear in most documents). Sparse matrices are memory-efficient and very fast to compute, but they have no concept of semantic similarity: "pump" and "pompa" would be two entirely unrelated columns even though they mean the same thing in different languages.

**N-grams:** Setting `ngram_range=(1, 2)` adds bigrams (two-word phrases) to the vocabulary. This lets the model treat `"preventive maintenance"` as a single concept rather than two separate words. Without bigrams, `"preventive"` and `"maintenance"` each score independently, which loses the compound meaning.

**Sublinear TF scaling:** Setting `sublinear_tf=True` replaces raw term frequency with `log(1 + tf)`. This prevents a term appearing 100 times from being scored 100× higher than one appearing once. In practice, a word appearing many times in a document provides diminishing informational return — log scaling reflects this reality.

**sklearn implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=200,      # Vocabulary size limit
    ngram_range=(1, 2),    # Unigrams + bigrams ("preventive maintenance")
    min_df=2,              # Ignore terms in fewer than 2 documents
    sublinear_tf=True      # Use log(1 + tf) — best practice
)

# corpus = list of preprocessed text strings
tfidf_matrix = vectorizer.fit_transform(corpus)
# Shape: (n_documents × n_features) — sparse matrix

feature_names = vectorizer.get_feature_names_out()

# Top 5 terms for document at index i
def top_terms(tfidf_matrix, feature_names, doc_idx, n=5):
    scores = tfidf_matrix[doc_idx].toarray()[0]
    ranked = sorted(zip(feature_names, scores),
                    key=lambda x: x[1], reverse=True)
    return ranked[:n]
```

---

### 4.2 Word2Vec

**The core training idea — Skip-gram:** Word2Vec learns by playing a fill-in-the-blank game across your corpus. Given the center word `"bearing"`, the model tries to predict the surrounding context words: `"ganti"`, `"rusak"`, `"vibrasi"`. It adjusts its internal numeric representations (vectors) each time it gets the prediction right or wrong. After training on thousands of sentences, words that consistently appear in similar contexts end up with numerically similar vectors — even if they never appeared next to each other directly.

**The distributional hypothesis:** The theoretical foundation is simple: *words that appear in the same context tend to have related meanings.* "Pompa" and "pump" will appear near words like "bocor," "tekanan," "ganti," and "pressure" — so after training, their vectors will be close together in space, even without explicitly telling the model they are translations of each other.

**What a vector IS:** A Word2Vec vector is a list of numbers — typically 100 to 300 floats — where the *position* of a word in that multidimensional space encodes its meaning. You cannot interpret any single dimension ("dimension 47 means severity"). The meaning is in the *distances* between words: `similarity("pompa", "pump")` is high because their vectors point in nearly the same direction; `similarity("pompa", "laporan")` is low because they appear in very different sentence contexts.

**Skip-gram vs. CBOW:**

| | Skip-gram (sg=1) | CBOW (sg=0) |
|---|---|---|
| **Task** | Predict context words from center word | Predict center word from context words |
| **Better for** | Rare words, smaller datasets | Frequent words, larger datasets |
| **Training speed** | Slower | Faster |
| **Recommended when** | Domain vocabulary is specialized and sparse | Large general corpus |
| **Our sessions** | Default choice for industrial text | — |

**Context-independence — the key limitation:** Word2Vec assigns exactly one vector per word, regardless of context. The word `"kritis"` gets the same vector whether it appears in `"kondisi kritis"` (critical condition) or `"mesin kritis"` (critical machine). This is fine for most industrial text, but it fails for words with genuinely different meanings in different contexts. More fundamentally, it means Word2Vec cannot understand that `"bearing rusak"` and `"bearing good news"` use the word `"bearing"` in completely different senses — it averages the meaning across all contexts it has seen.

**Practical limitations:** Word2Vec needs a reasonably large corpus to learn meaningful representations. With fewer than a few hundred documents, vectors for rare domain terms will be poorly trained and unreliable. If your plant vocabulary contains highly specialized terms that appear in only a handful of tickets, Word2Vec will not learn useful vectors for them. In these cases, TF-IDF often performs better because it does not require learning from co-occurrence.

**Key parameters:**

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=tokenized_docs,  # List of token lists
    vector_size=100,           # Embedding dimensions
    window=5,                  # Context window radius
    min_count=1,               # Minimum word frequency
    sg=1,                      # 1=Skip-gram, 0=CBOW
    epochs=50,                 # Training iterations
    seed=42                    # Reproducibility
)

# API reference
model.wv.most_similar('bearing')          # Top similar words + scores
model.wv['bearing']                       # Raw vector, shape (100,)
model.wv.similarity('pompa', 'pump')      # Cosine similarity score (0–1)
model.wv.most_similar(                    # Word arithmetic
    positive=['pompa', 'rusak'], negative=['normal'])
model.save('word2vec_industrial.model')   # Save model
loaded = Word2Vec.load('word2vec_industrial.model')
```

---

### 4.3 Sentence Transformers (Contextual Embeddings)

**The attention mechanism — intuitively:** In a classic model like Word2Vec, every word is processed independently. Transformers work differently: when encoding the word `"bearing"`, the model simultaneously looks at every other word in the sentence and asks, "how much should each of those words influence my understanding of this one?" This is the self-attention mechanism. The word `"rusak"` (broken) appearing near `"bearing"` shifts the representation of `"bearing"` toward failure context. The word `"good"` appearing near `"bearing"` shifts it toward the English idiom. Every word is contextualized by its surroundings.

**Why contextual embeddings fix Word2Vec's core limitation:** Because the vector for a word is computed fresh for each sentence, the same word can have a different representation in different contexts. `"Bearing rusak diinspeksi"` and `"Bearing good news from HQ"` — despite sharing the token `"bearing"` — will produce entirely different embedding vectors for that word. This is what makes transformers powerful for semantic search: two sentences can be meaningfully compared even when they share no words at all, as long as their meaning is similar.

**Sentence-transformers vs. raw BERT:** Raw BERT outputs a vector for every token in a sentence, not a single vector for the whole sentence. To get a sentence-level representation from raw BERT, you need to pool or aggregate these token vectors — and doing this naively gives poor results for similarity tasks. The `sentence-transformers` library solves this by training BERT-like models with a specific objective (Siamese network training on sentence pairs) that makes the pooled sentence vector directly meaningful for similarity comparison. Use `sentence-transformers` for any task involving document similarity, clustering, or semantic search. Use raw BERT (via HuggingFace `transformers`) only when you need token-level outputs or are fine-tuning for classification (Day 2).

**Cross-language capabilities:** The `paraphrase-multilingual-MiniLM-L12-v2` model was trained on parallel text in 50+ languages. This means Indonesian and English sentences about the same topic end up close together in the shared embedding space — even without translation. A query in English will retrieve relevant Indonesian documents, and vice versa. For a bilingual Indonesian industrial environment where documentation is in both languages, this is a significant practical advantage.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load multilingual model (~118MB — pre-download before session)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode texts (batch processing, GPU-optional)
embeddings = model.encode(texts, show_progress_bar=True)
# Shape: (n_texts × 384) — dense float32 matrix

# Pairwise similarity matrix
sim_matrix = cosine_similarity(embeddings)
# sim_matrix[i][j] = similarity between text i and text j (0 to 1)

# Find most similar to query
query_emb = model.encode(["pompa sentrifugal vibrasi"])
sims = cosine_similarity(query_emb, embeddings)[0]
top_idx = sims.argsort()[::-1][:5]  # Top 5
```

**Model comparison:**

| Model | Dimensions | Size | Languages | Recommended for |
|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 118 MB | 50+ incl. ID | **This course — balanced** |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 420 MB | 50+ incl. ID | Higher accuracy, slower |
| `all-MiniLM-L6-v2` | 384 | 80 MB | English only | English-only data |
| `indobenchmark/indobert-base-p1` | 768 | 680 MB | Indonesian | Fine-tuning (Day 2) |
| `bert-base-multilingual-cased` | 768 | 680 MB | 104 languages | Fine-tuning base (Day 2) |

---

## 5. Classical NLP vs Transformer Comparison

Both classical NLP and transformer-based approaches have a permanent place in the industrial practitioner's toolkit. The decision between them is not about which is "better" — it is about matching the tool to the problem's actual constraints. Transformers are not always the right answer. A TF-IDF classifier that achieves F1=0.85 in 30 seconds is almost always preferable to a fine-tuned BERT that achieves F1=0.87 after 2 hours of GPU training and requires a cloud inference endpoint to run. Start simple. Add complexity only when simplicity demonstrably fails.

> **A realistic decision narrative:** Imagine you have 500 maintenance tickets and want to automatically route them to the correct maintenance team (Electrical, Mechanical, Instrumentation, Civil). Here is how you would decide which approach to use.
>
> First, you have no labels — no one has categorized those 500 tickets. Start with TF-IDF + K-Means clustering (tonight's lab). Run 4 clusters, inspect the top terms, and ask a maintenance engineer whether the clusters make practical sense. If they do, you can use those clusters as a starting point for labeling.
>
> Now you have ~100 labeled tickets per category. Train a TF-IDF + Logistic Regression classifier. Check F1 score with 5-fold cross-validation. If F1 > 0.75, you are done — deploy the classical model. It is fast, explainable, and requires no GPU.
>
> If F1 < 0.75, the vocabulary overlap between categories may be too high for bag-of-words approaches. You need semantics. Try sentence-transformer embeddings + the same logistic regression head. If F1 improves to >0.80, done.
>
> If you still need higher accuracy and you have 500+ labeled examples, fine-tune IndoBERT (Day 2 approach). At this point you are investing significant effort, so be sure the accuracy gain justifies it.

| Aspect | Classical NLP | Transformer-Based |
|---|---|---|
| **Core method** | Bag-of-words, TF-IDF, regex | Self-attention mechanism |
| **Input representation** | Sparse word counts | Dense subword tokens |
| **Context handling** | None (each word independent) | Full bidirectional context |
| **Training data** | Small (50+ docs fine) | Pre-trained on huge corpora |
| **Fine-tuning data** | N/A | 200+ labeled examples |
| **Training time** | Seconds–minutes | Hours–days (fine-tune: 10–30 min) |
| **Inference speed** | Very fast (CPU) | Slower (GPU preferred) |
| **Memory footprint** | Minimal | 200MB–4GB depending on model |
| **Interpretability** | High (feature weights) | Low (black box) |
| **Multilingual** | Language-specific tools | Multilingual models available |
| **Indonesian support** | PySastrawi + custom dict | `indobert`, multilingual MiniLM |
| **Best for** | Keyword extraction, fast categorization | Semantic search, QA, generation |
| **Libraries** | NLTK, sklearn, spaCy, Gensim | HuggingFace Transformers, sbert |

**Decision flowchart** — with explanation at each node:

```
Do you have labeled data?
    │
    ├─ No  → Use TF-IDF + K-Means clustering (unsupervised)
    │        [You cannot train a supervised model without labels.
    │         Cluster first, then manually label the clusters to create
    │         a training set for the next step.]
    │
    └─ Yes → Is F1 score > 0.70 with TF-IDF + sklearn classifier?
                 │
                 ├─ Yes → Stick with classical NLP (simpler = better)
                 │        [A working simple model beats a complex one.
                 │         Classical models are fast, interpretable,
                 │         and deployable without GPU infrastructure.]
                 │
                 └─ No  → Do you have > 200 labeled examples?
                              │
                              ├─ No  → Use pre-trained embeddings + fine-tune head
                              │        [Sentence-transformer embeddings + logistic
                              │         regression can work with as few as 50 examples.
                              │         The pre-trained model provides the semantic
                              │         foundation; you only train the classification layer.]
                              │
                              └─ Yes → Fine-tune full BERT (Day 2 approach)
                                       [With 200+ examples, you have enough data to
                                        adapt the transformer's internal representations
                                        to your specific domain and task.]
```

---

## 6. Key Algorithms

### 6.1 K-Means Clustering

**How K-Means works, step by step:** K-Means is an iterative algorithm that partitions documents into K groups based on similarity. Here is what happens under the hood:

1. **Initialize:** Randomly place K cluster centers (centroids) in the vector space.
2. **Assign:** For every document, calculate its distance to each centroid. Assign the document to the cluster whose centroid is closest.
3. **Update:** Recalculate each centroid as the average position of all documents currently assigned to it.
4. **Repeat:** Go back to step 2. Keep iterating until assignments stop changing (convergence) or a maximum iteration limit is reached.

The algorithm is guaranteed to converge, but not to find the globally optimal clustering — it finds a locally optimal solution that depends on the random initialization in step 1.

**Why K-Means needs numeric vectors:** K-Means operates entirely on geometric distance — specifically, the Euclidean distance between points in vector space. It has no concept of words or text. This is why feature extraction (TF-IDF or embeddings) must come first: you are converting text into points in space, and K-Means groups those points by proximity.

**Choosing K — the elbow method and silhouette score:** The most practical challenge in K-Means is selecting K. Two complementary methods:

- **Elbow method:** Run K-Means for K = 2, 3, 4, ... 10. Plot the total inertia (sum of squared distances from each document to its cluster centroid) against K. The plot typically shows a rapid decrease followed by a leveling off — the "elbow" is the point of diminishing returns. Choose K at the elbow.
- **Silhouette score:** Measures how well each document fits its assigned cluster relative to neighboring clusters. Scores range from −1 to +1. A score above 0.5 indicates reasonably well-separated clusters; above 0.7 is strong; below 0.25 means the cluster structure is weak. When the elbow method is ambiguous, silhouette scores help break the tie.

> **Practical note on non-determinism:** K-Means is non-deterministic. Even with `random_state=42`, different runs with the same parameters on slightly different data can produce meaningfully different cluster assignments. This is expected behavior, not a bug. Always inspect the top terms for each cluster after every run and manually assign human-readable labels (e.g., "Cluster 2 = Electrical Failures") rather than relying on cluster numbers to be stable.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# NOTE: K-Means is non-deterministic across runs even with random_state.
# Always inspect top terms to assign cluster labels manually.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(tfidf_matrix)  # Returns cluster IDs (0, 1, 2, 3)

# Evaluate cluster quality
score = silhouette_score(tfidf_matrix, labels)
print(f"Silhouette Score: {score:.3f}")  # Range: -1 to 1, higher = better

# Top terms per cluster (for manual label assignment)
def top_terms_per_cluster(kmeans, feature_names, n=7):
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        top_idx = center.argsort()[-n:][::-1]
        terms = [feature_names[j] for j in top_idx]
        count = (labels == i).sum()
        print(f"Cluster {i} ({count} docs): {', '.join(terms)}")

top_terms_per_cluster(kmeans, feature_names)
```

---

### 6.2 Cosine Similarity Search

**Why cosine similarity instead of Euclidean distance for text:** Consider two maintenance reports about the same pump failure — one is a brief 30-word note, and one is a detailed 300-word investigation summary. Euclidean distance in TF-IDF space would make them appear far apart (because all the word-count values in the longer document are larger), even though they describe the same event. Cosine similarity sidesteps this problem entirely: instead of measuring the straight-line distance between two vectors, it measures the angle between them. Two documents that use the same vocabulary in the same proportions will have an angle of 0° (cosine = 1.0) regardless of document length. Length differences cancel out.

> **Intuition:** Imagine two arrows pointing in the same direction — one short, one long. Euclidean distance between their tips is large. But the angle between them is zero. Cosine similarity measures the angle, not the distance between tips. For text, direction (which words, in what proportions) matters more than magnitude (how many words total).

**Cosine similarity interpretation guide:**

| Score | Interpretation | Example in industrial text |
|---|---|---|
| 0.90 – 1.00 | Near-duplicate; same event described twice | Two technicians filing reports on the same incident |
| 0.70 – 0.90 | Same topic; same equipment or failure type | Different pump failures with similar symptom vocabulary |
| 0.50 – 0.70 | Related; same domain but different specifics | Pump failure report vs. compressor vibration report |
| < 0.50 | Different topics | Mechanical failure report vs. procurement delay email |

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Full pairwise similarity matrix
sim_matrix = cosine_similarity(embeddings)  # Shape: (n × n)

# Find top-K most similar documents to a query
def find_similar(query_idx, embeddings, texts, top_k=5):
    sims = cosine_similarity([embeddings[query_idx]], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][1:top_k+1]  # Exclude self (rank 0)
    for idx in top_idx:
        print(f"  [{sims[idx]:.3f}] {texts[idx][:70]}")

# Threshold-based filtering (e.g., similarity > 0.8)
similar_pairs = np.argwhere(sim_matrix > 0.8)
similar_pairs = similar_pairs[similar_pairs[:, 0] < similar_pairs[:, 1]]
```

---

## 7. Essential Libraries

The Python NLP ecosystem is a layered stack where each library occupies a specific role. `pandas` and `numpy` form the data foundation — everything else builds on them. `nltk` handles the linguistic primitives: tokenization and stopwords. `scikit-learn` provides TF-IDF vectorization and all classical ML algorithms (clustering, classification, evaluation). `gensim` is the go-to library for training Word2Vec models locally on domain-specific corpora — it is significantly more efficient than implementing Word2Vec in raw PyTorch. `sentence-transformers` wraps HuggingFace models with a clean API specifically optimized for sentence-level similarity tasks. `PySastrawi` fills the Indonesian-specific gap that no general library covers. `transformers` and `datasets` are the HuggingFace libraries used for fine-tuning pre-trained models — the core of Day 2.

**One-line install:**

```bash
pip install nltk scikit-learn gensim sentence-transformers \
            pandas numpy matplotlib PySastrawi \
            transformers datasets accelerate
```

| Library | Min Version | Purpose | Key Classes / Functions |
|---|---|---|---|
| `nltk` | ≥ 3.8 | Tokenization, stopwords, POS | `word_tokenize`, `stopwords`, `FreqDist` |
| `scikit-learn` | ≥ 1.3 | TF-IDF, clustering, evaluation | `TfidfVectorizer`, `KMeans`, `cosine_similarity` |
| `gensim` | ≥ 4.3 | Word embeddings, topic models | `Word2Vec`, `KeyedVectors`, `LdaModel` |
| `sentence-transformers` | ≥ 2.7 | Contextual sentence embeddings | `SentenceTransformer`, `.encode()` |
| `pandas` | ≥ 2.0 | Data manipulation | `DataFrame`, `read_csv`, `groupby`, `pivot_table` |
| `numpy` | ≥ 1.24 | Numerical arrays | `array`, `argsort`, `dot`, `linalg` |
| `matplotlib` | ≥ 3.7 | Charts and visualization | `pyplot`, `scatter`, `bar`, `savefig` |
| `PySastrawi` | ≥ 1.0 | Indonesian stemmer | `StemmerFactory`, `.createStemmer()` |
| `transformers` | ≥ 4.35 | HuggingFace models (Day 2+) | `AutoTokenizer`, `AutoModel`, `Trainer` |
| `datasets` | ≥ 2.14 | HuggingFace dataset loading (Day 2+) | `load_dataset`, `Dataset` |

**NLTK downloads required:**

```python
import nltk
nltk.download('punkt_tab')   # Tokenizer models (NLTK ≥ 3.9)
nltk.download('stopwords')   # Stopword lists (English, etc.)
nltk.download('punkt')       # Fallback tokenizer
```

---

## 8. Vocabulary / Glossary

| Term | Definition |
|---|---|
| **Token** | A single unit of text after splitting — typically a word, but can be a subword (in transformer models) or character. In industrial text, a token might be `"bearing"`, `"P-101"`, or `"preventive"`. Related: tokenization, vocabulary. |
| **Tokenization** | The process of splitting raw text into tokens. Simple tokenization splits on spaces; smarter tokenizers (like NLTK's `word_tokenize`) handle punctuation, contractions, and special characters. Every downstream NLP step depends on the quality of tokenization. |
| **Corpus** | A collection of text documents used for analysis or model training. In our context, a corpus might be all work orders from 2020–2024, or all safety incident reports from a specific plant. The corpus defines what your model knows. |
| **Vocabulary** | The set of all unique tokens across a corpus. A TF-IDF vocabulary of 500 terms means your documents are represented as 500-dimensional vectors. Vocabulary size is a direct trade-off between expressiveness and computational cost. |
| **TF-IDF** | Term Frequency–Inverse Document Frequency. A numerical statistic that reflects how important a word is to a document within a corpus. High TF-IDF = the word appears often in this document but rarely across the corpus. In maintenance text, `"bearing"` in a ticket about bearing failure will have high TF-IDF; `"maintenance"` in the same ticket will have low TF-IDF because it appears everywhere. |
| **Stopword** | A high-frequency word that carries little semantic meaning: prepositions (`"di"`, `"ke"`), conjunctions (`"dan"`, `"atau"`), auxiliary verbs (`"adalah"`, `"ada"`). Removing stopwords reduces noise in the feature space. The standard lists must be extended with domain-specific noise words for industrial text. |
| **Stemming** | Reducing an inflected word to its root form by removing affixes using rules, without reference to a dictionary. Indonesian stemming (PySastrawi) handles the complex prefix/suffix system: `"penggantian"` → `"ganti"`, `"memperbaiki"` → `"baik"`. May produce non-dictionary roots (e.g., English: `"better"` → `"bet"`). |
| **Lemmatization** | Reducing a word to its canonical dictionary form (lemma), using a lexicon. More accurate than stemming for English: `"better"` → `"good"`, `"running"` → `"run"`. For Indonesian, PySastrawi stemming and lemmatization produce similar results because the language's root-form system is well-defined. |
| **Embedding** | A dense vector representation of text in a continuous numerical space, where semantic similarity corresponds to geometric proximity. Unlike TF-IDF's sparse vectors, embeddings encode meaning: `"pompa"` and `"pump"` will have similar embeddings even if they never appear in the same document. |
| **Word2Vec** | A shallow neural model trained to predict context words from center words (Skip-gram) or vice versa (CBOW). Produces one fixed vector per word. Learns that `"pompa"` and `"kompressor"` are related because they appear near similar words (`"bocor"`, `"tekanan"`, `"vibrasi"`). Does not handle polysemy (one meaning per word). |
| **Transformer** | A neural network architecture built on self-attention mechanisms, where every token is contextualized by every other token in the sequence. The basis for BERT, GPT, T5, and virtually all state-of-the-art NLP models since 2018. More powerful than Word2Vec but computationally heavier. |
| **BERT** | Bidirectional Encoder Representations from Transformers. A transformer encoder pre-trained on masked language modeling and next-sentence prediction. Produces contextual embeddings — the same word gets different vectors in different sentences. Basis for IndoBERT and the multilingual models used in this course. |
| **Fine-tuning** | Taking a pre-trained model (which has learned general language patterns from a huge corpus) and continuing to train it on your specific task with your labeled data. Requires far less data than training from scratch. Analogy: a general engineer learns engineering fundamentals in university; fine-tuning is the on-the-job specialization for your specific plant. (Day 2 topic.) |
| **Cosine Similarity** | A measure of the angle between two vectors, ranging from 0 (perpendicular — completely unrelated) to 1 (same direction — identical meaning). Preferred over Euclidean distance for text because it is invariant to document length. `cosine_similarity("pompa bocor", "kebocoran pompa") ≈ 0.85`. |
| **K-Means** | An unsupervised clustering algorithm that partitions N documents into K groups by iteratively assigning each document to its nearest centroid, then recalculating centroids. Requires specifying K in advance. Non-deterministic (results vary with initialization). Used in tonight's lab to discover failure categories without labels. |
| **Silhouette Score** | A cluster quality metric from −1 to +1. For each document, it measures how similar that document is to its own cluster (a) versus the nearest other cluster (b): `(b − a) / max(a, b)`. A score near 1 means the document clearly belongs to its cluster; near 0 means it sits between two clusters; negative means it was probably assigned to the wrong cluster. |
| **Semantic Search** | Retrieval based on meaning and conceptual similarity rather than exact keyword matching. A query for `"pompa sentrifugal vibrasi berlebih"` will find documents about `"getaran abnormal centrifugal pump"` even with no shared words — because their embedding vectors are close. Enables knowledge Q&A over technical documents (Day 4). |
| **RAG** | Retrieval-Augmented Generation. A technique that combines semantic search over a document database with LLM text generation: the model retrieves relevant documents, then uses them as context to answer a question. Enables chatbots that answer questions about your specific plant manuals and procedures. (Day 4 topic.) |

---

## 9. Formulas Quick Reference

Each formula below is followed by a plain-English interpretation and guidance on what high or low values mean in practice.

**TF-IDF computation — worked example:**

Let corpus size N = 100 tickets, and consider term `"bearing"` in Ticket #7 (20 words):
- TF("bearing", doc7) = 3/20 = **0.15** *(appears 3 times out of 20 words)*
- IDF("bearing") = log(100/10) = **2.30** *(appears in 10 of 100 documents — moderately rare)*
- TF-IDF = 0.15 × 2.30 = **0.345** *(strong signal: this ticket is specifically about bearings)*

Compare with `"peralatan"` (equipment), which appears in 80 of 100 tickets:
- IDF("peralatan") = log(100/80) = **0.22** *(very common — weak discriminator)*
- Even if TF is high, TF-IDF stays low: **a ubiquitous term cannot distinguish documents.**

| Formula | Plain-English Meaning |
|---|---|
| `TF(t,d) = count(t,d) / len(d)` | What fraction of this document's words are term t? High value = the document talks a lot about t. |
| `IDF(t) = log( N / df(t) )` | How rare is term t across the whole corpus? High value = t appears in few documents = more distinctive. |
| `TF-IDF(t,d) = TF × IDF` | Combined score: how important is t specifically for document d? High = frequent in this doc, rare elsewhere. |
| `cos(A,B) = (A·B) / (‖A‖ × ‖B‖)` | Cosine of the angle between two document vectors. 1.0 = identical direction (same topic, same vocabulary proportions). 0.0 = perpendicular (completely unrelated). Length of vectors does not matter. |
| `Silhouette = (b − a) / max(a, b)` | How well does a document fit its cluster? a = average distance to cluster-mates; b = average distance to the nearest other cluster. Close to 1 = clearly belongs; close to 0 = on the boundary; negative = probably misassigned. |
| `Accuracy = (TP + TN) / (TP + TN + FP + FN)` | Fraction of all predictions that were correct. Misleading when classes are imbalanced (e.g., 95% of tickets are non-critical — a model that always predicts non-critical achieves 95% accuracy but is useless). |
| `F1 = 2 × (Precision × Recall) / (Precision + Recall)` | Harmonic mean of precision (of predicted positives, how many were correct?) and recall (of actual positives, how many did we catch?). Preferred metric for imbalanced classes. F1 = 1.0 is perfect; F1 < 0.5 is poor. |

---

## 10. Resources & Further Reading

**Official Documentation:**

- scikit-learn (TF-IDF, K-Means): `https://scikit-learn.org/stable/`
- NLTK: `https://www.nltk.org/`
- Gensim (Word2Vec): `https://radimrehurek.com/gensim/`
- sentence-transformers: `https://www.sbert.net/`
- HuggingFace Hub: `https://huggingface.co/`

**Indonesian NLP:**

- PySastrawi (Indonesian stemmer): `https://github.com/har07/PySastrawi`
- IndoBERT: `https://huggingface.co/indobenchmark/indobert-base-p1`
- IndoNLU benchmark: `https://github.com/IndoNLP/indonlu`
- Indonesian NLP resources: `https://github.com/keyiflerolsun/Turkce-NLP` *(similar multilingual pattern)*

**Key Papers (abstract only for now):**

| Paper | Authors | Year | Key Contribution |
|---|---|---|---|
| "Efficient Estimation of Word Representations in Vector Space" | Mikolov et al. | 2013 | Word2Vec |
| "GloVe: Global Vectors for Word Representation" | Pennington et al. | 2014 | GloVe embeddings |
| "BERT: Pre-training of Deep Bidirectional Transformers" | Devlin et al. | 2019 | BERT architecture |
| "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" | Reimers & Gurevych | 2019 | sentence-transformers |

**Practice Datasets:**

- Indonesian Twitter NLP: `https://github.com/IndoNLP/indonlu`
- Industrial maintenance text: Search "CMMS maintenance text dataset" on Kaggle
- Safety incident data: Search "safety incident report dataset" on Kaggle

---

## 11. Day 2 Preview

**Topic:** Document Classification & Supervised NLP

Tonight you worked entirely without labels — K-Means clustered your documents based on vocabulary patterns, and you assigned human-readable names to those clusters by reading the top terms. That is unsupervised learning: the algorithm discovers structure; you interpret it.

Day 2 flips the model. You will start with labeled examples — tickets where a human expert has already assigned a severity level (Low / Medium / High / Critical). A supervised classifier will learn from those labels and then predict the correct category for new, unseen tickets automatically. The bridge between tonight and tomorrow is this: the clusters you built tonight can become the training labels for tomorrow. Manual cluster inspection → human labeling → supervised model.

**What transfer learning means, intuitively:** Training a language model from scratch requires billions of words and weeks of GPU time. Transfer learning means you skip that step entirely. You take a model that has already learned the structure of language — grammar, semantics, context — from a massive general corpus, and you give it a small additional training task on your specific domain. The model transfers its general language knowledge and adapts it to your specific classification problem. Think of it like hiring an engineer who already has a degree (pre-trained knowledge) and giving them a two-week orientation at your plant (fine-tuning) instead of sending them back to university.

| Item | Detail |
|---|---|
| **Core concept** | Transfer learning — adapt pre-trained BERT to your specific task |
| **Use case** | Classify safety incident reports by severity level (Low / Medium / High / Critical) |
| **New concepts** | Transfer learning, fine-tuning, F1 score, confusion matrix, label imbalance |
| **New tools** | HuggingFace `transformers`, `Trainer` API, `datasets` library |
| **What you'll build** | Fine-tuned multilingual BERT safety report classifier |

**Prepare (optional):**
- Explore Indonesian models: `https://huggingface.co/models?language=id`
- Install tonight: `pip install transformers datasets accelerate`
- Read abstract: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

**How Day 2 connects to tonight:**
> Tonight you manually labeled K-Means clusters by reading top terms.
> Tomorrow you'll train a model to label documents automatically — supervised learning.

---

*Day 1 — Introduction to Text Mining | Industrial AI & LLM Training Program*
*Handout prepared for in-session reference and post-session review.*
