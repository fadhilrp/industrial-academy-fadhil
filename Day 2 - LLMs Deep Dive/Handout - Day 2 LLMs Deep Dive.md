---
tags: [handout, day2, llm, reference]
session: "Day 2 — Large Language Models Deep Dive"
program: "Industrial AI & LLM Training Program"
export: "PDF-ready — export via Obsidian or Pandoc"
---

# Day 2 Handout — Large Language Models Deep Dive
*Industrial AI & LLM Training Program*

---

## 1. The LLM Revolution in Industrial AI

Large Language Models represent a qualitative shift in what software can do with text. Before LLMs, processing text required building task-specific pipelines: a tokenizer, a feature extractor, a trained classifier, a named-entity recognizer — each component trained separately, each requiring labelled data. A maintenance ticket classifier trained on English SAP records could not classify Indonesian shift notes without retraining. A keyword-based search system could not find "centrifugal pump failure" when the ticket said "pompa sentrifugal rusak."

LLMs collapse this complexity into a single interface. A model pre-trained on hundreds of billions of words across dozens of languages has internalized grammar, domain knowledge, logical reasoning patterns, and cross-linguistic semantic relationships. You do not train it on your data. You describe what you want in plain language — a prompt — and the model does it.

> **What changed in 2022–2024:** The combination of instruction tuning (fine-tuning on human-written task descriptions) and RLHF (Reinforcement Learning from Human Feedback) transformed base language models into reliable instruction-followers. Before instruction tuning, you could not simply tell a model "categorize this ticket as Mechanical, SAP/ERP, Network/IT, or Safety" — you had to fine-tune it on thousands of labelled examples. After instruction tuning, zero-shot categorization works reliably out of the box.

**What is now possible in industrial AI:**

| Task | Before LLMs | With LLMs |
|---|---|---|
| Ticket categorization | Train classifier per category | Zero-shot: describe categories in prompt |
| Information extraction | Named entity recognition pipeline | Prompt: "extract equipment ID, failure mode, action" |
| Cross-language retrieval | Separate models per language | Single multilingual model |
| Root cause analysis | Manual expert review | LLM reasoning over linked tickets |
| Report drafting | Templates + manual writing | LLM with structured data input |
| Q&A over documents | Keyword search | RAG + LLM (Day 4) |

The practical impact: tasks that previously required months of data collection, labelling, and model training can now be prototyped in hours. The bottleneck is no longer model building — it is prompt design and validation.

---

## 2. Tokens and Context Windows

### What is a token?

LLMs do not read characters or words. They read **tokens** — variable-length text chunks produced by a subword tokenizer. Tokens are the atomic unit of input and output: every cost calculation, every context limit, and every latency measurement is expressed in tokens.

Common patterns:
- Common English words: 1 token (`"pump"`, `"failure"`, `"bearing"`)
- Long or uncommon words: 2–4 tokens (`"overheating"` → `"over"` + `"heat"` + `"ing"`)
- Numbers: variable (`"42"` → 1 token; `"87013019"` → 3–4 tokens)
- Indonesian words: 2–5 tokens each (less common in training data → more splits)
- Equipment IDs: 3–6 tokens (`"P-101"` → `"P"` + `"-"` + `"101"`)

### Byte-Pair Encoding (BPE)

BPE is the most common tokenization algorithm used by GPT, Claude, and Llama models. It builds a subword vocabulary by iteratively merging the most frequent adjacent byte pairs in the training corpus:

```
Step 0:  ['p', 'o', 'm', 'p', 'a']          (character level)
Step 1:  ['po', 'm', 'p', 'a']               (merge most common pair: p+o)
Step 2:  ['pom', 'pa']                        (merge: m+p becomes mp, etc.)
Step 3:  ['pompa']                            (if the full word is frequent enough)
```

The final vocabulary contains ~50,000–130,000 entries depending on the model. Words that appear frequently in the training corpus get their own single token; rare words get split into pieces. This means:

- **English text** (dominant in training data) tokenizes efficiently — ~1 token per word
- **Indonesian text** tokenizes less efficiently — ~1.3–1.5 tokens per word for common words, more for technical vocabulary
- **Mixed-language text** (code-switching, common in Indonesian industrial tickets) pays an overhead for each language switch

### Why the Indonesian premium matters at scale

A manufacturing plant processing 10,000 support tickets per month faces different cost profiles depending on ticket language:

| Ticket language | Avg tokens/ticket | Monthly tokens | Cost at Haiku ($0.25/M) |
|---|---|---|---|
| English only | ~35 | 350,000 | $0.09 |
| Indonesian only | ~45 | 450,000 | $0.11 |
| Mixed (realistic) | ~40 | 400,000 | $0.10 |

At these volumes, even Sonnet ($3/M) costs under $2/month — a rounding error. Tokens matter most at 100M+/month scale (enterprise-wide deployment across multiple plants).

### Context windows

The context window is the maximum number of tokens the model can process in a single call — both input and output combined. Exceeding it raises an error.

| Model | Context window | Practical implication |
|---|---|---|
| Claude Haiku 4.5 | 200,000 tokens | ~150,000 words — full novels fit |
| Claude Sonnet 4.6 | 200,000 tokens | Entire maintenance logs, multi-year history |
| GPT-3.5 Turbo | 16,385 tokens | ~12,000 words — watch for long conversations |
| GPT-4o | 128,000 tokens | Most industrial documents |
| Llama 3.2 (3B) | 128,000 tokens | Good for local deployment |

For industrial ticket triage (40–200 words per ticket), context is rarely a constraint. It becomes critical for RAG (Day 4) where retrieved context chunks fill the window.

---

## 3. Transformer Architecture Essentials

### The core insight: attention

The transformer architecture's key innovation is **attention** — a mechanism that lets each token in the input look at every other token and decide how much to weight its information when computing its representation.

Before transformers, RNNs processed text sequentially: token 1 → token 2 → ... → token N. Information from early tokens "faded" by the time the model processed token N. Transformers process all tokens simultaneously, with each token attending to all others — no fading, no sequential bottleneck.

**Analogy:** Imagine reading a maintenance report by scanning the whole page at once, drawing lines between every related word. The word "bearing" draws a strong line to "vibration," "replacement," "pump," and "P-101." Weak lines to "SAP," "login," "network." The attention mechanism learns which connections are informative by optimizing over billions of examples.

### What happens inside a transformer (simplified)

```
Input text
    │
    ▼
┌─────────────────────────────────────┐
│  TOKENIZER                          │  Text → token IDs
│  "Pompa P-101 vibrasi" → [12043, 47, ...]
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  EMBEDDING LAYER                    │  Token IDs → dense vectors (e.g. 4096-dim)
│  Each token gets a learned vector   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  TRANSFORMER BLOCKS (×N layers)     │
│   ├── Multi-head self-attention     │  Each token attends to all others
│   ├── Layer normalization           │  Stabilizes training
│   ├── Feed-forward network          │  Non-linear transformation
│   └── Residual connections          │  Preserves gradient flow
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  OUTPUT HEAD                        │  Final vector → probability over vocabulary
│  argmax → next token prediction     │
└─────────────────────────────────────┘
    │
    ▼
Generated text (one token at a time)
```

### Encoder vs. decoder models

| Architecture | Direction | Examples | Best for |
|---|---|---|---|
| **Encoder-only** | Bidirectional (sees full context) | BERT, RoBERTa | Classification, embeddings, NER |
| **Decoder-only** | Left-to-right (autoregressive) | GPT-4, Claude, Llama | Text generation, chat, reasoning |
| **Encoder-decoder** | Both directions | T5, BART | Translation, summarization |

Claude, GPT-4, and Llama are all decoder-only models. They generate tokens one at a time, each token conditioned on all previous tokens. BERT (used in Day 1's sentence-transformers) is encoder-only — it cannot generate text, but produces excellent embeddings because it sees the full context bidirectionally.

---

## 4. Instruction Tuning and RLHF

### Base model vs. instruction-tuned model

A **base model** is trained purely on next-token prediction over a massive text corpus. It is excellent at continuing text patterns ("the pump was vibrating, so the technician...") but unreliable for following instructions. Ask it to "categorize this ticket" and it may continue with more ticket text rather than providing a category.

An **instruction-tuned model** is further trained on (instruction, ideal response) pairs, teaching it to follow task descriptions. Ask it to categorize a ticket and it categorizes a ticket.

**RLHF (Reinforcement Learning from Human Feedback)** adds a third stage: human raters score multiple model responses for the same prompt, a reward model learns from those ratings, and the LLM is fine-tuned to maximize the reward model's score. This aligns the model's outputs with human preferences: responses become more helpful, more accurate, and less likely to produce harmful content.

The practical result: Claude, GPT-4, and instruction-tuned Llama models follow your system prompts reliably and produce consistent, well-formatted outputs when given clear instructions.

### Why system prompts work

System prompts work because instruction tuning taught the model to treat them as behavioral constraints. When you write:

```
You are an industrial support triage assistant.
Respond ONLY with valid JSON in this exact format:
{"category": "...", "priority": "...", "reason": "..."}
```

The model has been trained on thousands of examples where following similar instructions produced high reward scores. It treats this as a strong behavioral prior for the entire conversation. The more specific and consistent your system prompt, the more reliably the model follows it.

---

## 5. The Chat API Interface

### Message structure

Every modern LLM API uses a messages array where each element has a `role` and `content`:

```python
messages = [
    {"role": "user",      "content": "What does LOTO stand for?"},
    {"role": "assistant", "content": "LOTO stands for Lockout/Tagout..."},
    {"role": "user",      "content": "When is it required?"},
]
```

The model processes this array as a unified context — it sees the complete conversation history and generates the next `assistant` turn. This is how multi-turn conversations work: each new user message is appended to the array.

### Key parameters

| Parameter | What it does | Recommendation for triage |
|---|---|---|
| `temperature` | Randomness of token selection (0 = deterministic, 1+ = creative) | **0.0** — classification must be consistent |
| `max_tokens` | Maximum output length in tokens | **256–512** — JSON output is short |
| `top_p` | Nucleus sampling threshold (alternative to temperature) | Keep default (1.0) when using temperature |
| `stop` | List of strings that stop generation | `["}"]` to stop after JSON closes |
| `system` | System prompt (Anthropic) or system-role message (OpenAI) | Set your role + format instructions here |

### The usage object — tracking cost

Every response includes a usage object with exact token counts for that call:

```python
# Anthropic
response.usage.input_tokens   # tokens in your prompt (billed)
response.usage.output_tokens  # tokens in the response (billed)

# OpenAI
response.usage.prompt_tokens
response.usage.completion_tokens
response.usage.total_tokens
```

Always collect these in production. Sum them across all calls to track actual spend vs. estimates.

### Stop reasons

| Stop reason | Meaning | Action |
|---|---|---|
| `end_turn` | Model finished naturally | Expected — use the response |
| `max_tokens` | Output was cut off at max_tokens limit | Increase max_tokens or reduce prompt |
| `stop_sequence` | Hit a stop string you defined | Expected if you set stop sequences |
| `tool_use` | Model called a function | Parse the tool_use block |

---

## 6. Prompt Engineering Patterns

Prompt engineering is the discipline of writing inputs that reliably produce the outputs you need. Below are all four patterns used in tonight's lab, with full industrial examples and quality comparison.

### Pattern 1 — Zero-Shot

Provide the task description and ask directly. No examples.

```
Categorize this maintenance ticket into one of these categories:
Mechanical, SAP/ERP, Network/IT, Safety.

Ticket: Prosedur LOTO belum dilakukan sebelum teknisi masuk ke dalam tangki pembersihan

Category:
```

**Output:** `Safety` (correct, but may vary in format)

**Strengths:** Minimal prompt length (fewer tokens, lower cost). No examples to maintain.
**Weaknesses:** Output format is unpredictable. May return "This is a Safety ticket" instead of "Safety". May explain reasoning when you only want the category.

**When to use:** Quick prototyping, simple tasks with obvious answers, when token cost is critical.

---

### Pattern 2 — Few-Shot

Provide 2–8 (instruction, ideal output) pairs before the query. The model learns the output pattern from examples.

```
Categorize each ticket. Reply with ONLY the category name.

Ticket: Pompa sentrifugal P-101 vibrasi berlebihan, perlu ganti bearing segera
Category: Mechanical

Ticket: Error ME21N saat buat purchase order vendor master belum disetujui
Category: SAP/ERP

Ticket: SCADA server tidak bisa connect ke PLC area A network timeout
Category: Network/IT

Ticket: Near miss operator hampir tertimpa material jatuh dari rak gudang
Category: Safety

Ticket: Prosedur LOTO belum dilakukan sebelum teknisi masuk ke dalam tangki pembersihan
Category:
```

**Output:** `Safety` (consistently, exactly one word)

**Strengths:** Output format is learned from examples. Works with any model without fine-tuning. Handles format nuances (capitalisation, exact wording) that zero-shot misses.
**Weaknesses:** Longer prompt (4 examples × ~30 tokens = ~120 extra tokens per call). Examples must be carefully chosen to cover the output space.

**When to use:** When zero-shot produces inconsistent formats. When you have clear examples of what good output looks like.

---

### Pattern 3 — Chain-of-Thought (CoT)

Ask the model to reason step by step before answering. Critical for ambiguous tickets.

```
Categorize this industrial ticket. Think step by step:
1. What is the main subject? (equipment, system, person, process)
2. What keywords indicate the category?
3. What is the final category?

Categories: Mechanical, SAP/ERP, Network/IT, Safety

Ticket: Prosedur LOTO belum dilakukan sebelum teknisi masuk ke dalam tangki pembersihan

Reasoning:
```

**Output:**
```
1. Main subject: A safety procedure (LOTO) that was not followed before a technician
   entered a cleaning tank — this is about worker safety, not equipment or IT.
2. Keywords: "LOTO" (Lockout/Tagout — a mandatory safety procedure), "teknisi masuk
   ke dalam tangki" (technician entering a tank — confined space entry requires permits)
3. Category: Safety
```

**Strengths:** Reasoning trace is auditable. Dramatically improves accuracy on edge cases and cross-category ambiguous tickets. Reveals model's knowledge (or ignorance) of domain concepts.
**Weaknesses:** Output is 3–5× longer (more output tokens = more cost). Slower. Harder to parse programmatically.

**When to use:** High-stakes decisions (safety classification, escalation routing). Ambiguous tickets where multiple categories are plausible. Debugging — understand why the model is misclassifying.

---

### Pattern 4 — System Prompt Design

Assign a persona, define the output contract, and enforce it via the system message.

```
SYSTEM:
You are an industrial support triage assistant for a manufacturing plant.
Your job is to categorize maintenance and support tickets.

Categories:
- Mechanical: equipment failures, vibration, bearing, pump, motor, valve issues
- SAP/ERP: SAP transactions, purchase orders, goods receipt, user access, workflow
- Network/IT: server, network, connectivity, SCADA, printer, VPN, database issues
- Safety: near misses, APD violations, LOTO, permits, spills, fire alarms

Priority levels:
- Critical: immediate safety risk or production stoppage
- High: impacts production within 24 hours
- Medium: impacts operations but workaround exists
- Low: no immediate operational impact

Respond ONLY with valid JSON in this exact format:
{"category": "...", "priority": "...", "reason": "one sentence explanation"}

USER:
Ticket: Prosedur LOTO belum dilakukan sebelum teknisi masuk ke dalam tangki pembersihan
```

**Output:**
```json
{"category": "Safety", "priority": "Critical", "reason": "LOTO procedure was not followed before a technician entered a confined space, creating an immediate risk of serious injury or fatality."}
```

**Strengths:** Machine-parseable output. Category description in system prompt reduces ambiguity. Priority assigned automatically. Consistent across all tickets.
**Weaknesses:** Longest prompt (system prompt adds ~150 tokens per call). JSON parsing can fail if model deviates — need error handling.

**When to use:** Production systems where output feeds downstream code. When you need structured data (category, priority, metadata) not just a label. When consistency across thousands of calls matters.

---

### Prompt engineering rules of thumb

1. **Be specific, not clever.** "Reply with ONLY the category name, nothing else" works better than "Be concise."
2. **Show, don't just tell.** Few-shot examples outperform elaborate descriptions.
3. **Temperature = 0.0 for classification.** Every time.
4. **Put the most important instructions first and last.** Models pay more attention to the beginning (primacy) and end (recency) of prompts.
5. **Test on your hardest cases.** If your prompt handles the ambiguous tickets, it handles everything.
6. **Add negative constraints.** "Do not explain your reasoning" is often more effective than "Be brief."

---

## 7. Function Calling and Structured Output

### What function calling is

Function calling (Anthropic calls it "tool use") is an API feature where you define a function schema in JSON Schema format, pass it to the model, and the model returns structured data matching that schema instead of free text.

The model does not actually execute code. It decides when your function should be called and populates its parameters. Your application then calls the actual function with those parameters.

### Tool schema anatomy

```python
tool_schema = {
    "name": "categorize_ticket",
    "description": "Categorize an industrial support ticket with priority and action required.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["Mechanical", "SAP/ERP", "Network/IT", "Safety"],
                "description": "Ticket category"
            },
            "priority": {
                "type": "string",
                "enum": ["Critical", "High", "Medium", "Low"],
                "description": "Urgency level"
            },
            "equipment_id": {
                "type": "string",
                "description": "Equipment ID mentioned (e.g. P-101, GB-103) or null if none"
            },
            "action_required": {
                "type": "string",
                "description": "Concise action required to resolve the ticket"
            }
        },
        "required": ["category", "priority", "action_required"]
    }
}
```

Key elements:
- **`name`** — identifier; the model uses this to indicate which function it is calling
- **`description`** — natural-language description that helps the model decide when to use the tool
- **`input_schema`** — JSON Schema defining the parameters; `enum` constraints limit valid values
- **`required`** — parameters the model must always populate; others may be null/absent

### Parsing the response

```python
# Anthropic tool use response
for block in response.content:
    if block.type == "tool_use":
        result = block.input  # Already a Python dict — no JSON parsing needed
        category = result["category"]
        priority = result["priority"]

# OpenAI function call response
fn_args = response.choices[0].message.function_call.arguments
result = json.loads(fn_args)  # JSON string → dict
```

### Function calling vs. JSON system prompt

Both approaches can produce structured output. Use function calling when:
- You need **schema validation** guaranteed by the API (enum constraints enforced)
- You need **nullable optional fields** without JSON parse errors
- You are building a **multi-tool system** where the model chooses which function to call

Use JSON system prompt when:
- You need **provider portability** (works with Ollama, which doesn't support tool use)
- The schema is **simple** (2–3 fields, all required)
- You want **lower latency** (function calling adds a small overhead)

---

## 8. Model Selection Framework

### The accuracy/latency/cost triangle

No model wins on all three dimensions simultaneously. Selecting the right model is a design decision based on your workload's constraints.

| Model | Input cost ($/1M tok) | Typical latency | Context | Best use case |
|---|---|---|---|---|
| Claude Haiku 4.5 | $0.25 | <1s | 200k | High-volume triage, dev/test |
| Claude Sonnet 4.6 | $3.00 | 1–3s | 200k | Complex extraction, reasoning |
| GPT-3.5 Turbo | $0.50 | <1s | 16k | High-volume English tasks |
| GPT-4o | $2.50 | 2–5s | 128k | Multimodal, complex docs |
| Llama 3.2 3B (Ollama) | Free | 0.5–2s* | 128k | Air-gapped, privacy-sensitive |
| Qwen2.5 7B (Ollama) | Free | 1–3s* | 128k | Indonesian-heavy local workloads |

*Ollama latency depends on local hardware (CPU vs. GPU, RAM availability)

### Decision framework

**Step 1: Define your accuracy requirement**
What error rate is acceptable? Safety classification at a nuclear facility: near-zero. Routing support tickets in a warehouse: 90% is probably fine.

**Step 2: Set your budget constraint**
Monthly ticket volume × avg tokens per ticket × price per token = monthly cost. Calculate for each model candidate.

**Step 3: Measure actual latency**
Run 50 real calls with `time.time()`. p50 and p95 matter more than average. Latency varies by time of day on shared cloud APIs.

**Step 4: Check data privacy requirements**
If tickets contain PII (names, employee IDs) or production secrets, consider:
- Anthropic/OpenAI: data is not used for training (enterprise tier)
- Ollama: data never leaves your network — strongest privacy guarantee

**Step 5: Start with the best, optimize down**
Develop and validate on Sonnet. Measure accuracy. Then test on Haiku with the same prompts — if accuracy is acceptable, switch to Haiku for production. The cost savings are significant at scale.

---

## 9. Indonesian Language Support

### Multilingual LLM capabilities

Claude, GPT-4, and Llama 3 were all trained on large multilingual corpora. Indonesian (Bahasa Indonesia) is well-represented — estimated 0.5–1% of training tokens for frontier models, which corresponds to billions of words.

**Practical capabilities:**
- Reading and writing fluent Indonesian ✓
- Understanding Indonesian SAP error messages ✓
- Handling code-switching (mixed Indonesian/English) ✓
- Understanding Indonesian abbreviations (APD, LOTO, dll) ✓
- Industrial domain vocabulary in Indonesian ✓

**Current limitations:**
- Very informal or regional slang may be less reliable
- Highly specialized technical terms with no Indonesian equivalent may be transliterated differently per model
- Response latency may be slightly higher for non-English inputs (larger token count = more processing)

### Code-switching handling

Indonesian industrial text frequently mixes Indonesian and English — technical terms, equipment IDs, SAP transaction codes, and English abbreviations appear within Indonesian sentences:

> *"SCADA server tidak bisa connect ke PLC area A network timeout perlu cek switch"*

LLMs handle this naturally because the multilingual training data includes authentic mixed-language text from the internet. The model understands that "SCADA," "PLC," "network timeout," and "switch" are technical English terms embedded in an Indonesian sentence about IT infrastructure.

### Prompt language choice

For Indonesian-speaking participants using the API:
- **System prompt:** Can be in English or Indonesian — both work. English system prompts are slightly more reliable for format constraints (the model has seen more English instruction-following examples in training).
- **User messages:** Use whatever language your data is in — do not translate tickets before sending them. Sending in the original language preserves nuance and domain terminology.
- **Output language:** Explicitly specify if you need Indonesian output: `"Respond in Bahasa Indonesia."` Otherwise, the model may respond in English when given an English system prompt.

---

## 10. Cost Management at Scale

### Token pricing reference (February 2026)

| Model | Input ($/1M) | Output ($/1M) | Typical ratio |
|---|---|---|---|
| Claude Haiku 4.5 | $0.25 | $1.25 | Output ~4× input cost |
| Claude Sonnet 4.6 | $3.00 | $15.00 | Output ~5× input cost |
| GPT-3.5 Turbo | $0.50 | $1.50 | Output ~3× input cost |
| GPT-4o | $2.50 | $10.00 | Output ~4× input cost |

*Note: prices change frequently. Always check provider pricing pages for current rates.*

**Key insight:** Output tokens cost significantly more than input tokens. Minimize output length by:
- Specifying a tight output format (JSON with required fields only)
- Using `max_tokens` to cap output
- Adding: "Do not explain your reasoning unless asked"

### Batching

Most LLM APIs support sending multiple requests in parallel (async) or in batch mode. For 40 tickets:
- **Sequential:** 40 calls × 1s = 40s total
- **Concurrent (10 workers):** ~4–5s total

```python
import asyncio
import anthropic

async def categorize_batch(tickets, concurrency=10):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    async def call_one(ticket):
        async with semaphore:
            response = await client.messages.create(...)
            return response.content[0].text

    return await asyncio.gather(*[call_one(t) for t in tickets])
```

### Prompt caching

Anthropic supports **prompt caching** for system prompts and long context that is reused across many calls. If your system prompt is identical across all 40 ticket calls, caching it reduces input token costs by ~90% for cached content after the first call.

```python
# Add cache_control to static content
client.messages.create(
    system=[{
        "type": "text",
        "text": TRIAGE_SYSTEM,
        "cache_control": {"type": "ephemeral"}
    }],
    ...
)
```

For 40 tickets with a 150-token system prompt, caching saves 39 × 150 × $0.25/1M = $0.0000015. Negligible at this scale. At 100,000 tickets/day, savings become meaningful.

### Rate limits

| Provider | Free tier | Paid tier |
|---|---|---|
| Anthropic | 5 RPM, 10k TPM | Up to 4,000 RPM, 400k TPM |
| OpenAI | 3 RPM, 40k TPM | Scales with usage tier |
| Ollama | Unlimited (local) | No rate limits |

RPM = Requests Per Minute. TPM = Tokens Per Minute.

For lab use: add `time.sleep(0.1)` between sequential calls to avoid burst-rate limits on free tier keys. For production at scale: implement exponential backoff on rate limit errors (HTTP 429).

---

*Day 2 — LLMs Deep Dive | Industrial AI & LLM Training Program*
