---
tags: [handout, day5, deployment, production-systems, inference, fine-tuning, reference]
session: "Day 5 — LLM Deployment & Production Systems"
program: "Industrial AI & LLM Training Program"
export: "PDF-ready — export via Obsidian or Pandoc"
---

# Day 5 Handout — LLM Deployment & Production Systems
*Industrial AI & LLM Training Program*

---

## 1. Executive Summary

Over the previous four days, you built a complete AI pipeline for industrial maintenance operations. Day 1 introduced text mining fundamentals and the 40-ticket dataset — synthetic support tickets across four categories (Mechanical, Safety, Normal/Electrical, Komplex) drawn from real patterns in pump, motor, and conveyor maintenance. Day 2 took you deep into LLM internals: tokenization, attention mechanisms, and prompt engineering for structured industrial outputs. Day 3 added Retrieval-Augmented Generation so your system could answer questions grounded in your own maintenance manuals rather than hallucinating. Day 4 brought orchestration: LangGraph workflows, multi-agent routing, and automated ticket triage that can run a full classification-and-response loop without human intervention.

Day 5 closes the gap between a working demo and a production system.

**What you will learn today:**

- **Inference at scale** — vLLM’s PagedAttention and continuous batching enable 5–20x throughput over naive Transformers. You will run an OpenAI-compatible API server locally in under ten minutes.
- **Local deployment** — Ollama makes running Llama 3.2, Qwen 2.5, and Mistral on a laptop practical. This matters for Indonesian industrial facilities where data residency and BSSN compliance prohibit sending maintenance logs to external APIs.
- **Model quantization** — A 7B-parameter model that requires 28 GB in FP32 fits on a 4 GB GPU in INT4. You will understand the quality/memory trade-off at each precision level.
- **Production engineering patterns** — Token budget tracking, cost-aware routing, exponential backoff, circuit breakers, and semantic caching reduce cost and improve reliability.
- **Monitoring and observability** — LangSmith for LLM tracing, Prometheus + Grafana for infrastructure metrics, alerting when p95 latency exceeds 3 seconds or error rates climb above 5%.
- **Fine-tuning with Unsloth** — Supervised fine-tuning, continued pretraining on domain text, and RLHF via Direct Preference Optimization. Unsloth’s 2x speed and 70% VRAM reduction make this feasible on a free Colab T4 GPU.
- **End-to-end architecture** — A reference diagram tying preprocessing, routing, RAG, monitoring, and the fine-tuning loop into a single maintainable system.

**The practical goal:** by the end of this session, you will have a blueprint for deploying your maintenance AI system at a real industrial facility — whether that is a cement plant in East Java, a petrochemical refinery in Riau, or an automotive assembly plant in Karawang.

---

## 2. The Production Gap

### 2.1 Why Notebooks Fail in Production

A Jupyter notebook is an ideal learning and experimentation environment. It is a poor production system. The following failures are predictable when a notebook is promoted directly to production without re-engineering.

| Failure Mode | What Happens in Practice | Industrial Consequence |
|---|---|---|
| No concurrency | Sequential execution blocks on each API call | A queue of 200 overnight tickets takes 6 hours instead of 20 minutes |
| No error handling | One API timeout kills the entire run | Monday morning: zero tickets processed |
| No cost control | Accidentally classifying the same 40 tickets 500 times | Unexpected $300 API bill for what should cost $2 |
| No versioning | You update a prompt; last week’s results are now unreproducible | Audit trail for safety tickets (S category) is broken |
| No monitoring | You have no idea if accuracy has drifted | Safety-critical misclassifications go undetected for weeks |
| No authentication | Anyone with the notebook URL can invoke your API key | Credential leak |
| No rate limiting | Burst of 500 requests hits API rate limit; all fail with 429 | System appears to work in testing, fails at scale |
| Memory leaks | LangGraph state grows without bounds in long-running sessions | OOM crash after 8 hours |

### 2.2 The Production Readiness Checklist

Before any LLM-powered system touches real industrial data, verify each item below:

```
PRODUCTION READINESS CHECKLIST
================================
Infrastructure
  [ ] API keys stored in environment variables, not code
  [ ] Secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager)
  [ ] Containerized deployment (Docker / Kubernetes)
  [ ] Health check endpoint (/health returns 200 or 503)
  [ ] Graceful shutdown handling (SIGTERM)

Reliability
  [ ] Retry logic with exponential backoff on all API calls
  [ ] Circuit breaker for downstream LLM services
  [ ] Request timeout enforced (never wait > 30s for a response)
  [ ] Fallback response when model is unavailable
  [ ] Dead letter queue for failed ticket processing

Cost Control
  [ ] Token budget tracker per request and per day
  [ ] Cost-aware routing (simple -> cheap model, complex -> full model)
  [ ] Semantic caching for repeated or near-duplicate queries
  [ ] Daily spend alert threshold configured

Observability
  [ ] Structured logging (JSON, not print())
  [ ] Distributed tracing (LangSmith or OpenTelemetry)
  [ ] Metrics exported to Prometheus
  [ ] Dashboard in Grafana with SLA gauges
  [ ] Alert rules: p95 latency, error rate, cost/day

Data & Compliance
  [ ] PII detection before sending to external APIs
  [ ] Data residency documented (BSSN compliance if applicable)
  [ ] Prompt injection detection on user-supplied inputs
  [ ] Audit log for every classification decision

Accuracy
  [ ] Evaluation dataset (at minimum, your 40 tickets with known labels)
  [ ] Automated eval on every model/prompt change
  [ ] Drift detection on production outputs
  [ ] Human review queue for low-confidence predictions
```

---

## 3. Local vs Cloud Deployment

### 3.1 The Decision Matrix

There is no universally correct deployment option. The right choice depends on your regulatory environment, budget, team capability, and data sensitivity.

| Criterion | BaaS (Anthropic / OpenAI) | Self-Hosted Cloud (vLLM on GPU VM) | Local (Ollama) | Hybrid |
|---|---|---|---|---|
| **Setup cost** | Near-zero | High (GPU VM + DevOps) | Low (laptop/edge) | Medium |
| **Running cost** | Pay-per-token | Fixed GPU VM cost | Near-zero | Variable |
| **Typical cost at 10k tickets/day** | $5–$50/day | $30–$80/day (A100) | $0 (electricity) | $5–$20/day |
| **Data leaves facility** | Yes | Yes (cloud provider) | No | Partial |
| **BSSN compliance** | Difficult | Possible (gov cloud) | Yes | Configurable |
| **Model quality** | Best-in-class | Good (7B–70B OSS) | Good (7B–13B) | Best of both |
| **Latency** | 500ms–3s | 100ms–1s | 50ms–500ms | 100ms–2s |
| **Max throughput** | High (managed) | Very high (configurable) | Low–Medium | High |
| **Offline capability** | No | No | Yes | Partial |
| **Maintenance burden** | Very low | High | Low | Medium |
| **Recommended for** | Prototypes, low volume | High-volume batch, SLA-critical | Dev/test, privacy, offline | Most production systems |

### 3.2 Indonesian Industrial Compliance Context

Indonesian facilities must consider two regulatory frameworks:

**BSSN (Badan Siber dan Sandi Negara)** — The national cybersecurity agency issues guidelines on critical information infrastructure (Infrastruktur Informasi Kritis/IIK). Factories in energy, transportation, and manufacturing sectors classified as IIK must:
- Store operational data on servers physically located in Indonesia or in certified Indonesian cloud regions
- Conduct security assessments before deploying AI systems that process operational data
- Maintain audit logs for all automated decisions affecting safety-critical processes

**Practical implications for maintenance AI:**
- Pump P-101 failure predictions and motor M-202 maintenance logs are operational data — they describe the physical state of production equipment
- Sending these logs to  or  routes data through US-based servers
- Safe options: (a) run Ollama locally on the factory network, (b) use vLLM on a cloud VM in an Indonesian data center (e.g., AWS ap-southeast-3 Jakarta, Azure Indonesia Central), (c) use a hybrid architecture where raw ticket text stays local and only anonymized embeddings are sent externally

**Recommendation for most Indonesian industrial facilities:** deploy a local Ollama instance for sensitive ticket processing, and use BaaS only for non-sensitive analytical workloads (aggregate statistics, trend reports).

### 3.3 When to Choose Each Option

**Choose BaaS (Anthropic/OpenAI) when:**
- You are prototyping or running an internal pilot
- Volume is under 1,000 tickets/day
- Data is not subject to residency restrictions
- You need the absolute best model quality for safety-critical classifications

**Choose Self-Hosted vLLM when:**
- You need high throughput (>10,000 requests/day)
- You want full control over model versions
- You have GPU infrastructure available or can justify a dedicated GPU VM
- You need sub-100ms latency for real-time applications

**Choose Ollama (Local) when:**
- Data must not leave the facility network
- You operate in environments with unreliable internet
- You need zero ongoing API cost
- Development and testing on a laptop

**Choose Hybrid when:**
- Different ticket categories have different sensitivity levels (route Safety tickets locally, Normal tickets to cloud)
- You want local fallback when cloud API is down
- You are migrating from cloud to local over time

---

## 4. vLLM Architecture

### 4.1 The Memory Problem vLLM Solves

Standard Transformers inference is wasteful. The KV (Key-Value) cache — the memory that stores attention state for each token — is allocated statically per sequence. If you reserve memory for a 2,048-token sequence but the actual response is 200 tokens, you waste 90% of the allocation. Worse, with naive batching, one long request blocks all short requests in the same batch.

**PagedAttention** solves this the same way an operating system manages virtual memory: it divides the KV cache into fixed-size blocks (pages) and allocates them dynamically. Pages are shared across requests when possible (e.g., when multiple queries share a common system prompt). Fragmentation is minimized.

```
Naive Approach:
  Sequence 1: [||||||||||||||||||||..........] (allocated 30, used 20, 10 wasted)
  Sequence 2: [||||..........................] (allocated 30, used 4,  26 wasted)

PagedAttention:
  Block pool: [B1][B2][B3][B4][B5][B6][B7][B8]...
  Sequence 1:  B1  B2  B3       (allocates only what it needs, grows dynamically)
  Sequence 2:  B4               (tiny allocation for short sequence)
  Shared PFX:  B5  (system prompt block shared by both sequences -- zero copy)
```

### 4.2 Continuous Batching

Traditional batching waits for a full batch to finish before starting the next. If one request generates 2,000 tokens while others finish at 50 tokens, the short requests must wait.

Continuous batching (also called “iteration-level scheduling”) replaces finished sequences with new ones at each decode step. The GPU is never idle waiting for a single slow request.

**Result:** 5–20x throughput improvement over naive Transformers serving, with lower average latency for short requests.

### 4.3 Throughput Benchmark

| Serving Method | Requests/sec (7B, A100) | p50 Latency | Notes |
|---|---|---|---|
| Naive Transformers (batch=1) | ~2 req/s | 800ms | Sequential, no batching |
| Naive Transformers (batch=8) | ~6 req/s | 1,200ms | Static batching, padding waste |
| vLLM (continuous batching) | ~30–40 req/s | 300ms | PagedAttention + cont. batching |
| vLLM (tensor parallel, 4xA100) | ~120 req/s | 150ms | Multi-GPU scaling |

### 4.4 Installation and Quickstart

```bash
# Install vLLM (requires CUDA 12.1+, Python 3.9+)
pip install vllm
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.1, max_tokens=256)

tickets = [
    "Pump P-101 seal leaking, vibration 12 mm/s",
    "Motor M-202 overheating, temperature 85 C",
    "PPE violation at mixing station, no safety goggles",
    "Conveyor C-05 belt misalignment and gearbox oil leak simultaneously",
]

prompts = [
    f"Classify into M (Mechanical), S (Safety), N (Normal), K (Komplex).
Ticket: {t}
Category:"
    for t in tickets
]

outputs = llm.generate(prompts, sampling_params)
for ticket, out in zip(tickets, outputs):
    print(f"[{out.outputs[0].text.strip()}] {ticket[:55]}...")
```

### 4.5 OpenAI-Compatible API Server

vLLM exposes an OpenAI-compatible REST API. Any code written for the OpenAI Python SDK works with vLLM by changing only .

```bash
# Start the API server
python -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen2.5-7B-Instruct     --host 0.0.0.0 --port 8000     --max-model-len 4096 --dtype bfloat16
```

```python
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "Classify maintenance tickets as M/S/N/K."},
        {"role": "user",   "content": "Motor M-202 temperature alarm, 91 C"}
    ],
    max_tokens=50, temperature=0.0
)
print(response.choices[0].message.content)
```

### 4.6 Tensor Parallelism for Multi-GPU

```bash
# Use 4 GPUs for a 70B model
python -m vllm.entrypoints.openai.api_server     --model meta-llama/Llama-3.1-70B-Instruct     --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 8192
```

**Rule of thumb:** tensor-parallel-size should equal the number of GPUs, and must divide evenly into the model’s number of attention heads.

---

## 5. Ollama for Local Inference

### 5.1 Installation

```bash
# Linux (one-line installer)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve   # runs on http://localhost:11434

# macOS / Windows: download installer from https://ollama.com/download
```

### 5.2 Pull and Run Models

```bash
ollama pull llama3.2         # Meta Llama 3.2 3B -- fast, good for classification
ollama pull llama3.2:1b      # 1B variant -- extremely fast on CPU
ollama pull qwen2.5:7b       # Qwen 2.5 7B -- excellent Arabic/Indonesian support
ollama pull qwen2.5:14b      # 14B -- better reasoning
ollama pull mistral          # Mistral 7B -- strong general performance
ollama pull phi3             # Microsoft Phi-3 Mini -- 3.8B, very efficient
ollama pull nomic-embed-text # Embedding model for RAG

ollama run llama3.2          # Start interactive chat
ollama list                  # List downloaded models
ollama rm phi3               # Remove a model
```

### 5.3 Python Integration

```python
# Option A: Direct HTTP API
import requests

def classify_ticket_ollama(ticket_text: str, model: str = "llama3.2") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Classify maintenance tickets:
"
                "M = Mechanical (pumps, bearings, seals, vibration)
"
                "S = Safety (PPE violations, hazards, incidents)
"
                "N = Normal/Electrical (routine checks, minor electrical)
"
                "K = Komplex (multi-system, requires engineering review)
"
                "Reply with only the single letter: M, S, N, or K.")},
            {"role": "user",   "content": f"Ticket: {ticket_text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5}
    }
    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

# Test
for ticket in [
    "P-101 mechanical seal failure, leakage rate 2 L/hr",
    "Operator Yusuf not wearing hard hat in Zone B",
    "Conveyor C-05 belt misalignment + gearbox oil leak simultaneously",
]:
    print(f"[{classify_ticket_ollama(ticket)}] {ticket[:55]}...")
```

```python
# Option B: LangChain ChatOllama -- works with all Day 4 LangGraph workflows
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="qwen2.5:7b", temperature=0.0, num_predict=50)
response = llm.invoke([
    SystemMessage(content="Classify maintenance tickets as M/S/N/K."),
    HumanMessage(content="Pump P-101 bearing temperature 92 C, vibration 8.5 mm/s")
])
print(response.content)
```

### 5.4 Model Library Reference

| Model | Size | VRAM (Q4) | Speed (CPU) | Strengths | Best For |
|---|---|---|---|---|---|
| llama3.2:1b | 1B | 1 GB | Very fast | General chat | Prototyping, low-end hardware |
| llama3.2 | 3B | 2 GB | Fast | Reasoning, instruction | Classification, quick answers |
| phi3 | 3.8B | 2.3 GB | Fast | Code, reasoning | Technical ticket parsing |
| mistral | 7B | 4 GB | Medium | General purpose | Balanced quality/speed |
| qwen2.5:7b | 7B | 4.5 GB | Medium | Multilingual, Arabic/ID | Arabic maintenance tickets |
| llama3.1:8b | 8B | 5 GB | Medium | Strong reasoning | Complex K-category tickets |
| qwen2.5:14b | 14B | 9 GB | Slow on CPU | Best multilingual | Production Arabic/Indonesian |

### 5.5 Use Cases for Local Inference

1. **Development and testing** — iterate on prompts without accumulating API costs
2. **Privacy-sensitive data** — pump performance logs, employee incident reports, proprietary equipment parameters
3. **Offline environments** — remote mining operations, offshore platforms, unreliable connectivity
4. **Edge deployment** — local server at the facility, no internet dependency for classification
5. **High-volume batch processing** — classify 10,000 historical tickets overnight at zero cost

---

## 6. Model Quantization

### 6.1 Why Quantization Matters

Every parameter in a neural network is stored as a floating-point number. The precision determines both memory usage and computational cost.

**Memory formula:**
```
GPU Memory (GB) = (Parameter Count x Bytes per Parameter) / 1,000,000,000
```

| Model Size | FP32 (4 bytes) | FP16 (2 bytes) | INT8 (1 byte) | INT4 (0.5 bytes) |
|---|---|---|---|---|
| 7B parameters | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B parameters | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 30B parameters | 120 GB | 60 GB | 30 GB | 15 GB |
| 70B parameters | 280 GB | 140 GB | 70 GB | 35 GB |

**Practical implications:**
- A 7B model in FP32 requires an A100 80GB GPU — expensive
- The same model in INT4 runs on a 4 GB consumer GPU (RTX 3060) or 6 GB VRAM laptop
- A 70B model in INT4 fits on a single A100 80GB — previously impossible without multi-GPU

### 6.2 GGUF Format and llama.cpp

**GGUF** (GPT-Generated Unified Format) is the file format used by the llama.cpp ecosystem, including Ollama. GGUF files pack model weights, tokenizer, and metadata into a single portable file.

Naming convention: 

| Quantization Level | Bits | Quality Retention | Use Case |
|---|---|---|---|
| FP16 | 16 | 100% baseline | Full quality, high VRAM |
| Q8_0 | 8 | ~99% | Near-lossless, 2x compression |
| Q6_K | 6 | ~98.5% | High quality, moderate compression |
| Q5_K_M | 5 | ~97.5% | Good quality, good compression |
| Q4_K_M | 4 | ~96% | Best quality/size ratio — recommended default |
| Q4_0 | 4 | ~95% | Slightly lower than Q4_K_M |
| Q3_K_M | 3 | ~93% | Noticeable degradation |
| Q2_K | 2 | ~85% | Significant quality loss, emergency only |

The “K” in Q4_K_M denotes k-quants — mixed-precision using higher precision for the most important layers. “M” = medium variant (balanced size/quality).

### 6.3 Choosing Quantization by Available Hardware

| Available VRAM | Recommended Quantization | Max Model Size | Notes |
|---|---|---|---|
| 4 GB | Q4_K_M | 7B | Consumer GPU (RTX 3060), good quality |
| 6 GB | Q4_K_M | 13B | RTX 3060 Ti / GTX 1060 6GB |
| 8 GB | Q5_K_M | 13B or Q4_K_M 7B | RTX 3070 / RTX 4060 |
| 12 GB | Q6_K | 13B or Q4_K_M 20B | RTX 3080 12GB |
| 16 GB | Q8_0 | 13B or Q4_K_M 30B | RTX 3090 / RTX 4080 |
| 24 GB | FP16 | 13B or Q4_K_M 70B | RTX 3090 24GB / RTX 4090 |
| 40 GB | FP16 | 30B | A100 40GB |
| 80 GB | FP16 | 70B | A100 80GB / H100 |

### 6.4 Quantization in Practice with bitsandbytes

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bfloat16, store in 4-bit
    bnb_4bit_use_double_quant=True,          # nested quantization for extra savings
    bnb_4bit_quant_type="nf4"               # NormalFloat4 -- best for LLM weights
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"  # automatically places layers across available GPUs/CPU
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
# Expected: ~3.8 GB for 7B in 4-bit
```

---

## 7. Production Engineering Patterns

### 7.1 Token Budget Tracking

Every API call has a cost. Without tracking, costs accumulate invisibly.

```python
import time
from dataclasses import dataclass, field
from typing import Optional

PRICING = {
    "claude-3-haiku-20240307":    {"input": 0.25,  "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-opus-20240229":     {"input": 15.00, "output": 75.00},
    "gpt-4o":                     {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":                {"input": 0.15,  "output": 0.60},
}

@dataclass
class RequestRecord:
    request_id: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    category: Optional[str] = None

class TokenBudgetTracker:
    def __init__(self, daily_budget_usd: float = 50.0, alert_threshold: float = 0.8):
        self.daily_budget_usd = daily_budget_usd
        self.alert_threshold = alert_threshold
        self.records: list = []

    def record(self, request_id, model, tokens_in, tokens_out, latency_ms, category=None):
        pricing = PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000
        rec = RequestRecord(request_id=request_id, model=model, tokens_in=tokens_in,
                            tokens_out=tokens_out, cost_usd=cost, latency_ms=latency_ms,
                            category=category)
        self.records.append(rec)
        daily = self.daily_spend()
        if daily >= self.daily_budget_usd * self.alert_threshold:
            print(f"[BUDGET ALERT] ${daily:.2f} = {daily/self.daily_budget_usd*100:.0f}% of daily budget")
        return rec

    def daily_spend(self) -> float:
        cutoff = time.time() - 86400
        return sum(r.cost_usd for r in self.records if r.timestamp > cutoff)

    def summary(self) -> dict:
        if not self.records:
            return {"total_requests": 0, "total_cost_usd": 0.0}
        return {
            "total_requests": len(self.records),
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 4),
            "avg_latency_ms": round(sum(r.latency_ms for r in self.records) / len(self.records), 1),
            "daily_spend_usd": round(self.daily_spend(), 4),
            "budget_remaining_usd": round(self.daily_budget_usd - self.daily_spend(), 4),
        }
```

### 7.2 Cost-Aware Routing

Not all tickets require the same model. Simple, clearly-worded tickets can be routed to a fast, cheap model. Only ambiguous or complex (K category) tickets need the full model with RAG.

```python
from anthropic import Anthropic
import time

client = Anthropic()
tracker = TokenBudgetTracker()

SIMPLE_PATTERNS = {
    "M": ["seal leak", "bearing", "vibration mm/s", "mechanical seal", "pump P-"],
    "S": ["PPE", "hard hat", "safety violation", "incident", "hazard", "tidak memakai"],
    "N": ["routine check", "pemeriksaan rutin", "no anomaly", "normal operation"],
}

def estimate_complexity(ticket: str) -> str:
    ticket_lower = ticket.lower()
    matched = [c for c, patterns in SIMPLE_PATTERNS.items()
               if any(p.lower() in ticket_lower for p in patterns)]
    return "simple" if len(matched) == 1 else "complex"

def classify_with_routing(ticket: str, request_id: str) -> dict:
    complexity = estimate_complexity(ticket)
    model = "claude-3-haiku-20240307" if complexity == "simple" else "claude-3-5-sonnet-20241022"
    max_tokens = 20 if complexity == "simple" else 150

    start = time.time()
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        system="Classify: M=Mechanical, S=Safety, N=Normal, K=Komplex. Letter + brief reason.",
        messages=[{"role": "user", "content": ticket}]
    )
    latency_ms = (time.time() - start) * 1000
    tracker.record(request_id, model,
                   response.usage.input_tokens, response.usage.output_tokens, latency_ms)
    return {"model_used": model, "complexity": complexity,
            "response": response.content[0].text, "cost_usd": tracker.records[-1].cost_usd}
```

### 7.3 Retry with Exponential Backoff

```python
import time, random, anthropic
from functools import wraps

def with_retry(max_attempts=3, base_delay=1.0, max_delay=60.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except anthropic.RateLimitError:
                    if attempt == max_attempts - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"Rate limited. Waiting {delay:.1f}s (attempt {attempt+1}/{max_attempts})")
                    time.sleep(delay)
                except anthropic.APIStatusError as e:
                    if e.status_code in (500, 502, 503, 529) and attempt < max_attempts - 1:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        time.sleep(delay)
                    else:
                        raise
                except anthropic.APIConnectionError:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(base_delay * (2 ** attempt))
        return wrapper
    return decorator

@with_retry(max_attempts=3)
def classify_ticket(ticket: str) -> str:
    response = client.messages.create(
        model="claude-3-haiku-20240307", max_tokens=10,
        messages=[{"role": "user", "content": f"Classify (M/S/N/K): {ticket}"}]
    )
    return response.content[0].text.strip()
```

### 7.4 Circuit Breaker

```python
import time
from enum import Enum
from typing import Optional

class CircuitState(Enum):
    CLOSED    = "closed"     # normal -- requests flow through
    OPEN      = "open"       # failing -- requests blocked immediately
    HALF_OPEN = "half_open"  # testing -- one request allowed to probe recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                print("[CircuitBreaker] HALF_OPEN -- testing recovery")
            else:
                raise RuntimeError("Circuit breaker OPEN -- using fallback")
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            return result
        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"[CircuitBreaker] OPEN after {self.failure_count} failures")
            raise

breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)

def safe_classify(ticket: str) -> str:
    try:
        return breaker.call(classify_ticket, ticket)
    except RuntimeError:
        # Rule-based fallback when API is unavailable
        if any(w in ticket.lower() for w in ["ppe", "safety", "hazard", "incident"]):
            return "S"
        if any(w in ticket.lower() for w in ["seal", "bearing", "pump", "vibration"]):
            return "M"
        return "N"
```

### 7.5 Semantic Caching

Identical or near-identical tickets waste API calls. If “Pump P-101 seal leaking” was classified yesterday, the same ticket today should return the cached answer.

```python
import hashlib, time
from typing import Optional

class SemanticCache:
    def __init__(self, ttl_seconds=86400):  # 24-hour TTL
        self._cache: dict = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()

    def get(self, text: str) -> Optional[str]:
        entry = self._cache.get(self._key(text))
        if entry and (time.time() - entry["cached_at"]) < self.ttl:
            self.hits += 1
            return entry["result"]
        self.misses += 1
        return None

    def set(self, text: str, result: str):
        self._cache[self._key(text)] = {"result": result, "cached_at": time.time()}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

cache = SemanticCache()

def classify_with_cache(ticket: str) -> str:
    cached = cache.get(ticket)
    if cached:
        return cached
    result = safe_classify(ticket)
    cache.set(ticket, result)
    return result

# On a realistic industrial dataset, 30-60% of tickets are near-duplicates
# (e.g., "P-101 seal check" submitted by different shifts).
# Caching reduces API calls and cost proportionally.
```

---

## 8. System Monitoring & Observability

### 8.1 Key Metrics to Track

| Metric | Definition | Target | Alert Threshold |
|---|---|---|---|
| p50 latency | Median response time | < 500ms | — |
| p95 latency | 95th percentile response time | < 2,000ms | > 3,000ms = warning |
| p99 latency | 99th percentile response time | < 5,000ms | > 10,000ms = critical |
| Error rate | % of requests returning errors | < 1% | > 5% = critical |
| Token usage / request | Average input + output tokens | Baseline ± 20% | +50% spike = investigate |
| Cost / request | Average USD per classification | Establish baseline | 2x baseline = alert |
| Daily spend | Rolling 24h API cost | Within budget | 80% of budget = warning |
| Cache hit rate | % of requests served from cache | > 30% for mature system | < 10% = review caching |

### 8.2 LangSmith Integration

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]     = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"]     = "industrial-maintenance-ai"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

# All subsequent LangChain/LangGraph calls are automatically traced.
# No code changes needed.

from langsmith import traceable

@traceable(name="classify_maintenance_ticket", tags=["production", "classification"])
def classify_ticket_traced(ticket: str, facility_id: str) -> dict:
    return classify_with_routing(ticket, request_id=f"{facility_id}-{time.time()}")
```

### 8.3 Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

REQUEST_COUNT   = Counter("llm_requests_total", "Total LLM API requests",
                          ["model", "category", "status"])
REQUEST_LATENCY = Histogram("llm_request_latency_seconds", "LLM request latency", ["model"],
                            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
DAILY_COST      = Gauge("llm_daily_cost_usd",  "Cumulative LLM API cost last 24h")
CACHE_HIT_RATE  = Gauge("llm_cache_hit_rate",  "Semantic cache hit rate (0.0-1.0)")

def classify_with_metrics(ticket: str) -> str:
    start = time.time()
    model = "claude-3-haiku-20240307"
    try:
        result = classify_with_cache(ticket)
        REQUEST_COUNT.labels(model=model, category=result, status="success").inc()
        REQUEST_LATENCY.labels(model=model).observe(time.time() - start)
        DAILY_COST.set(tracker.daily_spend())
        CACHE_HIT_RATE.set(cache.hit_rate)
        return result
    except Exception:
        REQUEST_COUNT.labels(model=model, category="error", status="error").inc()
        raise

start_http_server(9090)  # Prometheus scrapes :9090/metrics; Grafana reads from Prometheus
```

### 8.4 Alerting Rules (Prometheus Alertmanager)

```yaml
groups:
  - name: llm_system_alerts
    rules:
      - alert: HighLatencyP95
        expr: histogram_quantile(0.95, llm_request_latency_seconds_bucket) > 3.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 LLM latency exceeds 3s"
          description: "Ticket classification p95 is {{ $value }}s."

      - alert: HighErrorRate
        expr: >
          rate(llm_requests_total{status="error"}[5m]) /
          rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM error rate above 5%"

      - alert: DailyBudgetWarning
        expr: llm_daily_cost_usd > 40.0
        labels:
          severity: warning
        annotations:
          summary: "Daily LLM spend approaching $50 budget"
```

### 8.5 SLA Definition

```
SERVICE LEVEL AGREEMENT -- Maintenance Ticket Classification System
====================================================================
Service:          Automated ticket classification and response drafting
Covered Hours:    24/7

Performance SLAs:
  Availability:   99.5% uptime (~3.6 hours downtime/month)
  p50 Latency:    < 500ms per ticket classification
  p95 Latency:    < 2,000ms per ticket classification
  p99 Latency:    < 5,000ms per ticket classification

Accuracy SLAs:
  Classification accuracy:  > 90% vs. human expert labels
  Safety ticket recall:     > 99% (S-category must never be missed)
  Human review rate:        < 10% of tickets require manual review

Cost SLAs:
  Cost per ticket:          < $0.01 average
  Daily spend cap:          $50 USD

Incident Response:
  P1 (Safety misclassification):  Immediate page, 15-minute response
  P2 (Service unavailable):        30-minute response
  P3 (High latency):               2-hour response
```

---

## 9. Fine-Tuning with Unsloth

### 9.1 When to Fine-Tune

| Technique | When to Use | Cost | Data Required | Maintenance |
|---|---|---|---|---|
| **Prompt Engineering** | Model knows the domain; formatting control needed | Very low | None | Easy |
| **RAG** | Factual grounding; knowledge changes frequently | Low | Documents only | Medium |
| **Few-Shot in Prompt** | 5–20 labeled examples; quick improvement needed | Low | 5–20 examples | Easy |
| **SFT Fine-Tuning** | Specific style/format; privacy requirement | Medium | 500–10,000 examples | Hard |
| **Continued Pretraining** | Domain adaptation; model lacks your vocabulary | High | 100k+ tokens raw text | Hard |
| **DPO / RLHF** | Outputs good but need preference alignment | High | 1,000+ preference pairs | Very hard |

**Fine-tune when:**
- Your industrial terminology is unique (equipment codes: P-101, M-202, C-05)
- Data cannot be sent to external APIs (BSSN compliance)
- You need sub-100ms latency on edge hardware
- RAG consistently retrieves irrelevant documents and prompt engineering has not fixed it
- You want consistent output in Bahasa Indonesia or Arabic without English contamination

**Do not fine-tune when:**
- You have fewer than 200 labeled examples
- Your team cannot maintain a training pipeline
- Prompt engineering already achieves > 90% accuracy on your eval set

### 9.2 Unsloth Overview

[Unsloth](https://github.com/unslothai/unsloth) is an open-source library making fine-tuning significantly faster and more memory-efficient than standard LoRA via HuggingFace PEFT.

| Metric | Standard LoRA (PEFT) | Unsloth |
|---|---|---|
| Training speed | Baseline | 2x faster |
| VRAM usage | Baseline | 70% less |
| Accuracy | Baseline | Identical (no approximations) |
| Supported models | All HF models | Llama 3.x, Qwen 2.5, Mistral, Phi-3, Gemma 2 |
| Cost on Colab T4 | ~2h for 7B SFT | ~1h for 7B SFT |
| License | Apache 2.0 | Apache 2.0 |

Unsloth achieves its gains through hand-written CUDA kernels for the backward pass, custom memory management, and dynamic quantization. The output is a standard  adapter file compatible with any PEFT-aware code.

### 9.3 Hardware Requirements

```
Minimum hardware for fine-tuning:
  7B model (4-bit):   6 GB VRAM    (RTX 3060, T4 Colab)
  13B model (4-bit):  12 GB VRAM   (RTX 3080, A10G)
  70B model (4-bit):  40 GB VRAM   (A100 40GB, 2x A40)

Recommended:
  Development:         Google Colab T4 (free, 15 GB VRAM)
  Production training: NVIDIA A10G (24 GB VRAM) -- ~$1/hour on AWS
  Large-scale:         NVIDIA A100 80GB -- ~$3/hour on AWS

For the 40-ticket exercise: Colab T4 is sufficient. Expected: 5-15 min for 3 epochs.
```

### 9.4 Supervised Fine-Tuning (SFT)

SFT teaches the model to follow instructions by training on (input, output) pairs.

**Dataset format (Alpaca):**

```json
{
  "instruction": "Classify this industrial maintenance ticket into M, S, N, or K.",
  "input": "Pump P-101 mechanical seal leaking at 2 L/hr. Scheduled for next planned shutdown.",
  "output": "M

Mechanical ticket. Worn seal on P-101. Severity: Medium. Action: Schedule replacement. Monitor leak rate hourly."
}
```

**Complete SFT setup with Unsloth:**

```python
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install trl transformers datasets accelerate peft bitsandbytes

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import torch

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048, dtype=None, load_in_4bit=True,
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)
print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
# Typical output: 41,943,040 (41M LoRA out of 7.6B total)

# 3. Prepare dataset
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

INSTRUCTION = (
    "You are an industrial maintenance AI. "
    "Classify the maintenance ticket. Category codes: M=Mechanical, S=Safety, N=Normal, K=Komplex."
)

training_examples = [
    {
        "instruction": INSTRUCTION,
        "input": "Pump P-101 mechanical seal leaking, vibration 12 mm/s",
        "output": "Category: M

Mechanical fault. Worn seal, elevated vibration (12 vs 7.1 mm/s threshold). Action: Immediate inspection, schedule seal + bearing replacement. Priority: High."
    },
    {
        "instruction": INSTRUCTION,
        "input": "Operator di area mixing tidak memakai kacamata pelindung",
        "output": "Category: S

Safety violation. Operator without protective eyewear in mixing area. Immediate corrective action per SOP-HSE-004. Document in safety log."
    },
    {
        "instruction": INSTRUCTION,
        "input": "Motor M-202 overheating (91 C), conveyor belt C-05 misaligned simultaneously",
        "output": "Category: K

Komplex multi-system fault. M-202 thermal threshold exceeded (max 85 C). Simultaneous C-05 misalignment may share root cause. Engineering review required. Do not restart."
    },
    # ... expand to all 40 tickets for production training ...
]

def format_examples(examples):
    texts = [
        alpaca_prompt.format(instr, inp, out) + tokenizer.eos_token
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return {"text": texts}

dataset = Dataset.from_list(training_examples).map(format_examples, batched=True)

# 4. Configure and train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset, dataset_text_field="text",
    max_seq_length=2048, dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, warmup_steps=5, learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1, output_dir="./maintenance-qwen-lora",
        optim="adamw_8bit", lr_scheduler_type="cosine", seed=42,
    ),
)
stats = trainer.train()
print(f"Training time: {stats.metrics['train_runtime']:.1f}s")
```

### 9.5 Continued Pretraining

Use continued pretraining to adapt a general model to your facility’s vocabulary: equipment codes, maintenance terminology, Bahasa Indonesia/Arabic phrases.

```python
from transformers import DataCollatorForLanguageModeling

# Raw maintenance logs -- no labels required, just raw text
raw_corpus_sample = """
Laporan Pemeliharaan Harian -- Pabrik Semen Tuban -- 15 Januari 2025

Unit P-101 (Pompa Slurry):
  Tekanan discharge: 4.2 bar (nominal 4.0-4.5 bar) -- Normal
  Temperatur bearing: 68 C (batas 85 C) -- Normal
  Getaran: 5.2 mm/s (batas 7.1 mm/s) -- Normal

Unit M-202 (Motor Conveyor Belt C-05):
  Arus motor: 42 A (nominal 40 A, batas 48 A) -- Normal
  Temperatur winding: 72 C (batas 90 C) -- Normal

Catatan shift malam:
- Oil level gearbox GB-07: 85% (OK)
- Kebocoran kecil flange pipa feed P-103 (~0.5 L/jam) -- pasang klem sementara
- Panel MCC-02: semua breaker normal
"""

# DataCollatorForLanguageModeling handles causal LM (next-token prediction)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Quality metric: perplexity on a held-out maintenance text sample.
# Target: 20-40% reduction in perplexity after continued pretraining.
```

### 9.6 RLHF with Direct Preference Optimization (DPO)

**What is RLHF?** Reinforcement Learning from Human Feedback trains a model to produce outputs that human raters prefer. The original technique required a separate reward model, which was complex and unstable.

**DPO** is a simpler alternative. Given a (prompt, chosen, rejected) dataset, DPO directly increases the likelihood of chosen responses relative to rejected ones. No reward model needed.

**Preference dataset format:**

```json
{
  "prompt": "Classify and respond to: Motor M-202 temperature alarm 91 C",
  "chosen": "Category: M -- High Priority

M-202 has exceeded thermal threshold (91C vs 85C limit).
Actions: (1) Reduce motor load. (2) Check cooling fan. (3) Inspect air filter. (4) Log in CMMS. (5) Notify shift supervisor.
Do NOT restart without engineering sign-off.",
  "rejected": "The motor is hot. Check the cooling. Temperature is too high."
}
```

```python
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# Start from SFT-trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./maintenance-qwen-lora",
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=8, lora_alpha=8, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0, bias="none", use_gradient_checkpointing="unsloth",
)

preference_data = [
    {
        "prompt": "Classify: Pump P-101 seal leaking 2 L/hr",
        "chosen": "Category: M -- Medium Priority

Mechanical seal on P-101 failing. Leak rate 2 L/hr is below emergency threshold (5 L/hr) but requires scheduled replacement. Actions: (1) Monitor leak rate every 2h. (2) Prepare replacement seal. (3) Schedule downtime with production.",
        "rejected": "M. The pump is leaking.",
    },
    # ... 50-200 preference pairs ...
]

dpo_trainer = DPOTrainer(
    model=model, ref_model=None,
    args=DPOConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=5e-5, beta=0.1,
        output_dir="./maintenance-qwen-dpo",
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
    ),
    train_dataset=Dataset.from_list(preference_data),
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

### 9.7 Saving and Deployment

```python
# Option A: LoRA adapter only (~100-400 MB)
model.save_pretrained("./maintenance-qwen-final-lora")
tokenizer.save_pretrained("./maintenance-qwen-final-lora")

# Option B: Merge LoRA into base model (larger, simpler to serve)
model.save_pretrained_merged(
    "./maintenance-qwen-merged", tokenizer, save_method="merged_16bit"
)

# Option C: Export as GGUF for Ollama deployment
model.save_pretrained_gguf(
    "./maintenance-qwen-gguf", tokenizer, quantization_method="q4_k_m"
)
# Output: ./maintenance-qwen-gguf/unsloth.Q4_K_M.gguf

# Option D: Push to HuggingFace Hub (private)
model.push_to_hub_merged(
    "your-org/maintenance-qwen-industrial", tokenizer,
    save_method="merged_16bit", token="hf-token", private=True
)

# Load GGUF with Ollama:
# Create Modelfile:
#   FROM ./maintenance-qwen-gguf/unsloth.Q4_K_M.gguf
#   SYSTEM "You are an industrial maintenance AI. Classify tickets as M/S/N/K."
#   PARAMETER temperature 0.1
#   PARAMETER num_predict 300
#
# Then:
#   ollama create maintenance-assistant -f Modelfile
#   ollama run maintenance-assistant
```

---

## 10. End-to-End Architecture

### 10.1 System Overview

```
==========================================================================
        INDUSTRIAL MAINTENANCE AI -- END-TO-END ARCHITECTURE
==========================================================================

  [Raw Ticket Input]
  (CMMS, email, WhatsApp, manual entry)
          |
          v
  +-------------------------------+
  |    Preprocessing Pipeline     |
  |  * Text normalization         |
  |  * Language detection (ID/AR) |
  |  * PII detection / masking    |
  |  * Token count estimation     |
  +---------------+---------------+
                  |
                  v
  +-------------------------------+
  |    Semantic Cache Check       |<-- Cache TTL: 24h | Hit rate target: >30%
  |  Hash -> lookup -> hit/miss   |
  +----------+----------+---------+
          HIT|          |MISS
             v          |
    [Cached Result]     |
                        v
          +---------------------------+
          |    Cost-Aware Router      |
          |  Simple -> Haiku/Mini     |
          |  Complex -> Sonnet/GPT-4  |
          +----------+--------+-------+
                     |        |
           SIMPLE    |        |  COMPLEX
                     v        v
        +-------------+  +----------------------------+
        |  Fast Model |  |  Full Model + RAG          |
        | (BaaS cheap)|  |  LangGraph Agent           |
        |  Retry +    |  |  * Query FAISS vector store|
        |  Circuit    |  |  * Retrieve SOP docs       |
        |  Breaker    |  |  * Multi-step reasoning    |
        +------+------+  +--------------+-------------+
               |                        |
               +----------+-------------+
                          v
             +------------------------+
             |    Response Draft      |
             |  * Category: M/S/N/K  |
             |  * Priority: H/M/L    |
             |  * Recommended action |
             |  * Confidence score   |
             +------------+-----------+
                          |
                          v
             +------------------------+
             |  Token Budget Tracker  |
             |  Record tokens in/out  |
             |  Compute cost_usd      |
             |  Check daily budget    |
             +------------+-----------+
                          |
                          v
             +------------------------+
             |   Confidence Gate      |
             |  Score > 0.8 -> auto  |
             |  Score < 0.8 -> queue |
             +--------+-------+-------+
              AUTO    |       | HUMAN REVIEW QUEUE
                      v       v
          +-----------+  +------------------+
          |Write CMMS |  | Human Review UI  |
          |Auto-assign|  | Expert validates |
          |Send alerts|  | Label -> dataset |
          +-----------+  +--------+---------+
                                  | Labeled data
                                  v
  +-----------------------------------------------+
  |          Monitoring & Observability            |
  |  LangSmith   -> Trace every LLM call          |
  |  Prometheus  -> Export latency / cost          |
  |  Grafana     -> Dashboard + alerting           |
  |  Drift detect -> Weekly accuracy check         |
  +----------------------+------------------------+
                         | Low accuracy / new patterns
                         v
  +-----------------------------------------------+
  |        Periodic Fine-Tuning Loop              |
  |  Trigger: monthly OR accuracy < 88%           |
  |  1. Collect new labeled tickets               |
  |  2. Merge with existing dataset              |
  |  3. SFT with Unsloth (Colab T4)              |
  |  4. Eval on holdout set                      |
  |  5. If accuracy improves -> deploy            |
  |  6. Save GGUF to Ollama / HuggingFace Hub    |
  +-----------------------------------------------+
```

### 10.2 Deployment Topology Options

```
OPTION A -- Full Cloud (BaaS)
------------------------------
[Factory CMMS] --HTTPS--> [Python Service on VPS] --API--> [Anthropic/OpenAI]
                                   |
                                   +-> [LangSmith (tracing)]
                                   +-> [Prometheus + Grafana (metrics)]

OPTION B -- Hybrid (Local + Cloud for complex only)
-----------------------------------------------------
[Factory CMMS] -> [Local Server (Ollama)] -> [Simple/Normal tickets]
                          |
                          +-> [Cloud API] -> [K-category tickets only]

OPTION C -- Full Local (BSSN compliant, zero external API calls)
-----------------------------------------------------------------
[Factory CMMS] -> [Local Server] -> [Ollama: maintenance-assistant GGUF]
                        |
                        +-> [Local Grafana + Prometheus]
                        +-> [Local FAISS vector store]
                        +-> [No external network required]
```

---

## 11. Essential Libraries & Tools Reference

| Library | Purpose | Install | Key Classes / Functions |
|---|---|---|---|
|  | High-throughput LLM inference server |  | , ,  |
|  | Local Ollama Python client |  | , ,  |
|  | Fast fine-tuning, 70% less VRAM | See GitHub readme | , ,  |
|  | RL and SFT training utilities |  | , , ,  |
|  | Parameter-efficient fine-tuning |  | , ,  |
|  | Core model and tokenizer loading |  | , , ,  |
|  | 4-bit and 8-bit quantization |  | , ,  |
|  | Multi-GPU and mixed precision |  | ,  |
|  | HuggingFace dataset loading |  | , ,  |
|  | Anthropic Claude API client |  | , ,  |
|  | LLM chains and prompting |  | , ,  |
|  | LangChain Ollama integration |  | ,  |
|  | LLM workflow state graphs |  | , , ,  |
|  | Vector similarity search |  | , ,  |
|  | Embedding models |  | , ,  |
|  | Metrics export |  | , , ,  |
|  | LLM tracing and evaluation |  | , ,  |
|  | System resource monitoring |  | , ,  |
|  | HTTP client for Ollama API | stdlib | , ,  |

---

## 12. Glossary

**Adapter** — A small set of trainable parameters added on top of a frozen pretrained model. LoRA adapters are the most common type; they insert rank-decomposition matrices into attention layers.

**Alpaca Format** — A fine-tuning dataset format with three fields: `instruction` (task description), `input` (context), and `output` (expected response). Originated from Stanford Alpaca (2023).

**Batch Inference** — Processing multiple inputs simultaneously, improving GPU utilization and throughput compared to sequential one-at-a-time processing.

**BaaS (Backend as a Service)** — Cloud LLM APIs where the model is hosted by a third party (Anthropic, OpenAI). You pay per token; no infrastructure management required.

**bfloat16 (BF16)** — A 16-bit floating-point format with the same exponent range as FP32 but reduced mantissa precision. Preferred for training on modern GPUs (Ampere+) because it handles gradient magnitudes better than FP16.

**bitsandbytes** — A Python library enabling 8-bit and 4-bit quantized model loading and training, allowing large models on consumer GPUs with minimal code changes.

**BSSN** — Badan Siber dan Sandi Negara. Indonesia’s National Cyber and Crypto Agency. Issues data residency and security guidelines for critical infrastructure operators including manufacturing facilities.

**Circuit Breaker** — A reliability pattern that stops sending requests to a failing service after a threshold of consecutive errors and returns a fallback response instead.

**Continuous Batching** — vLLM’s iteration-level scheduling strategy that replaces completed sequences with new requests at each decode step, eliminating head-of-line blocking.

**CUDA** — Compute Unified Device Architecture. NVIDIA’s parallel computing platform required for GPU-accelerated LLM inference and fine-tuning.

**DataCollator** — A HuggingFace utility batching and padding variable-length training examples. `DataCollatorForLanguageModeling` handles causal LM; `DataCollatorForCompletionOnlyLM` masks the instruction prefix so only response tokens contribute to loss.

**DPO (Direct Preference Optimization)** — A fine-tuning method training a model to prefer “chosen” over “rejected” responses using a (prompt, chosen, rejected) dataset. Simpler and more stable than full RLHF; no reward model required.

**Exponential Backoff** — A retry strategy that doubles wait time after each failure, with optional random jitter to prevent synchronized retries (thundering herd problem).

**Fine-Tuning** — Continuing training of a pretrained model on a task-specific dataset to improve task performance at far lower cost than training from scratch.

**FP16 (Float16)** — 16-bit floating-point. Half the memory of FP32. Used for inference and mixed-precision training. Can overflow with large gradient values.

**FP32 (Float32)** — 32-bit floating-point. Standard precision for neural network training. 4 bytes per parameter.

**GGUF** — GPT-Generated Unified Format. A single-file format for quantized LLM weights used by llama.cpp and Ollama. Packs weights, tokenizer, and metadata into one portable file.

**GPU** — Graphics Processing Unit. Processor with thousands of small cores optimized for parallel matrix operations. Essential for LLM inference and training.

**Gradient Accumulation** — Splitting a large effective batch across multiple smaller passes before updating weights. Enables large batch sizes on limited VRAM.

**HuggingFace Hub** — A public repository with tens of thousands of pretrained models, datasets, and adapters. Downloaded with `AutoModel.from_pretrained("org/model-name")`.

**INT4** — 4-bit integer quantization. Reduces memory by 8x vs FP32. Best quality-to-size ratio for consumer hardware using k-quants (Q4_K_M).

**INT8** — 8-bit integer quantization. Reduces memory by 4x vs FP32 with near-lossless quality (~99% retention).

**KV Cache** — Key-Value cache storing intermediate attention computations for processed tokens, enabling efficient autoregressive decoding. Memory grows linearly with sequence length and batch size.

**k-quants** — A mixed-precision quantization scheme in llama.cpp applying higher-bit precision to sensitive weight matrices. Denoted by “K” in GGUF filenames (e.g., Q4_K_M).

**LangSmith** — Cloud observability platform for LangChain and LangGraph applications. Records every LLM call with inputs, outputs, latency, and token counts.

**llama.cpp** — Open-source C/C++ inference engine for LLMs. Runs on CPU and consumer GPUs. The runtime underlying Ollama. Supports GGUF natively.

**LoRA (Low-Rank Adaptation)** — A PEFT method inserting trainable low-rank decomposition matrices into attention layers of a frozen base model.

**LR Scheduler** — Controls learning rate changes during training. Cosine decay (high initially, decreasing) is the standard choice for SFT fine-tuning.

**Mock Mode** — Development technique replacing real API calls with hardcoded simulated responses. Enables pipeline testing without costs or network.

**Model Quantization** — Reducing numerical precision of model weights (e.g., FP32 to INT4) to decrease memory usage and increase inference speed at a small quality cost.

**NF4 (NormalFloat4)** — A 4-bit data type optimized for normally distributed values (typical neural network weight distribution). Used by QLoRA and bitsandbytes.

**Ollama** — Open-source tool for running local LLMs. Manages downloads, serves an OpenAI-compatible REST API at localhost:11434, handles quantization via GGUF automatically.

**PagedAttention** — vLLM’s KV cache memory management innovation. Borrows OS virtual memory paging to allocate cache blocks dynamically and share them across requests with identical prefixes.

**PEFT (Parameter-Efficient Fine-Tuning)** — Methods updating only a small fraction of model parameters. LoRA and QLoRA are the most widely used PEFT methods.

**Perplexity** — A language model quality metric. Lower = better prediction of held-out text. Used to evaluate continued pretraining on domain data.

**Prometheus** — Open-source monitoring system and time-series database. Applications expose metrics via HTTP; Prometheus scrapes and stores them. Integrates with Grafana for dashboards.

**QLoRA (Quantized LoRA)** — Fine-tuning a 4-bit quantized base model with FP16 LoRA adapters. Enables 7B fine-tuning on 6 GB VRAM with minimal quality loss.

**RAG (Retrieval-Augmented Generation)** — Enhancing LLM responses by retrieving relevant documents from a vector store. Reduces hallucination and grounds responses in actual maintenance documentation.

**Rank (LoRA)** — Dimensionality of LoRA’s low-rank decomposition matrices. Higher rank = more capacity, but higher VRAM. Typical: 4, 8, 16, 32.

**RLHF (Reinforcement Learning from Human Feedback)** — Training paradigm where human preference comparisons guide policy optimization. Used to align LLMs with human quality standards.

**Reward Model** — In RLHF, a separately trained network predicting a quality score for (prompt, response) pairs. DPO bypasses the need for an explicit reward model.

**Semantic Caching** — Caching LLM responses by semantic similarity. If a new query is near-identical to a cached one, return the cached response without a new API call.

**SFT (Supervised Fine-Tuning)** — Fine-tuning on (input, desired_output) pairs using cross-entropy loss. The standard first step before DPO or RLHF.

**Tensor Parallelism** — Distributing a model’s weight matrices across multiple GPUs for 70B+ models that exceed single-GPU VRAM.

**TRL (Transformer Reinforcement Learning)** — HuggingFace library providing `SFTTrainer`, `DPOTrainer`, `PPOTrainer`, and other alignment trainers. Works with Unsloth.

**Token Budget** — A defined limit on input/output tokens per request or time period. Controls API costs and prevents runaway usage.

**Tokenizer** — Converts raw text into integer token IDs (and vice versa). Each model has its own tokenizer and vocabulary.

**Unsloth** — Open-source Python library accelerating LoRA/QLoRA fine-tuning by 2x and reducing VRAM usage 70% via hand-written CUDA kernels. Produces standard HuggingFace-compatible adapters.

**vLLM** — High-throughput LLM inference and serving engine from UC Berkeley. Uses PagedAttention and continuous batching to achieve 5–20x throughput vs naive Transformers serving.

**Warmup Steps** — Initial training phase where learning rate is gradually increased from near-zero to target value, preventing large gradient updates in early batches.

---

*End of Day 5 Handout — Industrial AI & LLM Training Program*

*For questions after the session: refer to the program repository or contact your instructor.*
