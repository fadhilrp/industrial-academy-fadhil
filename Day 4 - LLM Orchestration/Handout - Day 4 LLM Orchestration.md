---
tags: [handout, day4, langchain, langgraph, orchestration, reference]
session: "Day 4 — LLM Orchestration with LangChain & LangGraph"
program: "Industrial AI & LLM Training Program"
export: "PDF-ready — export via Obsidian or Pandoc"
---

# Day 4 Handout — LLM Orchestration with LangChain & LangGraph
*Industrial AI & LLM Training Program*

---

## 1. Why Orchestration?

### From Single Call to Workflow

In Day 2 and Day 3, we made individual LLM API calls — one prompt, one response. For simple
classification or one-shot RAG queries, this works well. But real support systems require:

- **Multiple steps**: classify → look up equipment → retrieve SOP → draft response → escalate
- **Conditional logic**: if priority is Critical, skip drafting and escalate immediately
- **Tool use**: call external APIs (CMMS, historian, ERP) and feed results to the LLM
- **Memory**: maintain context across a multi-turn technician conversation
- **Retry and fallback**: if SOP not found, try a refined query; if LLM fails, use a rule
- **Human oversight**: pause before creating a Work Order and require supervisor approval

**LLM Orchestration** is the engineering practice of connecting LLMs, tools, memory, and
control flow into coherent, reliable workflows. LangChain and LangGraph are the dominant
open-source frameworks for this.

### Chains vs. Agents vs. Graphs

| Concept | Structure | When to use |
|---------|-----------|-------------|
| **Chain (LCEL)** | Linear: A → B → C | Fixed, predictable pipelines |
| **Agent (ReAct)** | Dynamic loop: think → act → observe | When the LLM must decide which tools to use |
| **Graph (LangGraph)** | DAG with conditional edges | Complex workflows with branching, loops, HITL |

**Industrial rule of thumb:**
- Use a **chain** when you know every step at design time (e.g., batch ticket classification)
- Use an **agent** when the LLM needs to explore (e.g., open-ended equipment diagnosis)
- Use a **graph** when you need branching, retry, or human-in-the-loop (e.g., WO approval)

---

## 2. LangChain Architecture Overview

### The LCEL Pipe Operator

**LangChain Expression Language (LCEL)** uses `|` to chain components:

```python
chain = prompt | llm | output_parser
result = chain.invoke({'ticket': ticket_text})
```

Every component is a **Runnable** — it implements:

| Method | Description | Use case |
|--------|-------------|----------|
| `.invoke(input)` | Single synchronous call | Standard use |
| `.batch([inputs])` | Parallel calls | Multiple tickets at once |
| `.stream(input)` | Token-by-token streaming | Real-time UI |
| `.ainvoke(input)` | Async single call | Async web servers |
| `.astream(input)` | Async streaming | Async UI |

### Core LCEL Components

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic

# Prompt template with variables
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Kamu adalah support agent industrial.'),
    ('human', 'Klasifikasikan tiket: {ticket_text}')
])

# LLM (Anthropic Claude, consistent with Day 2 & Day 3)
llm = ChatAnthropic(model='claude-haiku-4-5-20251001', temperature=0.0)

# Output parser
parser = StrOutputParser()

# Chain: prompt → llm → parse
chain = prompt | llm | parser
```

### Prompt Template Patterns

| Pattern | Code | Best for |
|---------|------|----------|
| Simple string | `PromptTemplate.from_template("Classify: {text}")` | Single-variable |
| Chat messages | `ChatPromptTemplate.from_messages([('system', ...), ('human', ...)])` | Multi-role prompts |
| Few-shot | `FewShotChatMessagePromptTemplate` | Examples in prompt |
| Partial variables | `prompt.partial(system=SYSTEM_PROMPT)` | Shared system prompt |

### Chain Composition Patterns

```python
# Sequential chain
chain = classify_prompt | llm | parser

# Parallel chains with RunnableParallel
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel(
    category=(classify_prompt | llm | parser),
    summary=(summary_prompt | llm | parser)
)

# Conditional chain with RunnableBranch
from langchain_core.runnables import RunnableBranch
branch = RunnableBranch(
    (lambda x: x['priority'] == 'Kritis', escalate_chain),
    standard_chain  # default
)

# With fallback
chain_with_fallback = primary_chain.with_fallbacks([backup_chain])
```

---

## 3. Tools Deep Dive

### Defining Tools with @tool

```python
from langchain_core.tools import tool

@tool
def ticket_lookup(ticket_id: str) -> str:
    """Look up a maintenance ticket by ID (e.g. M01, K05).
    Returns ticket text, category, priority, and linked equipment."""
    # Implementation: query CMMS database
    return ticket_data

@tool
def equipment_status(equipment_id: str) -> str:
    """Get real-time equipment status (e.g. P-101, K-202).
    Returns temperature, pressure, vibration, next scheduled maintenance."""
    # Implementation: query PI Historian or SCADA
    return status_data

@tool
def sop_search(query: str) -> str:
    """Search the industrial knowledge base for SOPs and procedures.
    Uses semantic search (Day 3 RAG pipeline). Returns relevant procedure text."""
    # Implementation: rag_query() from Day 3
    return sop_content
```

### How Tool Schemas Work

LangChain generates a JSON schema from the function signature and docstring:

```json
{
  "name": "ticket_lookup",
  "description": "Look up a maintenance ticket by ID (e.g. M01, K05)...",
  "parameters": {
    "type": "object",
    "properties": {
      "ticket_id": {"type": "string", "description": "..."}
    },
    "required": ["ticket_id"]
  }
}
```

The LLM receives this schema and decides when and how to call the tool. The model's
`tool_calls` response is parsed by LangChain, which executes the Python function
and returns the observation.

### StructuredTool for Complex Parameters

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class WorkOrderInput(BaseModel):
    equipment_id: str = Field(description="Equipment ID, e.g. P-101")
    description: str = Field(description="Brief description of required maintenance")
    priority: str = Field(description="Priority: Kritis/Tinggi/Sedang/Rendah")

def create_work_order(equipment_id: str, description: str, priority: str) -> str:
    """Create a WO in SAP PM."""
    ...

wo_tool = StructuredTool.from_function(
    func=create_work_order,
    name='create_work_order',
    description='Create a maintenance Work Order in the CMMS/SAP PM system.',
    args_schema=WorkOrderInput
)
```

### Tool Routing Patterns

| Pattern | Description | Use case |
|---------|-------------|----------|
| **Agent decides** | LLM chooses tools dynamically | Open-ended queries |
| **Always run** | Fixed tool call before LLM | Mandatory lookups |
| **Conditional tool** | Route to tool based on state field | Category-specific SOPs |
| **Tool chain** | Tools in fixed sequence | Structured diagnostic workflow |

---

## 4. Agent Patterns

### The ReAct Loop

**ReAct** (Reason + Act) is the dominant agentic pattern for tool use:

```
User: "Analisis tiket M01 tentang pompa P-101"

Thought: I need to look up the ticket details first.
Action: ticket_lookup
Action Input: {"ticket_id": "M01"}
Observation: Ticket M01: Pompa P-101 bunyi berisik... Equipment: P-101

Thought: The ticket mentions P-101. Let me check its current status.
Action: equipment_status
Action Input: {"equipment_id": "P-101"}
Observation: P-101: operational, vibrasi=3.1 mm/s (normal <4.5), temp=72°C

Thought: Vibration is within limits but increasing. I should find the maintenance SOP.
Action: sop_search
Action Input: {"query": "pump P-101 vibration bearing maintenance"}
Observation: [Source: SOP-MECH-005] Check vibration < 4.5 mm/s...

Thought: I have enough context to draft a response.
Final Answer: Berdasarkan analisis...
```

### Building a ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# ReAct prompt (must include: {tools}, {tool_names}, {input}, {agent_scratchpad})
react_prompt = PromptTemplate.from_template("""
Answer using these tools: {tools}
Tool names: {tool_names}

Thought: reason step
Action: tool name
Action Input: input
Observation: result
Final Answer: answer in Indonesian

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,         # Show thought/action trace
    max_iterations=6,     # Prevent infinite loops
    handle_parsing_errors=True  # Don't crash on bad LLM output
)

result = executor.invoke({'input': 'Analisis tiket M01'})
```

### Stopping Conditions

| Condition | Parameter | Default |
|-----------|-----------|---------|
| Max iterations | `max_iterations=6` | 15 |
| Max execution time | `max_execution_time=30` | None |
| Early stopping | `early_stopping_method='generate'` | 'force' |
| Parse errors | `handle_parsing_errors=True` | False |

### Agent vs. Graph Decision Matrix

| Need | Agent | Graph |
|------|-------|-------|
| LLM decides tool sequence | ✓ | ✗ |
| Fixed conditional routing | ✗ | ✓ |
| Human-in-the-loop pause | ✗ | ✓ |
| Resume from checkpoint | ✗ | ✓ |
| Parallel node execution | ✗ | ✓ |
| Simple sequential tool use | ✓ | both |

---

## 5. Memory Abstractions

### Types of Memory

| Memory type | Storage | Token growth | Best for |
|-------------|---------|-------------|----------|
| **ConversationBufferMemory** | All messages | O(n) | Short sessions (<10 turns) |
| **ConversationBufferWindowMemory** | Last k turns | O(k) constant | Medium sessions |
| **ConversationSummaryMemory** | LLM-generated summary | O(1) after summarization | Long sessions |
| **VectorStoreRetrieverMemory** | Semantically retrieved | O(1) | Very long/non-linear sessions |

### ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Turn 1
conversation.predict(input="Pompa P-101 bunyi berisik.")
# Turn 2 — memory contains Turn 1 automatically
conversation.predict(input="Apakah perlu ganti bearing?")
```

### LangGraph State as Memory

The preferred modern approach: store conversation history in the graph state.

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]   # Full conversation history
    ticket_id: str
    category: str
    # ... other state fields

def agent_node(state: AgentState) -> dict:
    # Messages accumulate in state — graph manages history
    response = llm.invoke(state['messages'])
    return {'messages': state['messages'] + [response]}
```

**Advantages of state-based memory:**
- Explicit — you see exactly what's in context at every step
- Checkpointable — save/resume anywhere in the conversation
- Type-safe — TypedDict enforces schema
- Inspectable — can log/audit every message in the history

### Token Budget for Memory

```
Turn 1:  ~100 tokens
Turn 2:  ~100 tokens
Turn 5:  ~500 tokens (full buffer)
Turn 10: ~1,000 tokens (growing)
```

For Claude's 200,000-token window, buffer memory is rarely a problem for support
conversations (<50 turns). Use window or summary memory for:
- Call center transcripts (hundreds of turns)
- Document review sessions (large context per turn)
- Multi-day agent tasks

---

## 6. LangGraph Fundamentals

### Core Concepts

```
StateGraph = directed graph where nodes transform state

State (TypedDict)
 ├── Input fields  : ticket_id, ticket_text
 ├── Computed fields: category, priority, equipment_id
 ├── Retrieved fields: sop_content, equipment_status
 └── Output fields  : draft_response, escalated

Nodes = Python functions: (state) → dict (partial state update)
Edges = Transitions: node_a → node_b
Conditional edges = Route based on state: (state) → node_name
```

### Building a StateGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class TicketState(TypedDict):
    ticket_id: str
    ticket_text: str
    category: Optional[str]
    priority: Optional[str]
    sop_content: Optional[str]
    draft_response: Optional[str]
    escalated: bool

# Define nodes
def classify_node(state: TicketState) -> dict:
    # ... classify the ticket
    return {'category': 'Safety', 'priority': 'Kritis'}

def retrieve_sop_node(state: TicketState) -> dict:
    sop = sop_search(state['ticket_text'])
    return {'sop_content': sop}

def draft_node(state: TicketState) -> dict:
    draft = llm.invoke([...context from state...]).content
    return {'draft_response': draft}

def escalate_node(state: TicketState) -> dict:
    return {'escalated': True, 'draft_response': 'Ticket escalated to supervisor.'}

# Routing function
def route_after_classify(state: TicketState) -> str:
    return 'escalate' if state.get('priority') == 'Kritis' else 'retrieve_sop'

# Build graph
graph = StateGraph(TicketState)
graph.add_node('classify', classify_node)
graph.add_node('retrieve_sop', retrieve_sop_node)
graph.add_node('draft', draft_node)
graph.add_node('escalate', escalate_node)

graph.set_entry_point('classify')
graph.add_conditional_edges('classify', route_after_classify,
                             {'escalate': 'escalate', 'retrieve_sop': 'retrieve_sop'})
graph.add_edge('retrieve_sop', 'draft')
graph.add_edge('draft', END)
graph.add_edge('escalate', END)

app = graph.compile()
result = app.invoke({'ticket_id': 'K05', 'ticket_text': '...', 'escalated': False, ...})
```

### Node Function Contract

```python
def my_node(state: TicketState) -> dict:
    """
    Input:  complete current state (TypedDict)
    Output: dict with ONLY the fields this node updates
            (LangGraph merges output into state — no need to return all fields)
    """
    ...
    return {'category': 'Safety'}  # Only update category, leave rest unchanged
```

### Graph Visualization (ASCII)

```
START
  │
  ▼
classify
  │
  ├─── priority=='Kritis' ──→ escalate ──→ END
  │
  └─── otherwise ──→ retrieve_sop ──→ draft ──→ END
```

---

## 7. Multi-Step & Conditional Workflows

### Conditional Edge Patterns

```python
# Pattern 1: Simple binary routing
graph.add_conditional_edges(
    'classify',
    lambda state: 'escalate' if state['priority'] == 'Kritis' else 'normal',
    {'escalate': 'escalate_node', 'normal': 'draft_node'}
)

# Pattern 2: Multi-way routing
def route_by_category(state):
    category = state.get('category', 'General')
    routes = {
        'Safety': 'safety_handler',
        'Mechanical': 'mechanical_handler',
        'SAP': 'sap_handler',
    }
    return routes.get(category, 'generic_handler')

graph.add_conditional_edges('classify', route_by_category,
    {'safety_handler': 'safety_node', 'mechanical_handler': 'mech_node',
     'sap_handler': 'sap_node', 'generic_handler': 'generic_node'})
```

### Retry Loops in LangGraph

```python
class RetryState(TypedDict):
    query: str
    sop_content: Optional[str]
    retry_count: int

def retrieve_with_retry(state: RetryState) -> dict:
    result = sop_search(state['query'])
    if 'Tidak ada' in result and state['retry_count'] < 2:
        # Refine query and retry
        refined = refine_query(state['query'])
        return {'query': refined, 'retry_count': state['retry_count'] + 1}
    return {'sop_content': result}

def should_retry(state: RetryState) -> str:
    if 'Tidak ada' in (state.get('sop_content') or '') and state['retry_count'] < 2:
        return 'retrieve'   # Loop back
    return 'next_node'      # Continue

graph.add_conditional_edges('retrieve', should_retry,
    {'retrieve': 'retrieve', 'next_node': 'draft'})
```

### Parallel Node Execution

```python
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableParallel

# Method 1: RunnableParallel inside a node
def parallel_lookup_node(state: TicketState) -> dict:
    parallel = RunnableParallel(
        equipment=lambda s: equipment_status(s['equipment_id']),
        sop=lambda s: sop_search(s['ticket_text'])
    )
    results = parallel.invoke(state)
    return {'equipment_status': results['equipment'], 'sop_content': results['sop']}
```

### Subgraphs

```python
# Build a sub-graph for safety handling
safety_graph = StateGraph(TicketState)
safety_graph.add_node('safety_classify', ...)
safety_graph.add_node('safety_alert', ...)
safety_compiled = safety_graph.compile()

# Use as a node in parent graph
main_graph.add_node('safety_handler', safety_compiled)
```

---

## 8. Error Handling & Human-in-the-Loop

### Error Handling Strategies

| Strategy | LangChain mechanism | When to use |
|----------|-------------------|-------------|
| Fallback chain | `chain.with_fallbacks([backup])` | LLM API outage |
| Retry on error | `chain.with_retry(stop_after_attempt=3)` | Transient API errors |
| Graph retry loop | Conditional edge back to same node | Logic-level retry |
| Try/except in node | Standard Python `try/except` | Tool failures |
| Max iterations | `AgentExecutor(max_iterations=6)` | Infinite loop prevention |

```python
# Fallback: if primary LLM fails, use backup
primary = ChatAnthropic(model='claude-sonnet-4-6')
backup  = ChatAnthropic(model='claude-haiku-4-5-20251001')
chain_with_fallback = (prompt | primary | parser).with_fallbacks(
    [prompt | backup | parser]
)

# Retry: automatic exponential backoff on rate limits
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
```

### Human-in-the-Loop (HITL)

**`interrupt_before`:** Pause graph execution before a specified node. The application
receives control, can present the state to a human, collect their decision, and resume.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(
    checkpointer=memory,
    interrupt_before=['human_approval']  # Pause before this node
)

# Thread ID for this conversation
config = {'configurable': {'thread_id': 'ticket-M02'}}

# Run until interrupt
state = app.invoke(initial_state, config)
# → Graph pauses at 'human_approval', returns current state

# Human reviews state, makes decision
# Resume with updated state
updated_state = {**state, 'approved': True}
final_state = app.invoke(updated_state, config)
```

**`interrupt_after`:** Pause graph after a node (useful to review tool output before continuing).

### HITL Pattern for Work Order Approval

```
[Ticket arrives]
      │
      ▼
  classify → retrieve_sop → draft_response
                                  │
                          [interrupt_before]
                                  │
                          human_approval ← supervisor reviews draft
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                  approved               rejected
                     │                         │
               create_work_order        request_more_info
                     │                         │
                   END                       END
```

### Escalation Path Design

```python
ESCALATION_RULES = {
    ('Safety', 'Kritis'):  {'to': 'HSE Supervisor', 'sla_minutes': 15},
    ('Mechanical', 'Kritis'): {'to': 'Maintenance Supervisor', 'sla_minutes': 30},
    ('Electrical', 'Kritis'): {'to': 'Electrical Supervisor', 'sla_minutes': 30},
    ('SAP', 'Tinggi'):     {'to': 'SAP Key User', 'sla_minutes': 120},
}

def escalate_node(state: TicketState) -> dict:
    key = (state.get('category'), state.get('priority'))
    rule = ESCALATION_RULES.get(key, {'to': 'General Supervisor', 'sla_minutes': 240})
    return {
        'escalated': True,
        'draft_response': (
            f"[ESKALASI] Tiket {state['ticket_id']} diteruskan ke {rule['to']}. "
            f"SLA: {rule['sla_minutes']} menit."
        )
    }
```

---

## 9. Quick Reference

### LangChain Installation

```bash
pip install langchain langchain-core langchain-anthropic langgraph python-dotenv
pip install langchain-openai   # if using OpenAI
```

### Environment Setup (.env)

```bash
ANTHROPIC_API_KEY=your-key-here
# OPENAI_API_KEY=your-key-here  # alternative
```

### Key Classes & Functions

| Import | Class/Function | Purpose |
|--------|---------------|---------|
| `langchain_core.prompts` | `ChatPromptTemplate` | Multi-role prompt templates |
| `langchain_core.output_parsers` | `StrOutputParser`, `JsonOutputParser` | Parse LLM output |
| `langchain_core.tools` | `@tool` | Define callable tools |
| `langchain_core.messages` | `HumanMessage`, `AIMessage`, `SystemMessage` | Typed chat messages |
| `langchain_anthropic` | `ChatAnthropic` | Anthropic Claude LLM |
| `langchain.agents` | `create_react_agent`, `AgentExecutor` | ReAct agent |
| `langgraph.graph` | `StateGraph`, `END` | Graph builder |
| `langgraph.checkpoint.memory` | `MemorySaver` | In-memory checkpointer |

### LangGraph Node Signature

```python
# ✓ Correct: takes full state, returns partial state dict
def my_node(state: MyState) -> dict:
    return {'field_to_update': new_value}

# ✗ Wrong: returns full state (causes merge conflicts)
def bad_node(state: MyState) -> MyState:
    state['field'] = value
    return state
```

### LCEL Quick Reference

```python
# Simple chain
chain = prompt | llm | StrOutputParser()

# With retry
chain = (prompt | llm | StrOutputParser()).with_retry(stop_after_attempt=3)

# With fallback
chain = (prompt | primary_llm).with_fallbacks([prompt | backup_llm])

# Parallel
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel(cat=classify_chain, sum=summary_chain)

# Passthrough (keep input alongside output)
from langchain_core.runnables import RunnablePassthrough
chain = RunnablePassthrough.assign(category=classify_chain)
```

### Glossary

| Term | Definition |
|------|-----------|
| **LCEL** | LangChain Expression Language — pipe-based chain composition |
| **Runnable** | Any component with `.invoke()/.batch()/.stream()` |
| **Tool** | Python function callable by the LLM via JSON schema |
| **ReAct** | Reason + Act — agentic pattern interleaving thought and tool calls |
| **StateGraph** | LangGraph directed graph with typed state |
| **Node** | Graph processing function: `(state) → dict` |
| **Conditional edge** | Graph edge that routes to different nodes based on state |
| **HITL** | Human-in-the-Loop — pause graph for human review |
| **interrupt_before** | LangGraph mechanism to pause before a node |
| **Checkpointer** | Persistence backend for graph state (MemorySaver, SQLite, Redis) |
| **AgentExecutor** | LangChain wrapper running a ReAct loop with tools |
| **Scratchpad** | LLM's internal reasoning trace during ReAct loop |

---

## 10. Day 5 Preview — Deployment & Production Systems

Day 4 built the orchestration logic in a local notebook. Day 5 takes that agent
to production:

| Day 4 (tonight) | Day 5 (tomorrow) |
|----------------|-----------------|
| Notebook agent | FastAPI REST service with auth |
| Anthropic/OpenAI API | Local LLM: Ollama (Llama 3.1) or vLLM |
| Mock CMMS tools | Real API integrations (SAP RFC, historian) |
| Single process | Multi-worker, async |
| No persistence | PostgreSQL checkpointer for graph state |
| No monitoring | LangSmith tracing, Prometheus metrics |
| Jupyter notebook | Docker container, CI/CD pipeline |

**What you'll build in Day 5:**
1. Package the Day 4 agent as a FastAPI endpoint
2. Run Claude-equivalent models locally with Ollama
3. High-throughput inference with vLLM for GPU deployments
4. Monitor cost, latency, and quality in LangSmith
5. End-to-end architecture: from ticket ingestion to production support system

**Day 5 architecture:**

```
Plant Technician
     │  (ticket via app/API)
     ▼
FastAPI Service
     │
     ├── LangGraph Agent (Day 4)
     │       ├── Tools: ticket_lookup, sop_search, create_wo
     │       └── Memory: PostgreSQL checkpointer
     │
     ├── LLM Backend
     │       ├── Cloud: Anthropic API (Day 2–4)
     │       └── Local: Ollama / vLLM (Day 5)
     │
     └── Monitoring
             ├── LangSmith (trace every agent call)
             └── Prometheus (latency, cost, error rate)
```

*See you in Day 5. The agent you built tonight will serve real tickets tomorrow.*

---

*Day 4 Handout — Industrial AI & LLM Training Program*
*Building on: Day 3 RAG pipeline | Forward to: Day 5 Deployment*
