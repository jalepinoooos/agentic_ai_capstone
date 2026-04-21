"""
Node functions for the MediCare Hospital LangGraph agent.
Each node is a pure function: State → dict of updated fields.
Every node is tested in isolation (see tests/test_nodes.py) before graph assembly.
"""

import re
from langchain_groq import ChatGroq
from .state import MedicareState
from .tools import run_tool

# ── Constants ──────────────────────────────────────────────────────────────────
SLIDING_WINDOW = 6          # keep last 6 messages in context
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2
HELPLINE = "040-12345678"
EMERGENCY = "040-99999999"

# ── Shared LLM instance (imported from graph.py after init) ───────────────────
# We receive llm + embedder + collection via closure / module-level injection.
# See graph.py for how these are set.
_llm = None
_embedder = None
_collection = None


def init_nodes(llm: ChatGroq, embedder, collection):
    """Call once at startup to inject shared resources into this module."""
    global _llm, _embedder, _collection
    _llm = llm
    _embedder = embedder
    _collection = collection


# ── 1. memory_node ─────────────────────────────────────────────────────────────
def memory_node(state: MedicareState) -> dict:
    """
    Append the new question to messages, apply sliding window,
    and extract the patient name if introduced.
    """
    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": state["question"]})

    # Sliding window — keep last SLIDING_WINDOW messages
    messages = messages[-SLIDING_WINDOW:]

    # Extract patient name
    user_name = state.get("user_name")
    match = re.search(r"my name is\s+([A-Za-z]+)", state["question"], re.IGNORECASE)
    if match:
        user_name = match.group(1).capitalize()

    return {"messages": messages, "user_name": user_name}


# ── 2. router_node ─────────────────────────────────────────────────────────────
def router_node(state: MedicareState) -> dict:
    """
    Ask the LLM to classify the question into one of three routes.
    Reply must be a single word only.
    """
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-4:]
    )

    prompt = f"""You are a routing assistant for a hospital chatbot.
Classify the user's latest question into EXACTLY ONE of these routes:

- retrieve   → Question is about hospital services, doctors, timings, fees, insurance, pharmacy, lab, rooms, health packages, appointments
- tool       → Question requires current date/time, a calculation, or emergency contact info
- memory_only → Simple greeting, thank you, goodbye, or conversational filler that needs no lookup

Conversation so far:
{history_text}

Latest question: {state['question']}

Reply with ONE word only: retrieve, tool, or memory_only"""

    response = _llm.invoke(prompt)
    route = response.content.strip().lower().split()[0]
    if route not in ("retrieve", "tool", "memory_only"):
        route = "retrieve"  # safe default

    return {"route": route}


# ── 3. retrieval_node ──────────────────────────────────────────────────────────
def retrieval_node(state: MedicareState) -> dict:
    """Embed the question, query ChromaDB for top 3 chunks, format context."""
    query_embedding = _embedder.encode([state["question"]]).tolist()
    results = _collection.query(query_embeddings=query_embedding, n_results=3)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    sources = []
    for doc, meta in zip(docs, metas):
        topic = meta.get("topic", "Unknown")
        context_parts.append(f"[{topic}]\n{doc}")
        sources.append(topic)

    retrieved = "\n\n".join(context_parts)
    return {"retrieved": retrieved, "sources": sources}


# ── 4. skip_retrieval_node ─────────────────────────────────────────────────────
def skip_retrieval_node(state: MedicareState) -> dict:
    """No retrieval needed — return clean empty context."""
    return {"retrieved": "", "sources": []}


# ── 5. tool_node ───────────────────────────────────────────────────────────────
def tool_node(state: MedicareState) -> dict:
    """
    Decide which tool to call based on the question, run it,
    and store the result. Never raises — always returns a string.
    """
    question_lower = state["question"].lower()

    if any(w in question_lower for w in ("emergency", "ambulance", "urgent", "accident")):
        result = run_tool("emergency")
    elif any(w in question_lower for w in ("date", "time", "today", "day", "now")):
        result = run_tool("datetime")
    elif any(w in question_lower for w in ("calculate", "total", "how much", "+", "-", "*", "/")):
        # Extract numeric expression if present
        expr_match = re.search(r"[\d\s\+\-\*\/\(\)\.]+", state["question"])
        expr = expr_match.group().strip() if expr_match else state["question"]
        result = run_tool("calculator", expr)
    else:
        result = run_tool("datetime")  # safe fallback

    return {"tool_result": result}


# ── 6. answer_node ─────────────────────────────────────────────────────────────
def answer_node(state: MedicareState) -> dict:
    """
    Build the final answer using retrieved context or tool result.
    Grounding rule: answer ONLY from the provided context.
    """
    name_greeting = f"Hello {state['user_name']}! " if state.get("user_name") else ""
    retries = state.get("eval_retries", 0)

    escalation_note = ""
    if retries >= 1:
        escalation_note = (
            "\n\nIMPORTANT: Previous answer scored below faithfulness threshold. "
            "Be extra careful to use ONLY information explicitly stated in the context below. "
            "Do not add any information from general knowledge."
        )

    # Build context section
    context_section = ""
    if state.get("retrieved"):
        context_section = f"\n\nKNOWLEDGE BASE CONTEXT:\n{state['retrieved']}"
    if state.get("tool_result"):
        context_section += f"\n\nTOOL RESULT:\n{state['tool_result']}"

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-4:]
    )

    system_prompt = f"""You are a helpful patient assistant for MediCare General Hospital, Hyderabad.
Your role is to assist patients with information about the hospital's services.

STRICT RULES:
1. Answer ONLY from the KNOWLEDGE BASE CONTEXT or TOOL RESULT provided below.
2. If the information is not in the context, say clearly: "I don't have that information. Please call our helpline at {HELPLINE}."
3. NEVER give medical advice or clinical recommendations — redirect to doctors.
4. For any emergency, immediately provide the emergency number: {EMERGENCY}.
5. Do not fabricate doctor names, fees, or timings not explicitly mentioned.
6. Be empathetic, clear, and concise.{escalation_note}

CONVERSATION HISTORY:
{history_text}
{context_section}

Answer the patient's latest question: {state['question']}
{name_greeting}"""

    response = _llm.invoke(system_prompt)
    return {"answer": response.content.strip()}


# ── 7. eval_node ───────────────────────────────────────────────────────────────
def eval_node(state: MedicareState) -> dict:
    """
    Score faithfulness (0.0 – 1.0).
    Skips scoring if retrieved is empty (tool / memory_only routes).
    Increments eval_retries.
    """
    retries = state.get("eval_retries", 0)

    # Skip eval when there's no retrieved context to check against
    if not state.get("retrieved"):
        return {"faithfulness": 1.0, "eval_retries": retries}

    prompt = f"""You are a faithfulness evaluator.
Score whether the ANSWER uses ONLY information from the CONTEXT.

CONTEXT:
{state['retrieved']}

ANSWER:
{state['answer']}

Scoring rules:
- 1.0 = every claim in the answer is directly supported by the context
- 0.7-0.9 = mostly grounded, minor additions
- 0.4-0.6 = some hallucination present
- 0.0-0.3 = significant hallucination

Reply with a single decimal number between 0.0 and 1.0. Nothing else."""

    response = _llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.5  # assume partial if parse fails

    print(f"[eval_node] faithfulness={score:.2f} | retries={retries} | "
          f"{'RETRY' if score < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES else 'PASS'}")

    return {"faithfulness": score, "eval_retries": retries + 1}


# ── 8. save_node ───────────────────────────────────────────────────────────────
def save_node(state: MedicareState) -> dict:
    """Append the assistant's final answer to the messages history."""
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": state["answer"]})
    messages = messages[-SLIDING_WINDOW:]
    return {"messages": messages}
