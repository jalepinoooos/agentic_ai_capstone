"""
Graph assembly for the MediCare Hospital Agent.
Builds and compiles the LangGraph StateGraph with MemorySaver checkpointing.
"""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

from .state import MedicareState
from .knowledge_base import build_knowledge_base
from .nodes import (
    init_nodes,
    memory_node,
    router_node,
    retrieval_node,
    skip_retrieval_node,
    tool_node,
    answer_node,
    eval_node,
    save_node,
    FAITHFULNESS_THRESHOLD,
    MAX_EVAL_RETRIES,
)


# ── Routing functions ──────────────────────────────────────────────────────────

def route_decision(state: MedicareState) -> str:
    """Read state.route and return the next node name."""
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    elif route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: MedicareState) -> str:
    """
    If faithfulness is below threshold AND retries not exhausted → retry answer.
    Otherwise → save.
    """
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
        return "answer"
    return "save"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Initialise all resources, wire the graph, compile with MemorySaver,
    and return (app, embedder, collection).
    """
    # 1. LLM
    api_key = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.1)

    # 2. Knowledge base
    embedder, collection = build_knowledge_base()

    # 3. Inject into nodes module
    init_nodes(llm, embedder, collection)

    # 4. Build graph
    graph = StateGraph(MedicareState)

    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    # Entry point
    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory", "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("save", END)

    # Conditional edges
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    # Compile
    app = graph.compile(checkpointer=MemorySaver())
    print("Graph compiled successfully.")
    return app, embedder, collection


# ── Helper ─────────────────────────────────────────────────────────────────────

def ask(app, question: str, thread_id: str = "default") -> dict:
    """Invoke the compiled graph and return the final state."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: MedicareState = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name": None,
    }
    result = app.invoke(initial_state, config=config)
    return result
