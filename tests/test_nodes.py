"""
tests/test_nodes.py — Isolation tests for each node function.

Run: pytest tests/test_nodes.py -v

Each test creates a minimal mock state and calls the node directly,
without running the full graph.
"""

import pytest
from unittest.mock import MagicMock, patch
from medicare_assistant.state import MedicareState


def make_state(**overrides) -> MedicareState:
    base: MedicareState = {
        "question": "What are OPD timings?",
        "messages": [],
        "route": "retrieve",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name": None,
    }
    base.update(overrides)
    return base


# ── memory_node ────────────────────────────────────────────────────────────────

def test_memory_node_appends_question():
    from medicare_assistant.nodes import memory_node
    state = make_state(question="Hello, what are your timings?", messages=[])
    result = memory_node(state)
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"


def test_memory_node_extracts_name():
    from medicare_assistant.nodes import memory_node
    state = make_state(question="My name is Priya, what are OPD timings?")
    result = memory_node(state)
    assert result["user_name"] == "Priya"


def test_memory_node_sliding_window():
    from medicare_assistant.nodes import memory_node
    existing = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    state = make_state(question="New question", messages=existing)
    result = memory_node(state)
    assert len(result["messages"]) <= 6


# ── skip_retrieval_node ────────────────────────────────────────────────────────

def test_skip_retrieval_returns_empty():
    from medicare_assistant.nodes import skip_retrieval_node
    state = make_state(retrieved="old context", sources=["old"])
    result = skip_retrieval_node(state)
    assert result["retrieved"] == ""
    assert result["sources"] == []


# ── tool_node ──────────────────────────────────────────────────────────────────

def test_tool_node_datetime():
    from medicare_assistant.nodes import tool_node
    state = make_state(question="What is today's date?")
    result = tool_node(state)
    assert "Current date" in result["tool_result"]


def test_tool_node_emergency():
    from medicare_assistant.nodes import tool_node
    state = make_state(question="I have an emergency, need ambulance!")
    result = tool_node(state)
    assert "040-99999999" in result["tool_result"]


def test_tool_node_calculator():
    from medicare_assistant.nodes import tool_node
    state = make_state(question="Calculate 300 + 600")
    result = tool_node(state)
    assert "900" in result["tool_result"]


# ── eval_node ──────────────────────────────────────────────────────────────────

def test_eval_node_skips_when_no_retrieved():
    from medicare_assistant import nodes
    nodes._llm = MagicMock()
    from medicare_assistant.nodes import eval_node
    state = make_state(retrieved="", answer="Hello! How can I help?")
    result = eval_node(state)
    assert result["faithfulness"] == 1.0


# ── save_node ──────────────────────────────────────────────────────────────────

def test_save_node_appends_answer():
    from medicare_assistant.nodes import save_node
    state = make_state(
        messages=[{"role": "user", "content": "What are timings?"}],
        answer="OPD is open 8 AM to 1 PM.",
    )
    result = save_node(state)
    assert result["messages"][-1]["role"] == "assistant"
    assert "OPD" in result["messages"][-1]["content"]


# ── tools.py direct tests ──────────────────────────────────────────────────────

def test_tool_datetime_returns_string():
    from medicare_assistant.tools import get_current_datetime
    result = get_current_datetime()
    assert isinstance(result, str)
    assert "Current date" in result


def test_tool_calculator_valid():
    from medicare_assistant.tools import calculate
    assert "900" in calculate("300 + 600")


def test_tool_calculator_invalid():
    from medicare_assistant.tools import calculate
    result = calculate("import os")
    assert "Error" in result


def test_tool_calculator_division_by_zero():
    from medicare_assistant.tools import calculate
    result = calculate("10 / 0")
    assert "zero" in result.lower()
