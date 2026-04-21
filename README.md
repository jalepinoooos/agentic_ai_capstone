# Agentic AI Capstone — MediCare Hospital Assistant

#gauravi singh 2305780


---

## Project structure

```
capstone/
├── medicare_assistant/
│   ├── __init__.py
│   ├── state.py           # CapstoneState TypedDict
│   ├── knowledge_base.py  # 10 KB documents + ChromaDB setup
│   ├── tools.py           # datetime, calculator, emergency tools
│   ├── nodes.py           # 8 node functions
│   ├── graph.py           # Graph assembly + compile
│   └── api/
│       └── main.py        # FastAPI endpoint (optional)
├── tests/
│   ├── test_nodes.py      # Isolation tests (no API key needed)
│   └── test_integration.py # Full graph tests (needs GROQ_API_KEY)
├── capstone_streamlit.py  # Streamlit UI
├── ragas_eval.py          # RAGAS baseline evaluation
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your_key_here"
```

Get a free key at: https://console.groq.com

### 3. Run the Streamlit app

```bash
streamlit run capstone_streamlit.py
```

### 4. Run isolation tests (no API key needed)

```bash
pytest tests/test_nodes.py -v
```

### 5. Run integration tests (needs GROQ_API_KEY)

```bash
pytest tests/test_integration.py -v
```

### 6. Run RAGAS evaluation

```bash
python ragas_eval.py
```

### 7. Run FastAPI (optional)

```bash
uvicorn medicare_assistant.api.main:app --reload
# Test: curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"What are OPD timings?","thread_id":"test1"}'
```

---

## Architecture

```
User question
     ↓
[memory_node]     → append to history, sliding window (last 6), extract name
     ↓
[router_node]     → LLM → retrieve / tool / memory_only
     ↓
[retrieval_node / tool_node / skip_node]
     ↓
[answer_node]     → system prompt + context + history → LLM response
     ↓
[eval_node]       → faithfulness 0.0–1.0 → retry if < 0.7 (max 2 retries)
     ↓
[save_node]       → append answer to messages → END
```

---

## 6 mandatory capabilities

| # | Capability | Where |
|---|---|---|
| 1 | LangGraph StateGraph (8 nodes) | `graph.py` |
| 2 | ChromaDB RAG (10 docs) | `knowledge_base.py` |
| 3 | MemorySaver + thread_id | `graph.py` → `build_graph()` |
| 4 | Self-reflection eval node | `nodes.py` → `eval_node()` |
| 5 | Tool use (datetime, calculator, emergency) | `tools.py`, `nodes.py` |
| 6 | Streamlit deployment | `capstone_streamlit.py` |

---

## Submission checklist

- [ ] `day13_capstone.ipynb` — Kernel > Restart & Run All passes
- [ ] `capstone_streamlit.py` — multi-turn conversation works in browser
- [ ] `medicare_assistant/` package — all TODO sections replaced
- [ ] RAGAS baseline scores recorded in written summary
- [ ] 10 test questions + 2 red-team tests documented
