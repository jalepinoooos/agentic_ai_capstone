"""
api/main.py — FastAPI REST endpoint for the MediCare Hospital Agent.

Launch: uvicorn medicare_assistant.api.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache

from ..graph import build_graph, ask as agent_ask

api = FastAPI(title="MediCare Hospital Assistant API", version="1.0.0")


@lru_cache(maxsize=1)
def get_app():
    app, _, _ = build_graph()
    return app


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    route: str
    faithfulness: float
    sources: list[str]


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    app = get_app()
    result = agent_ask(app, req.question, thread_id=req.thread_id)
    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", ""),
        faithfulness=result.get("faithfulness", 0.0),
        sources=result.get("sources", []),
    )
