from typing import TypedDict, List, Optional


class MedicareState(TypedDict):
    question: str
    messages: List[dict]
    route: str                  # "retrieve" | "tool" | "memory_only"
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: Optional[str]
