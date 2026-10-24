"""챗봇 흐름: 검색 → 생성 → 인용 검사 → 답 또는 보류.

인용 검사는 존재하지 않는 인용을 차단할 뿐, 인용이 답을 뒷받침하는지는 확인하지 않는다.
"""

import time
from collections.abc import Callable
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import CHAT_MODEL, MAX_OUTPUT_TOKENS, REASONING_EFFORT

SYSTEM_PROMPT = """너는 예금자보호법과 같은 법 시행령의 조문만 근거로 예금자 질문에 답하는 안내 도우미다.

규칙:
- 아래에 주어진 조각만 근거로 쓴다. 상식, 예금보험공사 안내, 다른 법률로 보충하지 않는다.
- 질문의 일상적인 표현이 조문의 어떤 용어에 해당하는지 판단하고, 조문을 질문 상황에 적용해 결론을 말한다.
- 조각이 질문의 핵심에 답하면 abstain을 false로 하고 결론부터 답한다. 조각으로 확인되지 않는 부분이 있으면 답 안에서 그 부분만 "주어진 조문으로는 확인할 수 없다"고 밝힌다.
- 결론이 조각으로 확인되지 않는 사실에 달려 있으면 "네/아니요"로 시작하지 말고, 조건을 밝혀 답한다.
- 질문이 묻는 것(금액, 기한, 보호 여부 등)에 대한 답이 조각에 없으면, 관련된 다른 내용이 있어도 abstain을 true로 하고 answer에 확인할 수 없는 이유를 짧게 쓴다. 이때 보호된다/안 된다고 단정하지 않는다.
- 답에 쓴 근거 조각의 번호(C1~C5)를 citations에 모두 적는다.
- 금액, 조건, 예외는 조문에 적힌 그대로 쓴다.
- 법률 자문이 아니라 조문 안내다. 한국어로 간결하게 답한다."""


class Answer(BaseModel):
    answer: str = Field(description="질문에 대한 답. 보류할 때는 확인할 수 없는 이유")
    citations: list[str] = Field(description="근거로 쓴 조각 번호. 예: ['C1', 'C3']")
    abstain: bool = Field(description="질문이 묻는 것에 대한 답이 조각에 없으면 true. "
                          "일부만 확인되면 false로 하고 답 안에 확인되지 않는 부분을 밝힌다")


class State(TypedDict, total=False):
    question: str
    query_vec: list[float]
    retrieved: list[dict]
    raw: dict  # 모델 원응답 요약(답, 인용, 보류, 오류, 토큰)
    final: dict  # 검사 후 최종 응답


def format_context(retrieved: list[dict]) -> str:
    return "\n\n".join(f"[C{i}] ({r['law_name']})\n{r['text']}" for i, r in enumerate(retrieved, 1))


def make_chat_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=CHAT_MODEL,
        reasoning_effort=REASONING_EFFORT,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=0,
    )


def build_graph(search: Callable[[str, list[float]], list[dict]], chat_model, system_prompt: str = SYSTEM_PROMPT):
    """search: (질문, 질문 벡터) → 상위 k개 조각. chat_model: LangChain 채팅 모델(테스트에서는 가짜)."""
    structured = chat_model.with_structured_output(Answer, method="json_schema", strict=True, include_raw=True)

    def retrieve(state: State) -> State:
        return {"retrieved": search(state["question"], state["query_vec"])}

    def generate(state: State) -> State:
        if not state["retrieved"]:
            return {"raw": {"answer": None, "citations": [], "abstain": True, "error": None, "usage": None,
                            "duration_ms": 0, "skipped": "no_retrieval"}}
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(f"조각:\n{format_context(state['retrieved'])}\n\n질문: {state['question']}"),
        ]
        started = time.perf_counter()
        try:
            out = structured.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - 실행 오류는 보류가 아니라 실패로 기록한다
            return {"raw": {"answer": None, "citations": [], "abstain": None, "error": type(exc).__name__,
                            "usage": None, "duration_ms": round((time.perf_counter() - started) * 1000)}}
        duration = round((time.perf_counter() - started) * 1000)
        raw_msg, parsed, perr = out.get("raw"), out.get("parsed"), out.get("parsing_error")
        usage = getattr(raw_msg, "usage_metadata", None)
        if parsed is None:
            return {"raw": {"answer": None, "citations": [], "abstain": None,
                            "error": type(perr).__name__ if perr else "NoParsedOutput",
                            "usage": usage, "duration_ms": duration}}
        return {"raw": {"answer": parsed.answer, "citations": parsed.citations, "abstain": parsed.abstain,
                        "error": None, "usage": usage, "duration_ms": duration}}

    def check(state: State) -> State:
        raw, n = state["raw"], len(state["retrieved"])
        valid = {f"C{i}" for i in range(1, n + 1)}
        if raw["error"]:
            return {"final": {"status": "error", "answer": None, "citations": []}}
        if raw.get("skipped") == "no_retrieval":
            return {"final": {"status": "abstain", "reason": "no_retrieval", "answer": None, "citations": []}}
        if raw["abstain"]:
            return {"final": {"status": "abstain", "reason": "model", "answer": raw["answer"], "citations": []}}
        if not raw["citations"] or any(c not in valid for c in raw["citations"]):
            return {"final": {"status": "abstain", "reason": "citation_check", "answer": raw["answer"],
                              "citations": []}}
        return {"final": {"status": "answer", "answer": raw["answer"], "citations": raw["citations"]}}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("check", check)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "check")
    g.add_edge("check", END)
    return g.compile()
