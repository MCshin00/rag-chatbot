"""임베딩 호출과 토큰 추정. 재시도는 하지 않는다."""

import hashlib
import json
from pathlib import Path

import tiktoken

from .config import EMBEDDING_MODEL, EMBEDDING_PRICES


def count_tokens(texts: list[str]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")  # text-embedding-3 계열의 토크나이저
    return sum(len(enc.encode(t)) for t in texts)


def cost(tokens: int, model: str = EMBEDDING_MODEL) -> float:
    return tokens * EMBEDDING_PRICES[model] / 1e6


def embed(texts: list[str], model: str = EMBEDDING_MODEL) -> tuple[list[list[float]], int]:
    """한 번의 요청으로 임베딩한다. (벡터 목록, 실제 사용 토큰)."""
    from openai import OpenAI

    client = OpenAI(max_retries=0)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)], resp.usage.total_tokens


class QueryCache:
    """질문 임베딩을 모델·질문 문장 기준으로 저장해 두 방식과 여러 실행에서 다시 쓴다."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(model: str, question: str) -> str:
        return hashlib.sha256(f"{model}|{question}".encode()).hexdigest()

    def missing(self, model: str, questions: list[str]) -> list[str]:
        return list(dict.fromkeys(q for q in questions if self.key(model, q) not in self.data))

    def fill(self, model: str, questions: list[str]) -> int:
        """없는 질문만 한 번의 요청으로 임베딩해 저장한다. 실제 사용 토큰을 돌려준다."""
        todo = self.missing(model, questions)
        if not todo:
            return 0
        vecs, used = embed(todo, model)
        self.data.update({self.key(model, q): v for q, v in zip(todo, vecs)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        return used

    def get(self, model: str, question: str) -> list[float]:
        return self.data[self.key(model, question)]
