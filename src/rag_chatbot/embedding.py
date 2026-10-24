"""임베딩 호출과 토큰 추정. 재시도는 하지 않는다."""

import tiktoken

from .config import EMBEDDING_MODEL

EMBEDDING_PRICE_PER_1M = 0.02  # 2026-10-24 공식 모델 페이지


def count_tokens(texts: list[str]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")  # text-embedding-3-small의 토크나이저
    return sum(len(enc.encode(t)) for t in texts)


def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """한 번의 요청으로 임베딩한다. (벡터 목록, 실제 사용 토큰)."""
    from openai import OpenAI

    client = OpenAI(max_retries=0)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)], resp.usage.total_tokens
