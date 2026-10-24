"""검색 방식 조립: 벡터 검색만, 또는 벡터 + BM25를 RRF로 결합."""

from collections.abc import Callable

import psycopg

from . import store
from .config import HYBRID_POOL, RRF_K
from .lexical import BM25, rrf

Search = Callable[[str, list[float]], list[dict]]


def make_search(conn: psycopg.Connection, model: str, strategy: str, mode: str, k: int) -> Search:
    """(질문, 질문 벡터) → 상위 k개 조각."""
    if mode == "vector":
        return lambda question, vec: store.search(conn, model, strategy, vec, k)

    chunks = store.all_chunks(conn, model, strategy)
    by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = BM25([c["text"] for c in chunks])

    def hybrid(question: str, vec: list[float]) -> list[dict]:
        dense = store.search(conn, model, strategy, vec, HYBRID_POOL)
        scores = bm25.scores(question)
        lexical = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["chunk_id"]))[:HYBRID_POOL]
        fused = rrf([[c["chunk_id"] for c in dense], [chunks[i]["chunk_id"] for i in lexical if scores[i] > 0]], RRF_K)
        distance = {c["chunk_id"]: c["distance"] for c in dense}
        return [by_id[cid] | {"distance": distance.get(cid)} for cid in fused[:k]]

    return hybrid
