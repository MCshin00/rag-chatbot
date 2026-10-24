"""검색 지표: 근거 충족(주 지표)과 조 단위 Recall@k(보조 지표)."""

from .corpus import Corpus, Span, locate_quote


def covered(spans: list[Span], target: Span) -> bool:
    """여러 조각의 원문 범위를 합친 것이 target을 빈틈없이 덮는가."""
    pos = target.start
    for s in sorted((s for s in spans if s.doc_id == target.doc_id), key=lambda s: s.start):
        if s.start > pos:
            break
        pos = max(pos, s.end)
        if pos >= target.end:
            return True
    return pos >= target.end


def evidence_coverage(corpus: Corpus, question: dict, spans: list[Span]) -> str:
    """'full' | 'partial' | 'none'. 답 없는 문항은 'na'."""
    if not question["answerable"]:
        return "na"
    best = "none"
    for ev_set in question["evidence_sets"]:
        hits = [covered(spans, locate_quote(corpus, e["doc_id"], str(e["article"]), e["paragraph"], e["quote"]))
                for e in ev_set]
        if all(hits):
            return "full"
        if any(hits):
            best = "partial"
    return best


def article_recall(question: dict, retrieved_articles: list[tuple[str, tuple[str, ...]]], k: int) -> float | None:
    """retrieved_articles: 순위대로 (doc_id, 걸친 조 번호들). 묶음 중 가장 높은 값을 쓴다."""
    if not question["answerable"]:
        return None
    hit = {(doc_id, a) for doc_id, arts in retrieved_articles[:k] for a in arts}
    best = 0.0
    for ev_set in question["evidence_sets"]:
        gold = {(e["doc_id"], str(e["article"])) for e in ev_set}
        best = max(best, len(gold & hit) / len(gold))
    return best
