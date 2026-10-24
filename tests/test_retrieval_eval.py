"""조각 나누기, 근거 충족 계산, 그래프의 보류 분기를 대표 사례로 확인한다. 유료 호출 없음."""

from rag_chatbot.corpus import Span, article_chunks, fixed_chunks, load_corpus, locate_quote
from rag_chatbot.graph import Answer, build_graph
from rag_chatbot.lexical import BM25, bigrams, rrf
from rag_chatbot.metrics import article_recall, covered, evidence_coverage

CORPUS = load_corpus()
LIMIT_Q = {
    "answerable": True,
    "evidence_sets": [[
        {"doc_id": "act", "article": "32", "paragraph": 1, "quote": "각 예금자등의 예금등 채권의 합계액에서"},
        {"doc_id": "decree", "article": "18", "paragraph": 7, "quote": "보험금의 지급한도는 1억원"},
    ]],
}


def test_chunk_counts_and_deleted_articles_excluded():
    arts = article_chunks(CORPUS)
    assert len(arts) == 132
    assert not any(c.chunk_id in ("article:act:19", "article:act:38의2") for c in arts)
    fixed = fixed_chunks(CORPUS)
    assert all(len(c.text) <= 400 for c in fixed)
    # 이웃한 고정 크기 조각은 50자 겹친다
    a, b = fixed[0], fixed[1]
    assert a.doc_id == b.doc_id and a.span.end - b.span.start == 50


def test_fixed_chunk_spanning_article_one_does_not_cover_paragraph_seven():
    """제18조 ①만 담은 조각은 조 번호로는 '제18조'지만 ⑦의 한도 구절을 덮지 않는다."""
    target = locate_quote(CORPUS, "decree", "18", 7, "보험금의 지급한도는 1억원")
    art18 = CORPUS.articles[("decree", "18")].span
    first = next(c for c in fixed_chunks(CORPUS) if c.doc_id == "decree" and c.span.start <= art18.start < c.span.end)
    assert "18" in first.articles
    assert not covered([first.span], target)
    assert article_recall(LIMIT_Q, [("decree", first.articles)], 1) == 0.5  # 조 단위로는 절반을 찾은 것처럼 보인다


def test_coverage_unions_adjacent_spans_and_separates_partial():
    target = Span("decree", 100, 200)
    assert covered([Span("decree", 50, 150), Span("decree", 140, 220)], target)
    assert not covered([Span("decree", 50, 150), Span("decree", 160, 220)], target)  # 사이가 비면 안 된다
    act32 = CORPUS.articles[("act", "32")].span
    dec18 = CORPUS.articles[("decree", "18")].span
    assert evidence_coverage(CORPUS, LIMIT_Q, [act32, dec18]) == "full"
    assert evidence_coverage(CORPUS, LIMIT_Q, [act32]) == "partial"
    assert evidence_coverage(CORPUS, LIMIT_Q, []) == "none"


class FakeChat:
    """with_structured_output(...).invoke(...)가 정해 둔 응답을 돌려주는 가짜 모델."""

    def __init__(self, parsed=None, error=None):
        self.parsed, self.error = parsed, error

    def with_structured_output(self, *a, **k):
        return self

    def invoke(self, messages):
        if self.error:
            raise self.error
        return {"raw": None, "parsed": self.parsed, "parsing_error": None}


def run(parsed=None, error=None, retrieved=None):
    retrieved = [{"law_name": "예금자보호법", "text": "본문", "span": Span("act", 0, 2)}] if retrieved is None else retrieved
    graph = build_graph(lambda question, vec: retrieved, FakeChat(parsed, error))
    return graph.invoke({"question": "질문", "query_vec": [0.0]})["final"]


def test_answer_with_valid_citation_passes():
    assert run(Answer(answer="1억 원까지", citations=["C1"], abstain=False))["status"] == "answer"


def test_nonexistent_or_empty_citation_is_blocked():
    assert run(Answer(answer="1억 원까지", citations=["C4"], abstain=False))["reason"] == "citation_check"
    assert run(Answer(answer="1억 원까지", citations=[], abstain=False))["reason"] == "citation_check"


def test_model_abstain_and_errors_are_separated():
    assert run(Answer(answer="확인할 수 없다", citations=[], abstain=True))["reason"] == "model"
    assert run(error=TimeoutError())["status"] == "error"  # 실행 오류는 보류가 아니다
    assert run(retrieved=[])["reason"] == "no_retrieval"


def test_bm25_bigrams_match_through_particles_and_rrf_merges():
    assert "한도" in bigrams("지급한도는")  # 조사가 붙어도 같은 두 글자가 남는다
    docs = ["보험금의 지급한도는 1억원으로 한다", "양도성예금증서는 예금등에 포함되지 아니한다", "벌칙"]
    scores = BM25(docs).scores("CD 양도성예금증서도 보호되나요")
    assert max(range(3), key=scores.__getitem__) == 1
    assert rrf([["a", "b", "c"], ["c", "a"]], 60)[:2] == ["a", "c"]
