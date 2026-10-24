"""조문 JSONL을 읽어 법령별 본문과 위치 정보를 만들고, 두 방식으로 조각을 나눈다.

법령마다 삭제되지 않은 본칙 조문의 `text`를 줄바꿈 하나로 이어 붙인 문자열을 "본문"으로 둔다.
조각·조문·항·근거 구절의 위치는 모두 이 본문 안의 [start, end) 글자 위치로 나타낸다.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXED_SIZE = 400
FIXED_OVERLAP = 50


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int


@dataclass(frozen=True)
class ArticleLoc:
    doc_id: str
    article: str
    span: Span
    paragraphs: dict  # 항 번호(None 포함) → Span


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    doc_id: str
    law_name: str
    text: str
    span: Span
    articles: tuple[str, ...]  # 이 조각이 걸친 조 번호


@dataclass
class Corpus:
    law_names: dict  # doc_id → 법령명
    bodies: dict  # doc_id → 본문
    articles: dict  # (doc_id, 조 번호) → ArticleLoc

    def articles_in(self, span: Span) -> tuple[str, ...]:
        return tuple(
            loc.article for (doc_id, _), loc in self.articles.items()
            if doc_id == span.doc_id and loc.span.start < span.end and span.start < loc.span.end
        )


def load_corpus(root: Path = ROOT) -> Corpus:
    sources = json.loads((root / "data/sources.json").read_text(encoding="utf-8"))
    law_names, bodies, articles = {}, {}, {}
    for s in sources:
        stem = f"{s['doc_id']}_{s['effective_date'].replace('-', '')}"
        rows = [json.loads(line) for line in (root / "data/processed" / f"{stem}.jsonl").open(encoding="utf-8")]
        parts, pos = [], 0
        for row in rows:
            if row["deleted"]:
                continue
            start = pos
            paragraphs, cursor = {}, 0
            for p in row["paragraphs"]:
                found = row["text"].find(p["text"], cursor)
                if found < 0:
                    raise ValueError(f"{s['doc_id']} 제{row['article']}조 항 위치를 찾지 못함")
                paragraphs[p["no"]] = Span(s["doc_id"], start + found, start + found + len(p["text"]))
                cursor = found + len(p["text"])
            articles[(s["doc_id"], row["article"])] = ArticleLoc(
                s["doc_id"], row["article"], Span(s["doc_id"], start, start + len(row["text"])), paragraphs
            )
            parts.append(row["text"])
            pos += len(row["text"]) + 1
        law_names[s["doc_id"]] = s["law_name"]
        bodies[s["doc_id"]] = "\n".join(parts)
    return Corpus(law_names, bodies, articles)


def article_chunks(corpus: Corpus) -> list[Chunk]:
    chunks = []
    for (doc_id, number), loc in corpus.articles.items():
        chunks.append(Chunk(
            chunk_id=f"article:{doc_id}:{number}", strategy="article", doc_id=doc_id,
            law_name=corpus.law_names[doc_id], text=corpus.bodies[doc_id][loc.span.start:loc.span.end],
            span=loc.span, articles=(number,),
        ))
    return chunks


def fixed_chunks(corpus: Corpus, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    """법령별 본문을 size자 창으로 자르고, 다음 창은 size-overlap자 뒤에서 시작한다."""
    chunks = []
    step = size - overlap
    for doc_id, body in corpus.bodies.items():
        start, i = 0, 0
        while start < len(body):
            end = min(start + size, len(body))
            span = Span(doc_id, start, end)
            chunks.append(Chunk(
                chunk_id=f"fixed:{doc_id}:{i:03d}", strategy="fixed", doc_id=doc_id,
                law_name=corpus.law_names[doc_id], text=body[start:end], span=span,
                articles=corpus.articles_in(span),
            ))
            if end == len(body):
                break
            start, i = start + step, i + 1
    return chunks


def embedding_input(chunk: Chunk, with_law_name: bool) -> str:
    return f"{chunk.law_name}\n{chunk.text}" if with_law_name else chunk.text


def locate_quote(corpus: Corpus, doc_id: str, article: str, paragraph, quote: str) -> Span:
    """근거 구절이 지정한 조·항 안에서 차지하는 위치. 공백 차이는 무시한다."""
    loc = corpus.articles[(doc_id, article)]
    region = loc.paragraphs.get(paragraph, loc.span) if paragraph is not None else loc.span
    body = corpus.bodies[doc_id]
    # 공백을 뺀 문자열과 원래 위치의 대응표를 만들어 찾는다
    index = [i for i in range(region.start, region.end) if not body[i].isspace()]
    squeezed = "".join(body[i] for i in index)
    target = re.sub(r"\s+", "", quote)
    found = squeezed.find(target)
    if not target or found < 0:
        raise ValueError(f"구절을 찾지 못함: {doc_id} 제{article}조 {paragraph}항")
    return Span(doc_id, index[found], index[found + len(target) - 1] + 1)
