"""국가법령정보센터 법령 본문 HTML을 조문 단위 레코드로 바꾼다.

본칙만 다룬다. 부칙(div#arDivArea 이후)은 넣지 않는다.
"""

import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup, Tag

ARTICLE_HEAD = re.compile(r"^제(\d+)조(?:의(\d+))?\s*(?:\((.*?)\))?")
# ①~⑳
PARAGRAPH_MARK = re.compile(r"^([①-⑳])")


@dataclass
class Paragraph:
    no: int | None
    text: str


@dataclass
class Article:
    doc_id: str
    law_name: str
    effective_date: str
    article: str  # "18", "24의4"
    title: str
    chapter: str
    deleted: bool
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_notes(p: Tag) -> tuple[str, list[str]]:
    """문단에서 개정 이력 표시(<개정 ...>, [전문개정 ...])를 떼어 본문과 이력으로 나눈다."""
    notes = []
    for span in p.select("span.sfon"):
        note = _clean(span.get_text(" "))
        if note:
            notes.append(note)
        span.decompose()
    for el in p.select("input, label"):
        # 조문 제목 label은 머리에서 따로 읽는다
        el.decompose()
    return _clean(p.get_text("")), notes


def _parse_article(lawcon: Tag, chapter: str, meta: dict) -> Article | None:
    label = lawcon.select_one("label")
    if label is None:
        return None
    head = ARTICLE_HEAD.match(_clean(label.get_text()))
    if head is None:
        return None
    number = head.group(1) + (f"의{head.group(2)}" if head.group(2) else "")
    title = head.group(3) or ""

    lines: list[str] = []
    history: list[str] = []
    for p in lawcon.find_all("p"):
        text, notes = _split_notes(p)
        history.extend(notes)
        if text:
            lines.append(text)

    deleted = not title and (not lines or lines[0].startswith("삭제"))

    paragraphs: list[Paragraph] = []
    for line in lines:
        mark = PARAGRAPH_MARK.match(line)
        if mark:
            paragraphs.append(Paragraph(no=ord(mark.group(1)) - 0x2460 + 1, text=line))
        elif paragraphs:
            # 호·목은 바로 앞 항에 붙인다
            paragraphs[-1].text += "\n" + line
        else:
            paragraphs.append(Paragraph(no=None, text=line))

    head_text = f"제{number}조" + (f"({title})" if title else "")
    body = "\n".join(lines)
    return Article(
        doc_id=meta["doc_id"],
        law_name=meta["law_name"],
        effective_date=meta["effective_date"],
        article=number,
        title=title,
        chapter=chapter,
        deleted=deleted,
        text=f"{head_text} {body}".strip(),
        paragraphs=paragraphs,
        history=history,
    )


def parse_law_html(html: str, meta: dict) -> list[Article]:
    soup = BeautifulSoup(html, "lxml")
    addenda = soup.select_one("#arDivArea")
    if addenda is not None:
        addenda.decompose()

    articles: list[Article] = []
    chapter, section = "", ""
    for el in soup.select("p.gtit, div.lawcon"):
        if el.name == "p":
            for span in el.select("span.sfon"):
                span.decompose()
            heading = _clean(el.get_text())
            if re.match(r"^제\d+절", heading):
                section = heading
            else:
                chapter, section = heading, ""
            continue
        path = " > ".join(h for h in (chapter, section) if h)
        article = _parse_article(el, path, meta)
        if article is not None:
            articles.append(article)
    return articles
