"""저장된 공식 HTML을 직접 읽어 정한 기대값으로 대표 조문을 검증한다.

data/processed의 결과를 정답으로 복사하거나 기대값 생성에 파서를 쓰지 않는다.
원문 다운로드나 모델 호출 없이 실행한다.
"""

import json
from pathlib import Path

import pytest

from rag_chatbot.parsing import parse_law_html

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def laws():
    sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    result = {}
    for source in sources:
        filename = f"{source['doc_id']}_{source['effective_date'].replace('-', '')}.html"
        html = (ROOT / "data/raw" / filename).read_text(encoding="utf-8")
        result[source["doc_id"]] = (html, parse_law_html(html, source))
    return result


def test_unnumbered_body_and_metadata(laws):
    """시행령 제1조: 항 번호 없는 본문, 출처 정보, 개정 이력을 보존한다."""
    article = next(a for a in laws["decree"][1] if a.article == "1")
    body = (
        "이 영은 「예금자보호법」에서 위임된 사항과 그 시행에 필요한 사항을 "
        "규정함을 목적으로 한다."
    )
    assert article.to_dict() == {
        "doc_id": "decree",
        "law_name": "예금자보호법 시행령",
        "effective_date": "2025-09-01",
        "article": "1",
        "title": "목적",
        "chapter": "",
        "deleted": False,
        "text": f"제1조(목적) {body}",
        "paragraphs": [{"no": None, "text": body}],
        "history": ["<개정 2007. 2. 28., 2016. 6. 21.>"],
    }


def test_numbered_paragraphs_items_and_history(laws):
    """시행령 제2조: 1~3호는 ①에 속하고, ②와 개정 표기는 섞이지 않는다."""
    article = next(a for a in laws["decree"][1] if a.article == "2")
    first = "\n".join([
        "① 「예금자보호법」(이하 “법”이라 한다) 제2조제1호사목에서 "
        "“대통령령으로 정하는 자”란 다음 각 호의 어느 하나에 해당하는 자를 말한다.",
        "1. 「자본시장과 금융투자업에 관한 법률」 제4조제3항에 따른 채무증권만을 "
        "대상으로 같은 법 제12조제1항제3호에 따른 전문투자자에 대해서만 "
        "투자매매업 또는 투자중개업의 인가를 받은 자",
        "2. 「자본시장과 금융투자업에 관한 법률」 제4조제1항에 따른 증권을 "
        "대상으로 같은 법 제12조제1항제3호에 따른 전문투자자에 대해서만 "
        "투자중개업의 인가를 받은 자(해당 증권의 환매조건부매매를 중개하는 경우만 해당한다)",
        "3. 「자본시장과 금융투자업에 관한 법률」 제117조의4에 따라 "
        "온라인소액투자중개업자의 등록을 한 자",
    ])
    second = (
        "②법 제2조제1호자목에서 “대통령령으로 정하는 보험회사”란 "
        "재보험을 주로 하는 주식회사인 보험회사를 말한다."
    )
    assert [(p.no, p.text) for p in article.paragraphs] == [(1, first), (2, second)]
    assert article.title == "부보금융회사에서 제외되는 금융회사"
    assert article.text == f"제2조({article.title}) {first}\n{second}"
    assert article.history == [
        "<신설 2016. 6. 21.>",
        "<개정 2007. 2. 28., 2009. 6. 9., 2016. 3. 11., 2016. 6. 21.>",
        "[전문개정 2001. 3. 17.] [제목개정 2016. 3. 11.]",
    ]


def test_chapter_and_section_are_both_preserved(laws):
    """법률 제3조: 직전의 절 제목이 장 제목을 덮어쓰지 않는다."""
    article = next(a for a in laws["act"][1] if a.article == "3")
    assert article.chapter == "제2장 예금보험공사 > 제1절 통칙"
    assert article.title == "설립"
    assert article.text == (
        "제3조(설립) 이 법에 따른 예금보험제도 등을 효율적으로 운영하기 위하여 "
        "예금보험공사를 설립한다."
    )
    assert article.history == ["[전문개정 2015. 12. 22.]"]


def test_addenda_do_not_enter_main_articles(laws):
    """시행령: 부칙의 제1조·제2조와 경과 규정이 본칙에 들어오지 않는다."""
    html, articles = laws["decree"]
    addendum_text = "이 영은 공포한 날부터 시행한다."
    addendum_title = "예금등의 범위에 관한 특례"
    assert addendum_text in html
    assert addendum_title in html
    assert [a.title for a in articles if a.article == "1"] == ["목적"]
    assert [a.title for a in articles if a.article == "2"] == [
        "부보금융회사에서 제외되는 금융회사"
    ]
    assert all(addendum_text not in a.text for a in articles)
    assert all(addendum_title != a.title for a in articles)
