"""평가 문항 파일의 형식과 근거 구절을 검사한다.

근거 구절은 두 곳에서 확인한다.
- 파싱 결과(data/processed)의 해당 조·항 본문
- 원문 HTML(data/raw)의 해당 조문 영역. 파서를 거치지 않고 따로 읽는다

출력에는 문항 ID와 오류 종류만 쓴다. 문항 내용은 출력하지 않는다.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
TYPES = {
    "한도·합산": (4, 1),
    "보호 대상": (3, 2),
    "조건·계산": (3, 2),
    "복수 근거": (4, 1),
    "절차·기관": (2, 2),
    "답 없음(범위 밖)": (2, 1),
    "답 없음(정보 부족)": (2, 1),
}
FIELDS = ["id", "group", "type", "situation", "question", "answerable",
          "evidence_sets", "must_include", "must_not", "notes"]
RAW_HEAD = re.compile(r"제(\d+)조(?:의(\d+))?")


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_parsed() -> dict:
    sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    articles = {}
    for s in sources:
        stem = f"{s['doc_id']}_{s['effective_date'].replace('-', '')}"
        for line in (ROOT / "data/processed" / f"{stem}.jsonl").open(encoding="utf-8"):
            a = json.loads(line)
            articles[(a["doc_id"], a["article"])] = a
    return articles


def load_raw() -> dict:
    """원문 HTML에서 본칙 조문 영역의 글자를 조 번호별로 모은다."""
    sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    raw = {}
    for s in sources:
        stem = f"{s['doc_id']}_{s['effective_date'].replace('-', '')}"
        soup = BeautifulSoup((ROOT / "data/raw" / f"{stem}.html").read_text(encoding="utf-8"), "lxml")
        addenda = soup.select_one("#arDivArea")
        if addenda is not None:
            addenda.decompose()
        for block in soup.select("div.lawcon"):
            label = block.select_one("label")
            m = RAW_HEAD.match(norm(label.get_text())) if label else None
            if m:
                number = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
                raw[(s["doc_id"], number)] = norm(block.get_text(""))
    return raw


def check(path: Path) -> list[str]:
    parsed, raw = load_parsed(), load_raw()
    errors, ids, types = [], set(), Counter()
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    split = None
    for n, line in enumerate(lines, 1):
        try:
            q = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"{n}번째 줄: JSON 형식 오류")
            continue
        qid = q.get("id", f"{n}번째 줄")
        missing = [f for f in FIELDS if f not in q]
        if missing:
            errors.append(f"{qid}: 필드 없음 {missing}")
            continue
        if not re.fullmatch(r"[DH]\d{2}", qid):
            errors.append(f"{qid}: ID 형식")
        split = split or qid[0]
        if qid[0] != split:
            errors.append(f"{qid}: 한 파일에 개발·확인 ID가 섞임")
        if qid in ids:
            errors.append(f"{qid}: ID 중복")
        ids.add(qid)
        if q["type"] not in TYPES:
            errors.append(f"{qid}: 알 수 없는 유형")
        types[q["type"]] += 1
        unanswerable = q["type"].startswith("답 없음")
        if q["answerable"] == unanswerable:
            errors.append(f"{qid}: answerable 값과 유형이 맞지 않음")
        if unanswerable and q["evidence_sets"]:
            errors.append(f"{qid}: 답 없음 문항에 근거 묶음이 있음")
        if not unanswerable and not q["evidence_sets"]:
            errors.append(f"{qid}: 근거 묶음 없음")
        if not q["must_include"] and not unanswerable:
            errors.append(f"{qid}: must_include 없음")
        for si, ev_set in enumerate(q["evidence_sets"], 1):
            if not ev_set:
                errors.append(f"{qid}: 근거 묶음 {si}이 비어 있음")
            for ei, ev in enumerate(ev_set, 1):
                where = f"{qid}: 묶음 {si} 구절 {ei}"
                key = (ev.get("doc_id"), str(ev.get("article")))
                article = parsed.get(key)
                if article is None:
                    errors.append(f"{where}: 없는 조문")
                    continue
                if article["deleted"]:
                    errors.append(f"{where}: 삭제된 조문")
                quote = norm(ev.get("quote", ""))
                if len(quote) < 4:
                    errors.append(f"{where}: 구절이 너무 짧음")
                    continue
                para = ev.get("paragraph")
                if para is None:
                    body = article["text"]
                else:
                    match = [p for p in article["paragraphs"] if p["no"] == para]
                    if not match:
                        errors.append(f"{where}: 없는 항")
                        continue
                    body = match[0]["text"]
                if quote not in norm(body):
                    errors.append(f"{where}: 파싱 결과의 해당 조·항에 구절이 없음")
                if quote not in raw.get(key, ""):
                    errors.append(f"{where}: 원문 HTML의 해당 조문에 구절이 없음")
    if split:
        col = 0 if split == "D" else 1
        for t, quota in TYPES.items():
            if types[t] != quota[col]:
                errors.append(f"유형 개수: {t} {types[t]}개 (기준 {quota[col]}개)")
    return errors


def main() -> None:
    path = Path(sys.argv[1])
    errors = check(path)
    if errors:
        print(f"{path.name}: 오류 {len(errors)}건")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"{path.name}: 통과")


if __name__ == "__main__":
    main()
