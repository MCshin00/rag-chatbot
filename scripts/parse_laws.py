"""data/raw/의 법령 HTML을 조문 단위 JSONL(data/processed/)로 바꾼다."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chatbot.parsing import parse_law_html  # noqa: E402

from fetch_laws import raw_path  # noqa: E402


def main() -> None:
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    for source in sources:
        html = raw_path(source).read_text(encoding="utf-8")
        articles = parse_law_html(html, source)
        out = ROOT / "data" / "processed" / f"{raw_path(source).stem}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
        live = [a for a in articles if not a.deleted]
        print(f"{source['law_name']}: 조문 {len(articles)}개(삭제 {len(articles) - len(live)}개) → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
