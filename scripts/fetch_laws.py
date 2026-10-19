"""data/sources.json에 적힌 법령 본문 HTML을 국가법령정보센터에서 받아 data/raw/에 저장한다."""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODY_URL = (
    "https://www.law.go.kr/LSW/lsInfoR.do?lsiSeq={lsi_seq}&efYd={ef_yd}"
    "&efYn=Y&chrClsCd=010202&nwJoYnInfo=Y&ancYnChk=0&netPrivateYn=N"
)


def raw_path(source: dict) -> Path:
    return ROOT / "data" / "raw" / f"{source['doc_id']}_{source['effective_date'].replace('-', '')}.html"


def body_url(source: dict) -> str:
    return BODY_URL.format(lsi_seq=source["lsi_seq"], ef_yd=source["effective_date"].replace("-", ""))


def main() -> None:
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    for source in sources:
        request = urllib.request.Request(body_url(source), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
        path = raw_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        print(f"{source['law_name']} → {path.relative_to(ROOT)} ({len(html):,}자)")
        time.sleep(1)


if __name__ == "__main__":
    main()
