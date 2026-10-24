"""두 방식의 조각을 만들어 임베딩하고 pgvector에 저장한다.

기본은 추정만 한다(유료 호출 없음). --execute를 붙여야 실제로 임베딩한다.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chatbot import store
from rag_chatbot.config import EMBED_WITH_LAW_NAME, EMBEDDING_MODEL, EMBEDDING_PRICES
from rag_chatbot.corpus import article_chunks, embedding_input, fixed_chunks, load_corpus
from rag_chatbot.embedding import cost, count_tokens, embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(EMBEDDING_PRICES), default=EMBEDDING_MODEL)
    ap.add_argument("--execute", action="store_true", help="실제로 임베딩한다(유료)")
    args = ap.parse_args()

    corpus = load_corpus()
    plans = {"article": article_chunks(corpus), "fixed": fixed_chunks(corpus)}
    total = 0
    for name, chunks in plans.items():
        texts = [embedding_input(c, EMBED_WITH_LAW_NAME) for c in chunks]
        tokens = count_tokens(texts)
        total += tokens
        print(f"{name}: 조각 {len(chunks)}개, 추정 {tokens:,} 토큰, 요청 1회")
    print(f"합계 추정 {total:,} 토큰 → 약 ${cost(total, args.model):.4f} ({args.model})")
    if not args.execute:
        print("추정만 했다. 실제 임베딩은 --execute")
        return

    conn = store.connect()
    store.init(conn)
    log = {"model": args.model, "with_law_name": EMBED_WITH_LAW_NAME,
           "started_at": datetime.now(UTC).isoformat(), "requests": []}
    for name, chunks in plans.items():
        texts = [embedding_input(c, EMBED_WITH_LAW_NAME) for c in chunks]
        vectors, used = embed(texts, args.model)
        store.replace_strategy(conn, args.model, name, list(zip(chunks, texts, vectors)))
        log["requests"].append({"strategy": name, "chunks": len(chunks), "tokens": used})
        print(f"{name}: 저장 {store.count(conn, args.model, name)}개, 실제 {used:,} 토큰")
    log["cost_usd"] = cost(sum(r["tokens"] for r in log["requests"]), args.model)
    out = ROOT / "eval" / "runs" / f"index-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"기록: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
