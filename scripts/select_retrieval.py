"""검색 설정 선택(설정 고정 전): 후보마다 개발 문항의 검색만 돌려 근거 충족을 센다. 생성 호출은 없다.

기본은 비용 추정만 한다. --execute를 붙여야 없는 질문 임베딩을 만든다(유료).
고르는 규칙은 eval/criteria.md "검색 설정 선택"에 있다. 확인용 문항은 받지 않는다.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chatbot import config, store
from rag_chatbot.corpus import load_corpus
from rag_chatbot.embedding import QueryCache, cost, count_tokens
from rag_chatbot.metrics import article_recall, evidence_coverage
from rag_chatbot.retrieval import make_search

QUESTIONS = ROOT / "eval" / "dev_questions.jsonl"
QUERY_CACHE = ROOT / "eval" / "runs" / "query_embeddings.json"
# 단순한 순서. 동률이면 앞의 것을 고른다.
CANDIDATES = [
    ("text-embedding-3-small", "vector"),
    ("text-embedding-3-large", "vector"),
    ("text-embedding-3-small", "hybrid"),
    ("text-embedding-3-large", "hybrid"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [q for q in questions if q["answerable"]]
    texts = [q["question"] for q in answerable]
    cache = QueryCache(QUERY_CACHE)
    conn = store.connect()

    models = sorted({m for m, _ in CANDIDATES})
    for m in models:
        missing = cache.missing(m, texts)
        indexed = {s: store.count(conn, m, s) for s in ("article", "fixed")}
        print(f"{m}: 색인 {indexed}, 질문 임베딩 새로 {len(missing)}개, 약 ${cost(count_tokens(missing), m):.5f}")
        if 0 in indexed.values():
            sys.exit(f"{m} 조각이 색인돼 있지 않다. scripts/index_chunks.py --model {m} --execute 먼저")
    if not args.execute:
        print("추정만 했다. 실제 실행은 --execute")
        return
    for m in models:
        cache.fill(m, texts)

    corpus = load_corpus()
    report = {"at": datetime.now().isoformat(), "questions": [q["id"] for q in answerable],
              "top_k": config.TOP_K, "hybrid_pool": config.HYBRID_POOL, "rrf_k": config.RRF_K, "candidates": []}
    print(f"\n답 있는 개발 문항 {len(answerable)}개 × 두 방식. 완전/일부/미충족, Recall@5 평균")
    for model, mode in CANDIDATES:
        cand = {"model": model, "retrieval": mode, "by_strategy": {}, "rows": []}
        for strategy in ("article", "fixed"):
            search = make_search(conn, model, strategy, mode, config.TOP_K)
            tally, recalls = Counter(), []
            for q in answerable:
                hits = search(q["question"], cache.get(model, q["question"]))
                cov = evidence_coverage(corpus, q, [h["span"] for h in hits])
                r5 = article_recall(q, [(h["doc_id"], h["articles"]) for h in hits], 5)
                tally[cov] += 1
                recalls.append(r5)
                cand["rows"].append({"id": q["id"], "strategy": strategy, "coverage": cov, "recall@5": r5,
                                     "retrieved": [h["chunk_id"] for h in hits]})
            cand["by_strategy"][strategy] = {"full": tally["full"], "partial": tally["partial"],
                                             "none": tally["none"], "recall@5": sum(recalls) / len(recalls)}
        s = cand["by_strategy"]
        cand["full_total"] = s["article"]["full"] + s["fixed"]["full"]
        cand["recall@5_mean"] = (s["article"]["recall@5"] + s["fixed"]["recall@5"]) / 2
        report["candidates"].append(cand)
        print(f"{model:24} {mode:6} | article {s['article']['full']}/{s['article']['partial']}/{s['article']['none']}"
              f" R@5 {s['article']['recall@5']:.2f} | fixed {s['fixed']['full']}/{s['fixed']['partial']}/"
              f"{s['fixed']['none']} R@5 {s['fixed']['recall@5']:.2f} | 완전 합 {cand['full_total']}")

    best = max(enumerate(report["candidates"]),
               key=lambda ic: (ic[1]["full_total"], round(ic[1]["recall@5_mean"], 6), -ic[0]))[1]
    report["chosen"] = {"model": best["model"], "retrieval": best["retrieval"]}
    print(f"\n규칙에 따른 선택: {best['model']} + {best['retrieval']}")
    out = ROOT / "eval" / "runs" / f"retrieval-select-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"기록: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
