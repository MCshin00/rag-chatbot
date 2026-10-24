"""평가 실행: 문항마다 검색 → 생성 → 인용 검사를 돌리고 결과를 저장한다.

기본은 비용 추정만 한다. --execute를 붙여야 유료 호출을 한다.
결과 폴더는 새로 만들기만 하고 덮어쓰지 않는다. 문항 내용은 화면에 출력하지 않는다.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chatbot import config, store
from rag_chatbot.corpus import load_corpus
from rag_chatbot.embedding import EMBEDDING_PRICE_PER_1M, count_tokens, embed
from rag_chatbot.graph import SYSTEM_PROMPT_P1, build_graph, format_context, make_chat_model
from rag_chatbot.metrics import article_recall, evidence_coverage

CHAT_PRICE = {"input": 0.75, "cached_input": 0.075, "output": 4.50}  # $/1M, 2026-10-24 공식 모델 페이지
PRICE_CHECKED = "2026-10-24"
QUERY_CACHE = ROOT / "eval" / "runs" / "query_embeddings.json"


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "eval/*.md", "eval/*.jsonl"],
                           cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": head, "uncommitted_changes": bool(dirty)}


def chat_cost(usage: dict | None) -> float | None:
    if not usage:
        return None
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
    return ((usage["input_tokens"] - cached) * CHAT_PRICE["input"] + cached * CHAT_PRICE["cached_input"]
            + usage["output_tokens"] * CHAT_PRICE["output"]) / 1e6


def abstain_cause(final: dict, coverage: str) -> str | None:
    if final["status"] == "error":
        return "실행 오류"
    if final["status"] != "abstain":
        return None
    if final.get("reason") in ("citation_check",):
        return "검사 차단"
    return "생성 보류" if coverage == "full" else "검색 미충족"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--strategy", choices=["article", "fixed"], required=True)
    ap.add_argument("--ids", help="쉼표로 구분한 문항 ID(동작 확인용)")
    ap.add_argument("--tag", required=True, help="smoke, formal 등")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    qpath = ROOT / args.questions
    questions = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = args.ids.split(",")
        questions = [q for q in questions if q["id"] in wanted]
    corpus = load_corpus()

    cache = json.loads(QUERY_CACHE.read_text(encoding="utf-8")) if QUERY_CACHE.exists() else {}
    def qkey(q):
        return hashlib.sha256(f"{config.EMBEDDING_MODEL}|{q['question']}".encode()).hexdigest()
    missing = [q for q in questions if qkey(q) not in cache]

    conn = store.connect()
    n_chunks = store.count(conn, args.strategy)
    if n_chunks == 0:
        sys.exit(f"{args.strategy} 조각이 색인돼 있지 않다. scripts/index_chunks.py --execute 먼저")
    avg = conn.execute("SELECT avg(length(text)) FROM chunks WHERE strategy=%s", (args.strategy,)).fetchone()[0]
    sample_ctx = "가" * int(float(avg)) * config.TOP_K
    in_tokens = count_tokens([SYSTEM_PROMPT_P1, sample_ctx]) + 200
    upper = len(questions) * (in_tokens * CHAT_PRICE["input"] + config.MAX_OUTPUT_TOKENS * CHAT_PRICE["output"]) / 1e6
    q_tokens = count_tokens([q["question"] for q in missing]) if missing else 0
    print(f"문항 {len(questions)}개, 방식 {args.strategy}, 모델 {config.CHAT_MODEL} (effort {config.REASONING_EFFORT})")
    print(f"질문 임베딩: 새로 {len(missing)}개, 요청 {1 if missing else 0}회, 약 ${q_tokens * EMBEDDING_PRICE_PER_1M / 1e6:.5f}")
    print(f"생성: {len(questions)}회, 1회 입력 약 {in_tokens:,} 토큰(추정), 출력 최대 {config.MAX_OUTPUT_TOKENS}")
    print(f"생성 비용 상한(출력이 모두 최대일 때): 약 ${upper:.3f}")
    if not args.execute:
        print("추정만 했다. 실제 실행은 --execute")
        return

    if missing:
        vecs, used = embed([q["question"] for q in missing])
        for q, v in zip(missing, vecs):
            cache[qkey(q)] = v
        QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        QUERY_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"질문 임베딩 {len(missing)}개, 실제 {used} 토큰")
    else:
        used = 0

    split = "dev" if questions and questions[0]["id"].startswith("D") else "heldout"
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.tag}-{split}-{args.strategy}"
    run_dir = ROOT / "eval" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "run_id": run_id, "strategy": args.strategy, "questions_file": args.questions,
        "questions_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
        "question_ids": [q["id"] for q in questions],
        "chat_model": config.CHAT_MODEL, "reasoning_effort": config.REASONING_EFFORT,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS, "top_k": config.TOP_K,
        "embedding_model": config.EMBEDDING_MODEL, "embed_with_law_name": config.EMBED_WITH_LAW_NAME,
        "prompt_version": config.PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT_P1,
        "retries": 0, "prices_usd_per_1m": CHAT_PRICE, "price_checked": PRICE_CHECKED,
        "query_embedding_tokens": used, "git": git_state(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "meta.json").open("x", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))

    graph = build_graph(lambda vec: store.search(conn, args.strategy, vec, config.TOP_K), make_chat_model())
    tally, cost, calls = Counter(), 0.0, 0
    with (run_dir / "results.jsonl").open("x", encoding="utf-8") as out:
        for q in questions:
            state = graph.invoke({"question": q["question"], "query_vec": cache[qkey(q)]})
            calls += 0 if state["raw"].get("skipped") else 1
            retrieved = state["retrieved"]
            spans = [r["span"] for r in retrieved]
            coverage = evidence_coverage(corpus, q, spans)
            cited = [retrieved[int(c[1:]) - 1]["span"] for c in state["final"]["citations"]]
            citation_cov = evidence_coverage(corpus, q, cited) if state["final"]["status"] == "answer" else None
            ranked = [(r["doc_id"], r["articles"]) for r in retrieved]
            usage = state["raw"]["usage"]
            c = chat_cost(usage)
            cost += c or 0
            row = {
                "id": q["id"], "strategy": args.strategy,
                "retrieved": [{"rank": i, "chunk_id": r["chunk_id"], "distance": r["distance"],
                               "articles": list(r["articles"]), "span": [r["span"].start, r["span"].end]}
                              for i, r in enumerate(retrieved, 1)],
                "context": format_context(retrieved),
                "raw": {k: v for k, v in state["raw"].items() if k != "usage"},
                "usage": usage, "cost_usd": c,
                "final": state["final"],
                "metrics": {
                    "coverage": coverage,
                    "recall@1": article_recall(q, ranked, 1), "recall@3": article_recall(q, ranked, 3),
                    "recall@5": article_recall(q, ranked, 5),
                    "citation_coverage": citation_cov,
                    "abstain_cause": abstain_cause(state["final"], coverage),
                },
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            tally[state["final"]["status"]] += 1
            tally[f"coverage:{coverage}"] += 1
            print(f"{q['id']}: {state['final']['status']}")
    summary = {"calls": calls, "status": dict(tally), "chat_cost_usd": cost,
               "finished_at": datetime.now(UTC).isoformat()}
    (run_dir / "summary.json").open("x", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"끝: {run_dir.relative_to(ROOT)}  생성 호출 {calls}회, 비용 ${cost:.4f}")


if __name__ == "__main__":
    main()
