"""두 방식의 실행 결과로 판정표를 만든다. 방식 이름은 가리고 A/B로 보여 준다.

판정표에는 질문, 정답 기준, 답변, 자동 지표만 넣는다. 검색된 조각 본문은 넣지 않는다
(조각 모양으로 방식이 드러나기 때문이다). A/B와 방식의 대응은 blind_key.json에 따로 둔다.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COV = {"full": "완전 충족", "partial": "일부 충족", "none": "미충족", "na": "해당 없음", None: "-"}


def load(run_dir: Path) -> tuple[dict, dict]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in map(json.loads, (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines())}
    return meta, rows


def cell(text: str | None) -> str:
    return (text or "(없음)").replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = [load(ROOT / d) for d in args.run_dirs]
    strategies = [m["strategy"] for m, _ in runs]
    assert sorted(strategies) == ["article", "fixed"], strategies
    qfile = runs[0][0]["questions_file"]
    assert all(m["questions_file"] == qfile for m, _ in runs)
    questions = [json.loads(line) for line in (ROOT / qfile).read_text(encoding="utf-8").splitlines() if line.strip()]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    key, lines = {}, [
        f"# 판정표: {Path(qfile).name}",
        "",
        f"실행: {', '.join(args.run_dirs)}. 방식 이름은 가렸다(대응은 `{out.stem}-key.json`).",
        ("판정: 맞음 / 부분 / 틀림 / 잘못된 보류 / 적절한 보류 ([criteria.md](../criteria.md)). "
         "근거 없는 단정이면 판정 옆에 `단정`을 붙인다. 확정 칸은 사용자가 채운다."),
        "",
    ]
    for q in questions:
        if q["id"] not in runs[0][1]:
            continue
        flip = int(hashlib.sha256(q["id"].encode()).hexdigest(), 16) % 2 == 1
        order = [1, 0] if flip else [0, 1]
        key[q["id"]] = {"A": strategies[order[0]], "B": strategies[order[1]]}
        lines += [f"## {q['id']} · {q['type']}", "", f"**질문** {q['question']}", ""]
        lines.append(f"**꼭 말해야 할 것** {'; '.join(q['must_include']) or '(답 없음 문항)'}  ")
        lines.append(f"**하면 틀리는 것** {'; '.join(q['must_not']) or '-'}  ")
        if q["evidence_sets"]:
            for i, s in enumerate(q["evidence_sets"], 1):
                quotes = " / ".join(f"{e['doc_id']} 제{e['article']}조{'' if e['paragraph'] is None else ' ' + str(e['paragraph']) + '항'}: “{e['quote']}”" for e in s)
                lines.append(f"**근거 묶음 {i}** {quotes}  ")
        if q["notes"]:
            lines.append(f"**메모** {q['notes']}  ")
        lines += ["", "| | 상태 | 답변 | 근거 충족 | 인용 적합성 |", "|---|---|---|---|---|"]
        for label, idx in zip("AB", order):
            r = runs[idx][1][q["id"]]
            final, m = r["final"], r["metrics"]
            status = final["status"] + (f" ({m['abstain_cause']})" if m["abstain_cause"] else "")
            answer = final["answer"] if final["status"] != "error" else f"실행 오류: {r['raw'].get('error')}"
            cites = f" [{', '.join(final['citations'])}]" if final["citations"] else ""
            lines.append(f"| {label} | {status} | {cell(answer)}{cites} | {COV[m['coverage']]} | {COV[m['citation_coverage']]} |")
        lines += ["", "| | Claude 초안 | 이유 | Codex 재검토 | 이유 | 확정 |", "|---|---|---|---|---|---|",
                  "| A | | | | | |", "| B | | | | | |", ""]
    with out.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with out.with_name(f"{out.stem}-key.json").open("x", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    print(f"판정표: {out.relative_to(ROOT)} ({len(key)}문항)")


if __name__ == "__main__":
    main()
