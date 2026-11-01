"""Evaluarea serviciului pe cele 65 de întrebări TechNova, cu praguri care pică CI-ul.

    python -m ragsvc.evaluate              # tabel + exit 1 dacă un prag nu e atins

Metrici:
- retrieval hit@k: o dovadă apare în fragmentele trimise;
- citare corectă: răspunsul citează fragmentul care conține dovada;
- answer recall: câte cuvinte de conținut din referință apar în răspuns;
- refuz corect / greșit: la întrebările fără răspuns / la cele cu răspuns.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from ragsvc.answer import Answerer, ExtractiveAnswerer, answer
from ragsvc.index import HybridIndex
from ragsvc.text import tokens

ROOT = Path(__file__).resolve().parent.parent
# podele puțin sub valorile actuale: CI-ul pică la regresii, nu la zgomot
GATES = {"hit_at_k": 0.90, "citation_ok": 0.70, "answer_recall": 0.58, "abstain_ok": 0.40, "false_abstain_max": 0.20}


def load_index(kb: Path = ROOT / "kb") -> HybridIndex:
    idx = HybridIndex()
    for p in sorted(kb.glob("*.md")):
        idx.upsert(p.stem, p.read_text(encoding="utf-8"))
    return idx


def evaluate(idx: HybridIndex, answerer: Answerer, k: int = 3,
             questions: Path = ROOT / "data" / "questions.jsonl") -> dict[str, float]:
    rows = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    hit, cite, recall, false_abstain, abstain_ok = [], [], [], [], []
    for q in rows:
        sources = [c for c, _, _ in idx.search(q["question"], k=k)]
        result = answer(answerer, q["question"], sources)
        if not q["evidence"]:
            abstain_ok.append(float(result.abstained))
            continue
        has = lambda chunks: any(e in c.text for e in q["evidence"] for c in chunks)  # noqa: E731
        hit.append(float(has(sources)))
        false_abstain.append(float(result.abstained))
        cite.append(float(not result.abstained and has(result.citations)))
        ref = set(tokens(q["answer"]))
        recall.append(0.0 if result.abstained else len(ref & set(tokens(result.text))) / max(1, len(ref)))
    return {"questions": len(rows), "hit_at_k": statistics.mean(hit), "citation_ok": statistics.mean(cite),
            "answer_recall": statistics.mean(recall), "false_abstain": statistics.mean(false_abstain),
            "abstain_ok": statistics.mean(abstain_ok)}


def check(metrics: dict[str, float]) -> list[str]:
    fails = [f"{m} = {metrics[m]:.3f} < {v}" for m, v in GATES.items() if not m.endswith("_max") and metrics[m] < v]
    if metrics["false_abstain"] > GATES["false_abstain_max"]:
        fails.append(f"false_abstain = {metrics['false_abstain']:.3f} > {GATES['false_abstain_max']}")
    return fails


def main() -> int:
    m = evaluate(load_index(), ExtractiveAnswerer())
    print("| metrică | valoare | prag |\n|---|---:|---:|")
    for key, label in [("hit_at_k", "retrieval hit@3"), ("citation_ok", "citare corectă"),
                       ("answer_recall", "answer recall"), ("false_abstain", "refuz greșit (cu răspuns)"),
                       ("abstain_ok", "refuz corect (fără răspuns)")]:
        gate = GATES.get(key) or GATES.get(f"{key}_max")
        sign = "≥" if key in GATES else ("≤" if f"{key}_max" in GATES else "")
        print(f"| {label} | {m[key]:.3f} | {sign + ' ' + str(gate) if gate else '—'} |")
    fails = check(m)
    print("\n" + ("✅ toate pragurile sunt atinse" if not fails else "❌ " + "; ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
