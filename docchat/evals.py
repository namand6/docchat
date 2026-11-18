"""Measure the system instead of eyeballing it.

Eval file (JSONL), one question per line:
  {"q": "...", "pages": [3], "must": ["14 days"]}      answerable: expected source page(s), key facts
  {"q": "...", "pages": []}                            unanswerable: the right answer is to abstain

Retrieval: recall@k and MRR for each retriever, so the hybrid choice is backed by numbers.
Answers:   fact accuracy, citation precision (cited a correct page), groundedness, and abstention
           on unanswerable questions (the hallucination test).
"""

from __future__ import annotations

import json
from pathlib import Path

from .answer import ask
from .index import Index


def load(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def retrieval(index: Index, cases: list[dict], k: int = 5) -> dict:
    out = {}
    answerable = [c for c in cases if c.get("pages")]
    for mode in [m for m in Index.MODES if m != "embed" or index.embed]:
        hit1, hit, rr = 0, 0, 0.0
        for c in answerable:
            pages = [ch.page for ch, _ in index.search(c["q"], k=k, mode=mode)]
            first = next((i for i, p in enumerate(pages) if p in c["pages"]), None)
            if first is not None:
                hit += 1
                hit1 += first == 0
                rr += 1 / (first + 1)
        n = max(len(answerable), 1)
        out[mode] = {"recall@1": round(hit1 / n, 3), f"recall@{k}": round(hit / n, 3), "mrr": round(rr / n, 3)}
    return out


def answers(index: Index, cases: list[dict], llm, k: int = 5) -> dict:
    rows, acc, cite_ok, grounded, abst_ok, n_ans, n_un = [], 0, 0, 0, 0, 0, 0
    for c in cases:
        a = ask(index, c["q"], llm, k=k)
        if c.get("pages"):
            n_ans += 1
            facts = all(m.lower() in a.text.lower() for m in c.get("must", []))
            cites = any(s["page"] in c["pages"] for s in a.sources)
            acc += facts and not a.abstained
            cite_ok += cites
            grounded += a.grounded and not a.abstained
            rows.append({"q": c["q"], "ok": bool(facts and cites and not a.abstained), "answer": a.text, "issues": a.issues})
        else:
            n_un += 1
            abst_ok += a.abstained
            rows.append({"q": c["q"], "ok": a.abstained, "answer": a.text, "issues": [] if a.abstained else ["should have abstained"]})
    pct = lambda x, n: round(x / n, 3) if n else None
    return {"fact_accuracy": pct(acc, n_ans), "citation_precision": pct(cite_ok, n_ans), "grounded": pct(grounded, n_ans),
            "abstention": pct(abst_ok, n_un), "cases": rows}
