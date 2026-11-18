"""docchat: ask questions about a PDF and get answers with page citations.

  python -m docchat ingest handbook.pdf [more.pdf]   build the index (data/index.json)
  python -m docchat ask "How much PTO do I get?"     answer with citations
  python -m docchat eval evals.jsonl [--answers]     retrieval metrics (+ answer metrics, uses Claude)
  python -m docchat serve                            web page at http://127.0.0.1:8796
  python -m docchat sample                           make the demo handbook PDF + eval set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evals, ingest, sample
from .answer import ask
from .index import Index
from .llm import ClaudeCLI

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="docchat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ingest", "ask", "eval", "serve", "sample"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", choices=Index.MODES)
    ap.add_argument("--answers", action="store_true", help="eval: also grade answers (calls the model)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--port", type=int, default=8796)
    a = ap.parse_args(argv)
    data = Path(a.data)
    idx_path = data / "index.json"

    if a.command == "sample":
        pdf = sample.write_pdf(data / "northwind_handbook.pdf")
        ev = sample.write_evals(data / "northwind_evals.jsonl")
        print(f"wrote {pdf} and {ev}")
        return 0
    if a.command == "ingest":
        chunks = []
        for f in a.args:
            chunks += ingest.chunk_pages(ingest.read_pdf(f), Path(f).name, start_id=len(chunks))
        data.mkdir(parents=True, exist_ok=True)
        Index(chunks).save(idx_path)
        print(f"indexed {len(chunks)} chunks from {len(a.args)} file(s)")
        return 0
    if not idx_path.exists():
        print("no index yet: run `python -m docchat ingest <file.pdf>` first")
        return 1
    index = Index.load(idx_path)
    if a.command == "ask":
        ans = ask(index, " ".join(a.args), ClaudeCLI(a.model), k=a.k, mode=a.mode)
        print(ans.text)
        for s in ans.sources:
            print(f"  [{s['n']}] page {s['page']}: {s['text'][:100]}…")
        if ans.issues:
            print("  ⚠ " + "; ".join(ans.issues))
        return 0
    if a.command == "eval":
        cases = evals.load(a.args[0])
        print("Retrieval:", json.dumps(evals.retrieval(index, cases, a.k), indent=1))
        if a.answers:
            r = evals.answers(index, cases, ClaudeCLI(a.model), a.k)
            print("Answers:", json.dumps({k: v for k, v in r.items() if k != "cases"}, indent=1))
            for c in r["cases"]:
                if not c["ok"]:
                    print(f"  ✗ {c['q']}\n    {c['answer'][:160]}  {c['issues']}")
        return 0
    if a.command == "serve":
        from .web import serve
        serve(index, ClaudeCLI(a.model), a.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
