"""Grounded answers: the model sees numbered excerpts, must cite them like [2], and must say it
doesn't know when the excerpts don't contain the answer. Citations are checked in code."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Index

NOT_FOUND = "I couldn't find that in the document."


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)   # the excerpts actually cited
    retrieved: list[dict] = field(default_factory=list)
    grounded: bool = True
    abstained: bool = False
    issues: list[str] = field(default_factory=list)


def build_prompt(question: str, hits) -> str:
    excerpts = "\n\n".join(f"[{n}] (page {c.page}{', ' + c.heading if c.heading else ''})\n{c.text}"
                           for n, (c, _) in enumerate(hits, 1))
    return f"""Answer the question using only the excerpts below.
- Cite every claim with the excerpt number in square brackets, like [1] or [2][3].
- If the excerpts don't contain the answer, reply exactly: {NOT_FOUND}
- Be concise: at most 4 sentences. Do not use outside knowledge.

Excerpts:
{excerpts}

Question: {question}"""


def check(text: str, n_hits: int) -> tuple[list[int], list[str], bool]:
    """Returns (cited numbers, issues, abstained)."""
    if NOT_FOUND.lower().rstrip(".") in text.lower():
        return [], [], True
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)})
    issues = []
    bad = [c for c in cited if not 1 <= c <= n_hits]
    if bad:
        issues.append(f"cites excerpt(s) that don't exist: {bad}")
    if not cited:
        issues.append("no citations")
    uncited = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if len(s.split()) > 4 and not re.search(r"\[\d+\]", s)]
    if uncited:
        issues.append(f"{len(uncited)} sentence(s) without a citation")
    return [c for c in cited if 1 <= c <= n_hits], issues, False


def ask(index: Index, question: str, llm, k: int = 5, mode: str = "hybrid") -> Answer:
    hits = index.search(question, k=k, mode=mode)
    retrieved = [{"n": n, "page": c.page, "heading": c.heading, "text": c.text} for n, (c, _) in enumerate(hits, 1)]
    if not hits:
        return Answer(NOT_FOUND, retrieved=retrieved, abstained=True)
    text = llm.complete(build_prompt(question, hits)).strip()
    cited, issues, abstained = check(text, len(hits))
    return Answer(text, [retrieved[c - 1] for c in cited], retrieved, grounded=not issues, abstained=abstained, issues=issues)
