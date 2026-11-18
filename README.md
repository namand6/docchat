# docchat

Chat with a PDF and get answers that cite the page they came from, or an honest "I couldn't find
that" when the document doesn't say. Built to be measured, not eyeballed: it ships with an eval
harness that scores retrieval and answers, including whether it hallucinates on unanswerable questions.

```
PDF ─► pages ─► sentence-packed chunks (page + section kept) ─► BM25  ┐
                                                              ─► bge-small embeddings ┴► RRF ─► top-k
top-k ─► numbered excerpts ─► Claude (cite [n] or abstain) ─► citation check in code ─► answer + sources
```

## Results (sample handbook, 20 questions, 6 of them unanswerable)

Retrieval, k = 3:

| Retriever | recall@1 | recall@3 | MRR |
|---|---|---|---|
| BM25 (from scratch) | 0.80 | 0.90 | 0.84 |
| LSA (TF-IDF + SVD, from scratch) | 0.80 | 0.90 | 0.83 |
| Embeddings (bge-small, local ONNX) | **0.95** | **1.00** | **0.97** |
| Hybrid (BM25 + embeddings, RRF) | 0.90 | 0.95 | 0.93 |

Answers (Claude Sonnet, k = 4):

| Metric | Score |
|---|---|
| Fact accuracy (key fact present) | 1.00 |
| Citation precision (cites a correct page) | 1.00 |
| Abstention on unanswerable questions | 1.00 |
| Every sentence cited | 0.85 |

What the numbers say:
- Most eval questions are paraphrases ("when a parent dies" vs "bereavement leave"), where embeddings
  beat keyword search. Hybrid still wins on exact identifiers (system names, codes, policy numbers),
  so it stays the default; `--mode embed` is one flag away for prose-heavy documents.
- The citation checker caught the model adding an uncited inference ("a week off would exceed 3 days,
  so…"). That's the kind of sentence a reviewer should see flagged.
- The sample is small and synthetic; the harness is the point. Point it at any PDF and JSONL of questions.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install pypdf numpy fastembed reportlab pytest
.venv/bin/python -m docchat sample                        # demo PDF + eval set
.venv/bin/python -m docchat ingest data/northwind_handbook.pdf
.venv/bin/python -m docchat ask "Can I carry over unused vacation?"
.venv/bin/python -m docchat eval data/northwind_evals.jsonl --k 3 [--answers]
.venv/bin/python -m docchat serve                         # http://127.0.0.1:8796
.venv/bin/python -m pytest -q
```

Answers use the Claude Code CLI headless (no API key stored in the repo). Without fastembed installed,
hybrid falls back to BM25 + LSA, so everything still runs with numpy alone.

## Design notes

- **Chunking:** whole sentences packed to ~70 words with ~15 words of overlap, never crossing a page,
  so each citation maps to exactly one page. Section headings are detected and attached to chunks.
- **Prompt contract:** numbered excerpts only, cite every claim, exact abstention string, no outside
  knowledge. **Enforced in code:** citations must exist and be in range; uncited sentences are flagged.
- **Eval format:** `{"q": ..., "pages": [...], "must": [...]}`; `pages: []` marks an unanswerable question.
