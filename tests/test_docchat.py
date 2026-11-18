import json
from pathlib import Path

import pytest

from docchat import evals, ingest, sample
from docchat.answer import NOT_FOUND, ask, check
from docchat.index import BM25, Index, rrf, tokens
from docchat.llm import FakeLLM


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    return sample.write_pdf(tmp_path_factory.mktemp("d") / "h.pdf")


@pytest.fixture(scope="module")
def index(pdf):
    return Index(ingest.chunk_pages(ingest.read_pdf(pdf), "h.pdf"), use_embed=False)


def test_chunks_keep_pages_headings_and_overlap():
    pages = ["1 Intro\n" + " ".join(f"Sentence number {i} is here." for i in range(40)), "2 Next\nShort page."]
    ch = ingest.chunk_pages(pages, "d", size=30, overlap=8)
    assert {c.page for c in ch} == {1, 2}
    assert ch[0].heading == "1 Intro" and ch[-1].heading == "2 Next"
    assert all(len(c.text.split()) <= 30 + 5 for c in ch)
    first_end = ch[0].text.split(". ")[-1]
    assert first_end.rstrip(".") in ch[1].text  # overlap carries context forward


def test_clean_rejoins_hyphenation():
    assert ingest.clean("reim-\nbursable  costs") == "reimbursable costs"


def test_tokens_stem_and_drop_stopwords():
    assert tokens("The requests were requested") == ["request", "request"]


def test_bm25_prefers_rare_terms():
    bm = BM25([["laptop", "refresh"], ["laptop", "laptop", "stolen"], ["meal", "allowance"]])
    s = bm.scores(["stolen", "laptop"])
    assert s.argmax() == 1 and s[2] == 0


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [2, 1, 3]])[:2] in ([1, 2], [2, 1])
    assert rrf([[5, 1], [1, 6]])[0] == 1


def test_retrieval_finds_right_pages(index):
    for mode in ("bm25", "lsa", "hybrid"):
        pages = [c.page for c, _ in index.search("meal allowance during business travel", k=2, mode=mode)]
        assert 6 in pages, mode
    with pytest.raises(ValueError):
        index.search("x y", mode="nope")


def test_retrieval_metrics_shape(index):
    r = evals.retrieval(index, [json.loads(json.dumps(e)) for e in sample.EVALS], k=3)
    assert set(r) == {"bm25", "lsa", "hybrid"}
    assert r["bm25"]["recall@3"] >= 0.8 and r["bm25"]["recall@1"] <= r["bm25"]["recall@3"]


def test_check_citations():
    assert check("You get 20 days. [1] Five carry over [2].", 3) == ([1, 2], [], False)
    cited, issues, _ = check("You get 20 days [7].", 3)
    assert cited == [] and "don't exist" in issues[0]
    assert "no citations" in check("You get twenty days of leave per year.", 3)[1]
    assert check(NOT_FOUND, 3)[2] is True


def test_ask_grounded_and_abstains(index):
    good = FakeLLM(lambda p: "You get 75 dollars a day for meals [1].")
    a = ask(index, "daily meal allowance when traveling", good, k=3)
    assert a.grounded and a.sources and a.sources[0]["n"] == 1
    assert "75 dollars" in good.prompts[0] and "Do not use outside knowledge" in good.prompts[0]
    a = ask(index, "what is the 401k match", FakeLLM(lambda p: NOT_FOUND), k=3)
    assert a.abstained and a.sources == []
    a = ask(index, "the", FakeLLM(lambda p: "x"))   # only stopwords: nothing to search
    assert a.abstained and a.retrieved == []


def test_answer_eval_scores_hallucination(index):
    cases = [{"q": "daily meal allowance", "pages": [6], "must": ["75"]}, {"q": "401k match?", "pages": []}]
    llm = FakeLLM(lambda p: "The match is 4 percent [1]." if "401k" in p else "It is 75 dollars [1].")
    r = evals.answers(index, cases, llm, k=3)
    assert r["fact_accuracy"] == 1.0 and r["abstention"] == 0.0
    assert r["cases"][1]["issues"] == ["should have abstained"]


def test_index_roundtrip(tmp_path, index):
    index.save(tmp_path / "i.json")
    again = Index.load(tmp_path / "i.json")
    assert [c.text for c in again.chunks] == [c.text for c in index.chunks]
