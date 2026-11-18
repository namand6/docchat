"""Hybrid retrieval: BM25 for exact terms + embeddings for meaning, fused with reciprocal rank fusion.

- bm25:  implemented here (numpy).
- lsa:   TF-IDF + truncated SVD, implemented here; the dependency-free baseline.
- embed: a small local neural model (BAAI/bge-small-en-v1.5 via fastembed, ONNX, no GPU, no API).
- hybrid: RRF of bm25 + embed (falls back to lsa if fastembed isn't installed).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .ingest import Chunk

STOP = set("""a an and are as at be but by for from has have how i if in into is it its of on or that the
their them then there these they this to was were what when where which who why will with you your do does
can could should would may might not no yes than so such about over under after before during""".split())


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        # light stemming so "requests"/"requested"/"request" meet
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=float)
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s


class Dense:
    """TF-IDF → truncated SVD (latent semantic analysis) → unit vectors; cosine similarity."""

    def __init__(self, docs: list[list[str]], dims: int = 96):
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
        n = len(docs)
        df = np.zeros(len(self.vocab))
        for d in docs:
            for t in set(d):
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1
        m = np.stack([self._tfidf(d) for d in docs]) if docs else np.zeros((0, len(self.vocab)))
        k = max(1, min(dims, min(m.shape) - 1)) if min(m.shape) > 1 else 1
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        self.proj = vt[:k].T
        self.vecs = self._unit(m @ self.proj)

    def _tfidf(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, c in Counter(toks).items():
            if t in self.vocab:
                v[self.vocab[t]] = 1 + math.log(c)
        return v * self.idf

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norm == 0, 1, norm)

    def scores(self, query: list[str]) -> np.ndarray:
        q = self._unit(self._tfidf(query) @ self.proj)
        return self.vecs @ q


class Embed:
    MODEL = "BAAI/bge-small-en-v1.5"
    _model = None

    def __init__(self, texts: list[str], vecs: np.ndarray | None = None):
        self.vecs = vecs if vecs is not None else self._encode(texts, query=False)

    @classmethod
    def available(cls) -> bool:
        try:
            import fastembed  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _encode(cls, texts: list[str], query: bool) -> np.ndarray:
        if cls._model is None:
            from fastembed import TextEmbedding
            cls._model = TextEmbedding(cls.MODEL)
        it = cls._model.query_embed(texts) if query else cls._model.passage_embed(texts)
        return np.array(list(it), dtype=np.float32)

    def scores(self, query: str) -> np.ndarray:
        return self.vecs @ self._encode([query], query=True)[0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0) + 1 / (k + r + 1)
    return sorted(score, key=score.get, reverse=True)


class Index:
    MODES = ("bm25", "lsa", "embed", "hybrid")

    def __init__(self, chunks: list[Chunk], vecs: np.ndarray | None = None, use_embed: bool | None = None):
        self.chunks = chunks
        toks = [tokens(f"{c.heading} {c.text}") for c in chunks]
        self.bm25 = BM25(toks)
        self.dense = Dense(toks)
        use = Embed.available() if use_embed is None else use_embed
        self.embed = Embed([f"{c.heading}. {c.text}" for c in chunks], vecs) if use and chunks else None

    def _ranking(self, mode: str, query: str, q: list[str]) -> list[int]:
        if mode == "bm25":
            s = self.bm25.scores(q)
            return [int(i) for i in np.argsort(-s) if s[i] > 0]
        if mode == "lsa":
            return [int(i) for i in np.argsort(-self.dense.scores(q))]
        if mode == "embed":
            if not self.embed:
                raise RuntimeError("embed mode needs `pip install fastembed`")
            return [int(i) for i in np.argsort(-self.embed.scores(query))]
        raise ValueError(f"mode must be one of {self.MODES}")

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        q = tokens(query)
        if not q or not self.chunks:
            return []
        if mode == "hybrid":
            semantic = "embed" if self.embed else "lsa"
            order = rrf([self._ranking("bm25", query, q)[:50], self._ranking(semantic, query, q)[:50]])
        else:
            order = self._ranking(mode, query, q)
        return [(self.chunks[i], 0.0) for i in order[:k]]

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps([c.to_dict() for c in self.chunks]))
        if self.embed is not None:
            np.save(Path(path).with_suffix(".npy"), self.embed.vecs)

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        chunks = [Chunk(**c) for c in json.loads(Path(path).read_text())]
        vec_file = Path(path).with_suffix(".npy")
        vecs = np.load(vec_file) if vec_file.exists() else None
        if vecs is not None and len(vecs) != len(chunks):
            vecs = None
        return cls(chunks, vecs, use_embed=(vecs is not None) or None)
