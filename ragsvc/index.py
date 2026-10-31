"""Indexul: documente → secțiuni → căutare hibridă (BM25 + n-grame de caractere, fuzionate cu RRF).

Alegerile vin din experimentele din [rag-eval-lab](https://github.com/cs1lyuha/rag-eval-lab):
chunking pe secțiuni `##`, hibrid fără reranking, k=3.

Reindexarea e incrementală: fiecare document are un hash de conținut. `upsert` cu același
conținut nu face nimic, iar un document schimbat își înlocuiește doar secțiunile lui. Statisticile
globale (IDF, matricea densă) se recalculează, fiind ieftine pe un corpus de ordinul miilor de secțiuni.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ragsvc.text import tokens, words


@dataclass(frozen=True)
class Chunk:
    id: str          # „livrare#costul-livrarii”
    doc: str
    title: str       # titlul documentului
    section: str
    text: str


def slugify(value: str) -> str:
    import unicodedata
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def split_sections(name: str, markdown: str) -> list[Chunk]:
    title_m = re.search(r"^# (.+)$", markdown, flags=re.M)
    title = title_m.group(1).strip() if title_m else name
    heads = list(re.finditer(r"^## (.+)$", markdown, flags=re.M))
    if not heads:
        body = re.sub(r"^# .+$", "", markdown, flags=re.M).strip()
        return [Chunk(f"{name}#intro", name, title, title, body)] if body else []
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(markdown)
        body = markdown[h.end():end].strip()
        if body:
            out.append(Chunk(f"{name}#{slugify(h.group(1))}", name, title, h.group(1).strip(), body))
    return out


class HybridIndex:
    def __init__(self, dim: int = 2048, rrf_k: int = 60) -> None:
        self.dim, self.rrf_k = dim, rrf_k
        self.docs: dict[str, str] = {}          # nume → hash conținut
        self.chunks: list[Chunk] = []
        self.lock = threading.RLock()
        self._rebuild()

    # ---------- ingestie ----------

    def upsert(self, name: str, markdown: str) -> str:
        """Întoarce „unchanged”, „created” sau „updated”."""
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("numele documentului: litere mici, cifre, - și _ (maxim 64)")
        digest = hashlib.sha256(markdown.encode()).hexdigest()
        with self.lock:
            if self.docs.get(name) == digest:
                return "unchanged"
            status = "updated" if name in self.docs else "created"
            self.chunks = [c for c in self.chunks if c.doc != name] + split_sections(name, markdown)
            self.docs[name] = digest
            self._rebuild()
            return status

    def delete(self, name: str) -> bool:
        with self.lock:
            if name not in self.docs:
                return False
            del self.docs[name]
            self.chunks = [c for c in self.chunks if c.doc != name]
            self._rebuild()
            return True

    # ---------- statistici ----------

    def _grams(self, text: str) -> Counter[int]:
        out: Counter[int] = Counter()
        for w in words(text):
            padded = f"<{w}>"
            for n in (3, 4):
                for i in range(max(1, len(padded) - n + 1)):
                    out[int(hashlib.blake2b(padded[i:i + n].encode(), digest_size=4).hexdigest(), 16) % self.dim] += 1
        return out

    def _rebuild(self) -> None:
        texts = [f"{c.title} {c.section}\n{c.text}" for c in self.chunks]
        self.bm25_docs = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.bm25_docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        df: Counter[str] = Counter()
        for d in self.bm25_docs:
            df.update(d.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        gram_df = np.zeros(self.dim)
        grams = [self._grams(t) for t in texts]
        for g in grams:
            for b in g:
                gram_df[b] += 1
        self.gram_idf = np.log((1 + n) / (1 + gram_df)) + 1
        self.matrix = np.vstack([self._vec(g) for g in grams]) if grams else np.zeros((0, self.dim))

    def _vec(self, grams: Counter[int]) -> np.ndarray:
        v = np.zeros(self.dim)
        for b, count in grams.items():
            v[b] = 1 + math.log(count)
        v *= self.gram_idf
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    # ---------- căutare ----------

    def _bm25(self, q: list[str], i: int, k1: float = 1.5, b: float = 0.75) -> float:
        doc, length, s = self.bm25_docs[i], self.lengths[i], 0.0
        for t in q:
            tf = doc.get(t, 0)
            if tf:
                s += self.idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / self.avg_len))
        return s

    def search(self, query: str, k: int = 3, depth: int = 30) -> list[tuple[Chunk, float, float]]:
        """Întoarce (chunk, scor RRF, scor BM25). Scorul BM25 brut ajută la decizia de refuz."""
        with self.lock:
            if not self.chunks:
                return []
            q = tokens(query)
            bm = [self._bm25(q, i) for i in range(len(self.chunks))]
            dense = self.matrix @ self._vec(self._grams(query))
            fused: dict[int, float] = {}
            for ranking in (sorted(range(len(bm)), key=lambda i: -bm[i])[:depth],
                            list(np.argsort(-dense, kind="stable")[:depth])):
                for rank, i in enumerate(ranking, start=1):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1 / (self.rrf_k + rank)
            top = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
            return [(self.chunks[i], s, bm[i]) for i, s in top]
