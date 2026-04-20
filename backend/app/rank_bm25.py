from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


class BM25Okapi:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(doc) for doc in corpus]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs = []
        df = Counter()
        for doc in self.corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            df.update(freqs.keys())
        self.idf = {
            term: math.log(1 + (len(self.corpus) - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: Iterable[str]) -> list[float]:
        scores: list[float] = []
        q_tokens = list(query_tokens)
        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_len[idx] or 1
            for term in q_tokens:
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores

    def get_top_n(self, query_tokens: Iterable[str], documents: Sequence, n: int = 5):
        scores = self.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [documents[i] for i, _ in ranked[:n]]

