from __future__ import annotations

import re
from dataclasses import dataclass

from .dataset_loader import cached_policy_documents
from .models import PolicyDocument
from .rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrievedPolicy:
    document: PolicyDocument
    score: float


class PolicyRetriever:
    def __init__(self, dataset_path):
        self.documents = list(cached_policy_documents(str(dataset_path)))
        self.tokenized_corpus = [tokenize(doc.content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[RetrievedPolicy] = []
        for index, score in ranked[:top_k]:
            results.append(RetrievedPolicy(document=self.documents[index], score=float(score)))
        return results

    @staticmethod
    def is_low_confidence(results: list[RetrievedPolicy], threshold: float) -> bool:
        if not results:
            return True
        return results[0].score < threshold

