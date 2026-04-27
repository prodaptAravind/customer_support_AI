from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import PolicyChunk

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except ImportError:  # pragma: no cover - handled at runtime if Pinecone is unavailable
    Pinecone = None


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if abs(maximum - minimum) < 1e-12:
        return [1.0 for _ in scores]
    return [(value - minimum) / (maximum - minimum) for value in scores]


def _document_text(document: PolicyChunk) -> str:
    return " ".join(
        part
        for part in [
            document.title,
            document.category,
            document.solution,
            document.alternate_solution,
            document.company_response,
            document.content,
        ]
        if part
    )


@dataclass(frozen=True)
class RerankedPolicy:
    document: PolicyChunk
    score: float


class BaseReranker:
    backend_name = "off"

    def rerank(self, query: str, results: list[Any], top_n: int | None = None) -> list[RerankedPolicy]:
        raise NotImplementedError


class LocalReranker(BaseReranker):
    backend_name = "local"

    def rerank(self, query: str, results: list[Any], top_n: int | None = None) -> list[RerankedPolicy]:
        if not results:
            return []

        query_tokens = set(tokenize(query))
        backend_scores = [float(item.score) for item in results]
        normalized_backend_scores = _normalize_scores(backend_scores)

        reranked: list[RerankedPolicy] = []
        for index, item in enumerate(results):
            document_tokens = set(tokenize(_document_text(item.document)))
            overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            combined_score = (0.65 * normalized_backend_scores[index]) + (0.35 * overlap)
            reranked.append(RerankedPolicy(document=item.document, score=combined_score))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_n] if top_n is not None else reranked


class PineconeReranker(BaseReranker):
    backend_name = "pinecone"

    def __init__(self, *, api_key: str, model: str):
        if Pinecone is None:
            raise RuntimeError("Pinecone SDK is not installed. Install backend requirements to enable reranking.")
        self.client = Pinecone(api_key=api_key)
        self.model = model

    def rerank(self, query: str, results: list[Any], top_n: int | None = None) -> list[RerankedPolicy]:
        if not results:
            return []

        documents = [{"id": item.document.id, "text": _document_text(item.document)} for item in results]
        rerank_result = self.client.inference.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n or len(results),
            return_documents=True,
            parameters={"truncate": "END", "rank_fields": ["text"]},
        )

        ranked: list[RerankedPolicy] = []
        for item in rerank_result.data:
            document_index = int(item.index)
            ranked.append(RerankedPolicy(document=results[document_index].document, score=float(item.score)))

        if top_n is not None:
            return ranked[:top_n]
        return ranked


def build_reranker(*, enable_reranking: bool, pinecone_api_key: str | None, rerank_model: str) -> BaseReranker | None:
    if not enable_reranking:
        return None
    if pinecone_api_key:
        try:
            return PineconeReranker(api_key=pinecone_api_key, model=rerank_model)
        except Exception:
            return LocalReranker()
    return LocalReranker()
