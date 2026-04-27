from __future__ import annotations

import re
from dataclasses import dataclass

from .dataset_loader import cached_policy_chunks
from .models import PolicyChunk
from .reranking import BaseReranker, build_reranker
from .pinecone_retriever import PineconePolicyRetriever
from .rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrievedPolicy:
    document: PolicyChunk
    score: float


class BM25PolicyRetriever:
    def __init__(self, dataset_path, chunk_size: int, chunk_overlap: int):
        self.documents = list(
            cached_policy_chunks(str(dataset_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
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


class PolicyRetriever:
    def __init__(self, config):
        self.config = config
        self.backend_name = "bm25"
        self.reranking_backend = "off"
        self._backend = BM25PolicyRetriever(config.dataset_path, config.chunk_size, config.chunk_overlap)
        self.reranker: BaseReranker | None = build_reranker(
            enable_reranking=getattr(config, "enable_reranking", False),
            pinecone_api_key=getattr(config, "pinecone_api_key", None),
            rerank_model=getattr(config, "rerank_model", "bge-reranker-v2-m3"),
        )
        if self.reranker is not None:
            self.reranking_backend = self.reranker.backend_name

        if getattr(config, "pinecone_api_key", None) and getattr(config, "openai_api_key", None):
            try:
                self._backend = PineconePolicyRetriever(
                    dataset_path=config.dataset_path,
                    openai_api_key=config.openai_api_key,
                    openai_embedding_model=config.openai_embedding_model,
                    pinecone_api_key=config.pinecone_api_key,
                    index_name=config.pinecone_index_name,
                    namespace=config.pinecone_namespace,
                    cloud=config.pinecone_cloud,
                    region=config.pinecone_region,
                    dimension=config.pinecone_dimension,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    index_host=config.pinecone_index_host,
                    wait_seconds=config.pinecone_wait_seconds,
                    seed_index=False,
                    create_if_missing=False,
                )
                self.backend_name = "pinecone"
            except Exception:
                self._backend = BM25PolicyRetriever(config.dataset_path, config.chunk_size, config.chunk_overlap)
                self.backend_name = "bm25"

        self.documents = self._backend.documents

    def search(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        candidate_k = max(top_k, getattr(self.config, "rerank_candidate_k", top_k))
        results = self._backend.search(query, top_k=candidate_k)
        if self.reranker is None or len(results) <= 1:
            return results[:top_k]
        reranked = self.reranker.rerank(query, results, top_n=top_k)
        return [RetrievedPolicy(document=item.document, score=item.score) for item in reranked]

    @staticmethod
    def is_low_confidence(results: list[RetrievedPolicy], threshold: float) -> bool:
        if not results:
            return True
        return results[0].score < threshold
