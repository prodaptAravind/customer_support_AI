from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .openai_client import OpenAIClient

from .dataset_loader import cached_policy_chunks
from .models import PolicyChunk

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
except ImportError:  # pragma: no cover - handled at runtime if Pinecone is unavailable
    Pinecone = None
    ServerlessSpec = None


@dataclass
class RetrievedPolicy:
    document: PolicyChunk
    score: float


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return getattr(value, "__dict__", {})


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class PineconePolicyRetriever:
    def __init__(
        self,
        *,
        dataset_path,
        openai_api_key: str,
        openai_embedding_model: str,
        pinecone_api_key: str,
        index_name: str,
        namespace: str,
        cloud: str,
        region: str,
        dimension: int,
        chunk_size: int,
        chunk_overlap: int,
        index_host: str | None = None,
        wait_seconds: float = 2.0,
        seed_index: bool = False,
        create_if_missing: bool = False,
    ):
        if Pinecone is None or ServerlessSpec is None:
            raise RuntimeError(
                "Pinecone SDK is not installed. Install the backend requirements to enable Pinecone retrieval."
            )

        self.documents = list(
            cached_policy_chunks(str(dataset_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.wait_seconds = wait_seconds
        self.embedding_model = openai_embedding_model
        self.openai = OpenAIClient(api_key=openai_api_key, model=openai_embedding_model)
        self.client = Pinecone(api_key=pinecone_api_key)
        self.index = self._ensure_index(
            cloud=cloud,
            region=region,
            index_host=index_host,
            create_if_missing=create_if_missing,
        )
        if seed_index:
            self._upsert_documents()

    def _ensure_index(self, *, cloud: str, region: str, index_host: str | None, create_if_missing: bool):
        if index_host:
            return self.client.Index(host=index_host)

        try:
            info = self.client.describe_index(name=self.index_name)
        except Exception:
            if not create_if_missing:
                raise RuntimeError(
                    f"Pinecone index '{self.index_name}' was not found. "
                    "Run the bootstrap script once to create and seed it."
                )
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            info = self._wait_for_index_ready()

        info_map = _as_mapping(info)
        reported_dimension = info_map.get("dimension")
        if reported_dimension is not None and int(reported_dimension) != self.dimension:
            raise RuntimeError(
                f"Pinecone index '{self.index_name}' has dimension {reported_dimension}, "
                f"but the configured embedding dimension is {self.dimension}."
            )

        host = info_map.get("host")
        if host:
            return self.client.Index(host=host)
        return self.client.Index(name=self.index_name)

    def _wait_for_index_ready(self):
        deadline = time.time() + 120
        last_info = None
        while time.time() < deadline:
            try:
                last_info = self.client.describe_index(name=self.index_name)
            except Exception:
                time.sleep(self.wait_seconds)
                continue
            status = _get_field(last_info, "status", {})
            status_map = _as_mapping(status)
            if status_map.get("ready") is True:
                return last_info
            time.sleep(self.wait_seconds)
        raise TimeoutError(f"Pinecone index '{self.index_name}' was not ready in time.")

    def _embed(self, text: str) -> list[float]:
        response = self.openai.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
            dimensions=self.dimension,
        )
        return list(response.data[0].embedding)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float",
            dimensions=self.dimension,
        )
        return [list(item.embedding) for item in response.data]

    def _upsert_documents(self) -> None:
        batch_size = 32
        for start in range(0, len(self.documents), batch_size):
            batch = self.documents[start : start + batch_size]
            embeddings = self._embed_batch([doc.content for doc in batch])
            vectors = []
            for doc, embedding in zip(batch, embeddings):
                vectors.append(
                    {
                        "id": doc.id,
                        "values": embedding,
                        "metadata": {
                            "id": doc.id,
                            "source_id": doc.source_id,
                            "title": doc.title,
                            "category": doc.category,
                            "solution": doc.solution,
                            "alternate_solution": doc.alternate_solution,
                            "company_response": doc.company_response,
                            "content": doc.content,
                            "chunk_index": doc.chunk_index,
                            "chunk_count": doc.chunk_count,
                        },
                    }
                )
            if vectors:
                self.index.upsert(vectors=vectors, namespace=self.namespace)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        query_vector = self._embed(query)
        if not any(abs(value) > 1e-12 for value in query_vector):
            return []
        result = self.index.query(
            namespace=self.namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )
        matches = _get_field(result, "matches", [])
        return [self._match_to_retrieved_policy(match) for match in matches]

    def _match_to_retrieved_policy(self, match: Any) -> RetrievedPolicy:
        metadata = _as_mapping(_get_field(match, "metadata", {}))
        document = PolicyChunk(
            id=str(metadata.get("id") or _get_field(match, "id", "")),
            source_id=str(metadata.get("source_id", "")),
            title=str(metadata.get("title", "")),
            category=str(metadata.get("category", "")),
            solution=str(metadata.get("solution", "")),
            alternate_solution=str(metadata.get("alternate_solution", "")),
            company_response=str(metadata.get("company_response", "")),
            content=str(metadata.get("content", "")),
            chunk_index=int(metadata.get("chunk_index", 1) or 1),
            chunk_count=int(metadata.get("chunk_count", 1) or 1),
        )
        return RetrievedPolicy(document=document, score=float(_get_field(match, "score", 0.0)))
