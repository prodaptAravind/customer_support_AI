from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    complaint: str = Field(min_length=1, description="Customer complaint text")
    mode: str = Field(default="strict", pattern="^(strict|friendly)$")
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    top_k: int = Field(default=3, ge=1, le=5)
    include_metrics: bool = Field(default=False)


@dataclass(frozen=True)
class PolicyDocument:
    id: str
    title: str
    category: str
    solution: str
    alternate_solution: str
    company_response: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyChunk:
    id: str
    source_id: str
    title: str
    category: str
    solution: str
    alternate_solution: str
    company_response: str
    content: str
    chunk_index: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievedDocument(BaseModel):
    id: str
    source_id: str
    title: str
    category: str
    score: float
    solution: str
    alternate_solution: str
    company_response: str
    content: str
    chunk_index: int
    chunk_count: int


class RagasMetrics(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_utilization: float | None = None
    backend: str
    used_ragas: bool
    error: str | None = None


class GenerateResponse(BaseModel):
    answer: str
    mode: str
    temperature: float
    max_tokens: int
    fallback_used: bool
    retrieved_docs: list[RetrievedDocument]
    prompt: dict[str, str]
    llm_provider: str
    reranking_backend: str | None = None
    ragas_metrics: RagasMetrics | None = None
    raw_llm_response: dict[str, Any] | None = None
