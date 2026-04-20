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


class RetrievedDocument(BaseModel):
    id: str
    title: str
    category: str
    score: float
    solution: str
    alternate_solution: str
    company_response: str
    content: str


class GenerateResponse(BaseModel):
    answer: str
    mode: str
    temperature: float
    max_tokens: int
    fallback_used: bool
    retrieved_docs: list[RetrievedDocument]
    prompt: dict[str, str]
    llm_provider: str
    raw_llm_response: dict[str, Any] | None = None

