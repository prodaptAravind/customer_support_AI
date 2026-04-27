from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_PATTERN = re.compile(r"[^.!?]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _sentence_split(text: str) -> list[str]:
    return [chunk.strip() for chunk in SENTENCE_PATTERN.findall(text) if chunk.strip()]


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


@dataclass(frozen=True)
class RagasMetricResult:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_utilization: float | None
    backend: str
    used_ragas: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_utilization": self.context_utilization,
            "backend": self.backend,
            "used_ragas": self.used_ragas,
            "error": self.error,
        }


class RagasEvaluator:
    def __init__(self, *, api_key: str, model: str, embedding_model: str, timeout_seconds: float):
        from openai import OpenAI

        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._ragas_ready = False
        self._ragas_error: str | None = None
        self._llm = None
        self._embeddings = None
        self._metrics = None
        self._load_ragas_stack()

    def _load_ragas_stack(self) -> None:
        try:
            from ragas import SingleTurnSample
            try:
                from ragas.embeddings import embedding_factory
            except Exception:  # pragma: no cover - version compatibility path
                from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
            try:
                from ragas.metrics.collections import (
                    AnswerRelevancy,
                    ContextPrecision,
                    ContextUtilization,
                    Faithfulness,
                )
            except Exception:  # pragma: no cover - version compatibility path
                from ragas.metrics import (  # type: ignore
                    AnswerRelevancy,
                    ContextPrecision,
                    ContextUtilization,
                    Faithfulness,
                )
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._ragas_error = str(exc)
            self._ragas_ready = False
            return

        self._SingleTurnSample = SingleTurnSample
        self._llm = llm_factory(self.model, client=self.client)
        self._embeddings = embedding_factory("openai", model=self.embedding_model, client=self.client)
        self._metrics = {
            "faithfulness": Faithfulness(llm=self._llm),
            "answer_relevancy": AnswerRelevancy(llm=self._llm, embeddings=self._embeddings),
            "context_precision": ContextPrecision(llm=self._llm),
            "context_utilization": ContextUtilization(llm=self._llm),
        }
        self._ragas_ready = True

    @property
    def backend_name(self) -> str:
        return "ragas" if self._ragas_ready else "heuristic"

    def evaluate(self, *, user_input: str, response: str, retrieved_contexts: list[str]) -> RagasMetricResult:
        if self._ragas_ready:
            try:
                sample = self._SingleTurnSample(
                    user_input=user_input,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
                scores = {}
                for name, metric in self._metrics.items():
                    scores[name] = float(metric.score(sample))
                return RagasMetricResult(
                    faithfulness=scores.get("faithfulness"),
                    answer_relevancy=scores.get("answer_relevancy"),
                    context_precision=scores.get("context_precision"),
                    context_utilization=scores.get("context_utilization"),
                    backend=self.backend_name,
                    used_ragas=True,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                self._ragas_error = str(exc)

        return self._heuristic_scores(user_input=user_input, response=response, retrieved_contexts=retrieved_contexts)

    def _heuristic_scores(self, *, user_input: str, response: str, retrieved_contexts: list[str]) -> RagasMetricResult:
        query_tokens = _tokenize(user_input)
        response_tokens = _tokenize(response)
        contexts = [context or "" for context in retrieved_contexts]

        if not contexts:
            return RagasMetricResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_utilization=0.0,
                backend="heuristic",
                used_ragas=False,
                error=self._ragas_error,
            )

        faithfulness_scores = []
        utilization_scores = []
        precision_scores = []

        for sentence in _sentence_split(response):
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            faithfulness_scores.append(max(_overlap_ratio(sentence, context) for context in contexts))

        if not faithfulness_scores:
            faithfulness = 0.0
        else:
            faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)

        for context in contexts:
            context_tokens = _tokenize(context)
            if not context_tokens:
                precision_scores.append(0.0)
                utilization_scores.append(0.0)
                continue
            precision_scores.append(len(query_tokens & context_tokens) / max(1, len(query_tokens)))
            utilization_scores.append(len(response_tokens & context_tokens) / max(1, len(response_tokens)))

        answer_relevancy = len(query_tokens & response_tokens) / max(1, len(query_tokens))
        context_precision = sum(precision_scores) / len(precision_scores)
        context_utilization = sum(utilization_scores) / len(utilization_scores)

        return RagasMetricResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_utilization=context_utilization,
            backend="heuristic",
            used_ragas=False,
            error=self._ragas_error,
        )


@lru_cache(maxsize=4)
def build_ragas_evaluator(
    api_key: str,
    model: str,
    embedding_model: str,
    timeout_seconds: float,
) -> RagasEvaluator:
    return RagasEvaluator(
        api_key=api_key,
        model=model,
        embedding_model=embedding_model,
        timeout_seconds=timeout_seconds,
    )
