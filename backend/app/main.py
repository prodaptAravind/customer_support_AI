from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .logger import JsonlLogger
from .models import GenerateRequest, GenerateResponse, RetrievedDocument, RagasMetrics
from .openai_client import OfflineFallbackClient, OpenAIClient
from .prompting import build_prompt, fallback_prompt, mode_defaults
from .ragas_evaluator import build_ragas_evaluator
from .retrieval import PolicyRetriever

@lru_cache(maxsize=1)
def get_config():
    return load_config()


app = FastAPI(title="AI-Assisted Customer Support Response Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_config().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_app_state() -> dict:
    config = get_config()
    logger = JsonlLogger(config.log_path)
    retriever = PolicyRetriever(config)
    llm_client = (
        OpenAIClient(config.openai_api_key, config.openai_model)
        if config.openai_api_key
        else OfflineFallbackClient()
    )
    return {
        "config": config,
        "logger": logger,
        "retriever": retriever,
        "llm_client": llm_client,
        "ragas_evaluator": None,
    }


@app.on_event("startup")
def startup() -> None:
    app.state.runtime = build_app_state()


def runtime() -> dict:
    if not hasattr(app.state, "runtime"):
        app.state.runtime = build_app_state()
    return app.state.runtime


def _docs_to_response_docs(results) -> list[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    for item in results:
        docs.append(
            RetrievedDocument(
                id=item.document.id,
                source_id=item.document.source_id,
                title=item.document.title,
                category=item.document.category,
                score=item.score,
                solution=item.document.solution,
                alternate_solution=item.document.alternate_solution,
                company_response=item.document.company_response,
                content=item.document.content,
                chunk_index=item.document.chunk_index,
                chunk_count=item.document.chunk_count,
            )
        )
    return docs


@app.get("/api/health")
def health():
    state = runtime()
    config = state["config"]
    retriever = state["retriever"]
    llm_client = state["llm_client"]
    return {
        "status": "ok",
        "documents": len(retriever.documents),
        "llm_provider": llm_client.provider_name,
        "retrieval_backend": retriever.backend_name,
        "reranking_backend": getattr(retriever, "reranking_backend", "off"),
        "ragas_enabled": config.ragas_enabled,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


def _get_ragas_evaluator(state: dict):
    config = state["config"]
    if not config.openai_api_key:
        return None
    evaluator = state.get("ragas_evaluator")
    if evaluator is None:
        evaluator = build_ragas_evaluator(
            config.openai_api_key,
            config.ragas_model,
            config.ragas_embedding_model,
            config.ragas_timeout_seconds,
        )
        state["ragas_evaluator"] = evaluator
    return evaluator


@app.post("/api/respond", response_model=GenerateResponse)
def respond(request: GenerateRequest):
    state = runtime()
    config = state["config"]
    retriever = state["retriever"]
    logger = state["logger"]
    llm_client = state["llm_client"]

    defaults = mode_defaults(request.mode)
    temperature = request.temperature if request.temperature is not None else float(defaults["temperature"])
    max_tokens = request.max_tokens if request.max_tokens is not None else int(defaults["max_tokens"])

    search_results = retriever.search(request.complaint, top_k=request.top_k)
    response_docs = _docs_to_response_docs(search_results)
    low_confidence = retriever.is_low_confidence(search_results, config.fallback_threshold)

    if low_confidence:
        prompt = fallback_prompt()
        answer = "Please escalate this issue to a human support agent."
        raw_response = {
            "provider": llm_client.provider_name,
            "reason": "low_similarity_score",
        }
        logger.log(
            {
                "query": request.complaint,
                "mode": request.mode,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "fallback_used": True,
                "retrieved_docs": [doc.model_dump() for doc in response_docs],
                "prompt": prompt.__dict__,
                "answer": answer,
                "provider": llm_client.provider_name,
            }
        )
        return GenerateResponse(
            answer=answer,
            mode=request.mode,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_used=True,
            retrieved_docs=response_docs,
            prompt={"system": prompt.system, "user": prompt.user},
            llm_provider=llm_client.provider_name,
            reranking_backend=getattr(retriever, "reranking_backend", "off"),
            raw_llm_response=raw_response,
        )

    prompt = build_prompt(request.mode, request.complaint, response_docs)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    try:
        llm_result = llm_client.generate(messages, temperature=temperature, max_tokens=max_tokens)
        answer = llm_result.content or (response_docs[0].company_response if response_docs else "")
    except Exception as exc:
        answer = response_docs[0].company_response if response_docs else "Please escalate this issue to a human support agent."
        logger.log(
            {
                "query": request.complaint,
                "mode": request.mode,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "fallback_used": True,
                "retrieved_docs": [doc.model_dump() for doc in response_docs],
                "prompt": prompt.__dict__,
                "answer": answer,
                "provider": llm_client.provider_name,
                "error": str(exc),
            }
        )
        return GenerateResponse(
            answer=answer,
            mode=request.mode,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_used=True,
            retrieved_docs=response_docs,
            prompt={"system": prompt.system, "user": prompt.user},
            llm_provider=llm_client.provider_name,
            reranking_backend=getattr(retriever, "reranking_backend", "off"),
            raw_llm_response={"error": str(exc), "provider": llm_client.provider_name},
        )
    if not llm_result.content:
        answer = response_docs[0].company_response if response_docs else "Please escalate this issue to a human support agent."
        logger.log(
            {
                "query": request.complaint,
                "mode": request.mode,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "fallback_used": True,
                "retrieved_docs": [doc.model_dump() for doc in response_docs],
                "prompt": prompt.__dict__,
                "answer": answer,
                "provider": llm_client.provider_name,
                "error": "Empty LLM response",
            }
        )
        return GenerateResponse(
            answer=answer,
            mode=request.mode,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_used=True,
            retrieved_docs=response_docs,
            prompt={"system": prompt.system, "user": prompt.user},
            llm_provider=llm_client.provider_name,
            reranking_backend=getattr(retriever, "reranking_backend", "off"),
            raw_llm_response={"error": "Empty LLM response", "provider": llm_client.provider_name},
        )
    ragas_metrics = None
    if request.include_metrics or config.ragas_enabled:
        evaluator = _get_ragas_evaluator(state)
        if evaluator is not None:
            metric_result = evaluator.evaluate(
                user_input=request.complaint,
                response=llm_result.content,
                retrieved_contexts=[doc.content for doc in response_docs],
            )
            ragas_metrics = RagasMetrics(**metric_result.to_dict())
    logger.log(
        {
            "query": request.complaint,
            "mode": request.mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "fallback_used": False,
            "retrieved_docs": [doc.model_dump() for doc in response_docs],
            "prompt": prompt.__dict__,
            "answer": llm_result.content,
            "provider": llm_client.provider_name,
            "ragas_metrics": ragas_metrics.model_dump() if ragas_metrics else None,
        }
    )
    return GenerateResponse(
        answer=answer,
        mode=request.mode,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback_used=False,
        retrieved_docs=response_docs,
        prompt={"system": prompt.system, "user": prompt.user},
        llm_provider=llm_client.provider_name,
        reranking_backend=getattr(retriever, "reranking_backend", "off"),
        ragas_metrics=ragas_metrics,
        raw_llm_response=llm_result.raw_response,
    )
