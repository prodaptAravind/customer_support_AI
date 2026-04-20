from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .logger import JsonlLogger
from .models import GenerateRequest, GenerateResponse, RetrievedDocument
from .prompting import build_prompt, fallback_prompt, mode_defaults
from .retrieval import PolicyRetriever
from .sarvam_client import OfflineFallbackClient, SarvamClient


config = load_config()
retriever = PolicyRetriever(config.dataset_path)
logger = JsonlLogger(config.log_path)
llm_client = (
    SarvamClient(config.sarvam_api_key, config.sarvam_base_url, config.sarvam_model)
    if config.sarvam_api_key
    else OfflineFallbackClient()
)

app = FastAPI(title="AI-Assisted Customer Support Response Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _docs_to_response_docs(results) -> list[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    for item in results:
        docs.append(
            RetrievedDocument(
                id=item.document.id,
                title=item.document.title,
                category=item.document.category,
                score=item.score,
                solution=item.document.solution,
                alternate_solution=item.document.alternate_solution,
                company_response=item.document.company_response,
                content=item.document.content,
            )
        )
    return docs


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "documents": len(retriever.documents),
        "llm_provider": llm_client.provider_name,
    }


@app.post("/api/respond", response_model=GenerateResponse)
def respond(request: GenerateRequest):
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
            "reason": "low_bm25_score",
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
            raw_llm_response=raw_response,
        )

    prompt = build_prompt(request.mode, request.complaint, response_docs)
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    llm_result = llm_client.generate(messages, temperature=temperature, max_tokens=max_tokens)
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
        }
    )
    return GenerateResponse(
        answer=llm_result.content,
        mode=request.mode,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback_used=False,
        retrieved_docs=response_docs,
        prompt={"system": prompt.system, "user": prompt.user},
        llm_provider=llm_client.provider_name,
        raw_llm_response=llm_result.raw_response,
    )

