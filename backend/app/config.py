from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


@dataclass(frozen=True)
class AppConfig:
    dataset_path: Path
    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    enable_reranking: bool
    rerank_model: str
    rerank_candidate_k: int
    chunk_size: int
    chunk_overlap: int
    log_path: Path
    fallback_threshold: float
    cors_origins: list[str]
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_dimension: int
    pinecone_index_host: str | None
    pinecone_wait_seconds: float
    ragas_enabled: bool
    ragas_model: str
    ragas_embedding_model: str
    ragas_timeout_seconds: float
    retrieval_backend: str


def _default_dataset_path() -> Path:
    env_path = os.getenv("POLICY_DATASET_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "Complaint Dataset.xlsx"


def _load_missing_from_env_file(path: Path) -> None:
    if not path.exists():
        return
    values = dotenv_values(path)
    for key, value in values.items():
        if value is None:
            continue
        if not os.getenv(key):
            os.environ[key] = value


def load_config() -> AppConfig:
    root_dir = Path(__file__).resolve().parents[2]
    backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir / ".env", override=False)
    load_dotenv(backend_dir / ".env", override=False)
    _load_missing_from_env_file(root_dir / ".env.example")
    _load_missing_from_env_file(backend_dir / ".env.example")

    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() in {"1", "true", "yes", "on"}
    ragas_enabled = os.getenv("RAGAS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    return AppConfig(
        dataset_path=_default_dataset_path(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        enable_reranking=enable_reranking,
        rerank_model=os.getenv("RERANK_MODEL", "bge-reranker-v2-m3"),
        rerank_candidate_k=int(os.getenv("RERANK_CANDIDATE_K", "10")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        log_path=Path(os.getenv("LOG_PATH", root_dir / "logs" / "requests.jsonl")),
        fallback_threshold=float(os.getenv("FALLBACK_THRESHOLD", "0.25")),
        cors_origins=cors_origins,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "customer-support-policies"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "complaints"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        pinecone_dimension=int(os.getenv("PINECONE_DIMENSION", "1536")),
        pinecone_index_host=os.getenv("PINECONE_INDEX_HOST") or None,
        pinecone_wait_seconds=float(os.getenv("PINECONE_WAIT_SECONDS", "2")),
        ragas_enabled=ragas_enabled,
        ragas_model=os.getenv("RAGAS_MODEL", "gpt-4o-mini"),
        ragas_embedding_model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
        ragas_timeout_seconds=float(os.getenv("RAGAS_TIMEOUT_SECONDS", "20")),
        retrieval_backend="pinecone" if pinecone_api_key and os.getenv("OPENAI_API_KEY") else "bm25",
    )
