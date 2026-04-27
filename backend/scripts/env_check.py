from __future__ import annotations

import sys
from dataclasses import dataclass

from backend.app.config import load_config


@dataclass(frozen=True)
class EnvCheckResult:
    openai_api_key: bool
    pinecone_api_key: bool
    dataset_path_exists: bool
    chunk_size: int
    chunk_overlap: int
    ready_for_bootstrap: bool


def check_env() -> EnvCheckResult:
    config = load_config()
    openai_api_key = bool(config.openai_api_key)
    pinecone_api_key = bool(config.pinecone_api_key)
    dataset_path_exists = config.dataset_path.exists()
    ready_for_bootstrap = openai_api_key and pinecone_api_key and dataset_path_exists
    return EnvCheckResult(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        dataset_path_exists=dataset_path_exists,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        ready_for_bootstrap=ready_for_bootstrap,
    )


def main() -> int:
    result = check_env()
    print("Environment check")
    print(f"OPENAI_API_KEY: {'set' if result.openai_api_key else 'missing'}")
    print(f"PINECONE_API_KEY: {'set' if result.pinecone_api_key else 'missing'}")
    print(f"Dataset path: {'found' if result.dataset_path_exists else 'missing'}")
    print(f"Chunk size: {result.chunk_size}")
    print(f"Chunk overlap: {result.chunk_overlap}")
    print(f"Ready for Pinecone bootstrap: {'yes' if result.ready_for_bootstrap else 'no'}")
    return 0 if result.ready_for_bootstrap else 1


if __name__ == "__main__":
    raise SystemExit(main())
