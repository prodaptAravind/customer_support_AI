from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import load_config
from backend.app.pinecone_retriever import PineconePolicyRetriever
from backend.scripts.env_check import check_env


def main() -> None:
    print("Loading configuration...")
    config = load_config()
    print("Checking environment...")
    env_result = check_env()
    if not env_result.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to create Pinecone embeddings.")
    if not env_result.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to create the Pinecone index.")
    if not env_result.dataset_path_exists:
        raise RuntimeError(f"Dataset file not found: {config.dataset_path}")

    print(
        f"Environment ready. Dataset chunks will use size={config.chunk_size}, overlap={config.chunk_overlap}."
    )
    print(f"Creating or connecting to Pinecone index '{config.pinecone_index_name}'...")
    retriever = PineconePolicyRetriever(
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
        seed_index=True,
        create_if_missing=True,
    )
    print("Index seeded successfully.")
    print(
        {
            "status": "ready",
            "index_name": config.pinecone_index_name,
            "namespace": config.pinecone_namespace,
            "chunks": len(retriever.documents),
        }
    )


if __name__ == "__main__":
    main()
