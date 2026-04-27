# Backend

FastAPI service for the AI-assisted customer support response generator.

## What it does

- Loads the local `Complaint Dataset.xlsx` file
- Retrieves the top policy matches with Pinecone when configured, or BM25 as a fallback
- Reranks the initial matches with Pinecone inference reranking when available, or a local reranker otherwise
- Builds strict or friendly prompts
- Calls OpenAI when `OPENAI_API_KEY` is set
- Can compute RAGAS-style evaluation metrics for a generated answer when requested
- Falls back to a human-escalation response when no good match is found
- Logs every request in JSONL format

## Run

```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

To create and seed the Pinecone index manually:

```powershell
cd c:\Users\Administrator\Desktop\customer_support_AI
python -m backend.scripts.bootstrap_pinecone
```

If you are already inside `backend/`, run:

```powershell
python bootstrap_pinecone.py
```

To check whether the environment is ready:

```powershell
python check_env.py
```

## Environment

- `OPENAI_API_KEY`
- `OPENAI_MODEL` default: `gpt-5.4-mini`
- `OPENAI_EMBEDDING_MODEL` default: `text-embedding-3-small`
- `ENABLE_RERANKING` default: `true`
- `RERANK_MODEL` default: `bge-reranker-v2-m3`
- `RERANK_CANDIDATE_K` default: `10`
- `CHUNK_SIZE` default: `500`
- `CHUNK_OVERLAP` default: `80`
- `POLICY_DATASET_PATH` optional override for the spreadsheet path
- `FALLBACK_THRESHOLD` default: `0.25`
- `LOG_PATH` default: `logs/requests.jsonl`
- `RAGAS_ENABLED` default: `false`
- `RAGAS_MODEL` default: `gpt-4o-mini`
- `RAGAS_EMBEDDING_MODEL` default: `text-embedding-3-small`
- `RAGAS_TIMEOUT_SECONDS` default: `20`
- `PINECONE_API_KEY` enables Pinecone-backed retrieval
- `PINECONE_INDEX_NAME` default: `customer-support-policies`
- `PINECONE_NAMESPACE` default: `complaints`
- `PINECONE_REGION` default: `us-east-1`
- `PINECONE_CLOUD` default: `aws`
- `PINECONE_DIMENSION` default: `1536`
- `PINECONE_INDEX_HOST` optional manual index host

When `PINECONE_API_KEY` is set, the backend creates or reuses a Pinecone index and stores recursive chunks of the local complaint policies as vectors using OpenAI embeddings. If the key is missing, it falls back to the chunked BM25 retriever.

If OpenAI generation fails at request time, the endpoint returns the top retrieved policy response instead of dropping the request.
