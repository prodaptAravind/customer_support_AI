# AI-Assisted Customer Support Response Generator

This project turns a customer complaint into a policy-based support reply using:

- A local complaint/policy dataset from `Complaint Dataset.xlsx`
- Pinecone retrieval when `PINECONE_API_KEY` is set, with recursive chunked BM25 fallback otherwise
- Pinecone reranking when available, with a local reranker fallback
- OpenAI chat responses for the final answer
- Optional RAGAS-style evaluation metrics for generated responses
- A React UI for trying strict and friendly response modes

## Project Structure

- `backend/` FastAPI service, Pinecone/BM25 retrieval, prompt building, logging
- `frontend/` React + Vite UI
- `Complaint Dataset.xlsx` local policy dataset used by the backend

## Backend Setup

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

To create and seed the Pinecone index manually, run:

```powershell
cd c:\Users\Administrator\Desktop\customer_support_AI
python -m backend.scripts.bootstrap_pinecone
```

If you are already inside the `backend` folder, use:

```powershell
python bootstrap_pinecone.py
```

To check whether your environment is ready before bootstrap:

```powershell
python check_env.py
```

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the UI can talk to the backend without extra CORS work.

## Environment Variables

- `OPENAI_API_KEY` for OpenAI chat responses
- `OPENAI_MODEL` default: `gpt-5.4-mini`
- `OPENAI_EMBEDDING_MODEL` default: `text-embedding-3-small`
- `ENABLE_RERANKING` default: `true`
- `RERANK_MODEL` default: `bge-reranker-v2-m3`
- `RERANK_CANDIDATE_K` default: `10`
- `CHUNK_SIZE` default: `500`
- `CHUNK_OVERLAP` default: `80`
- `POLICY_DATASET_PATH` optional path override for the spreadsheet
- `FALLBACK_THRESHOLD` default: `0.25`
- `LOG_PATH` default: `logs/requests.jsonl`
- `CORS_ORIGINS` default: `http://localhost:5173`
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

Copy `.env.example` to `.env` and fill in the API keys you want to use.

## Notes

- If `OPENAI_API_KEY` is not set, the backend uses an offline fallback response generator so the app still runs for local testing.
- If `PINECONE_API_KEY` is not set, the backend falls back to recursive chunked BM25 retrieval.
- If reranking is enabled, the backend uses Pinecone inference reranking when Pinecone is configured and a local lexical reranker otherwise.
- RAGAS metrics are opt-in from the UI with the `Include RAGAS metrics` checkbox.
- If OpenAI generation fails, the backend returns the top policy response instead of failing silently.
- The backend writes JSONL logs containing the query, retrieved docs, prompt, and generation parameters.
