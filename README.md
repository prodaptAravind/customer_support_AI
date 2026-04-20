# AI-Assisted Customer Support Response Generator

This project turns a customer complaint into a policy-based support reply using:

- A local complaint/policy dataset from `Complaint Dataset.xlsx`
- BM25 retrieval, without embeddings or a vector database
- Sarvam AI chat completions for the final answer
- A React UI for trying strict and friendly response modes

## Project Structure

- `backend/` FastAPI service, BM25 retrieval, prompt building, logging
- `frontend/` React + Vite UI
- `Complaint Dataset.xlsx` local policy dataset used by the backend

## Backend Setup

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the UI can talk to the backend without extra CORS work.

## Environment Variables

- `SARVAM_API_KEY` for Sarvam chat completions
- `SARVAM_BASE_URL` default: `https://api.sarvam.ai`
- `SARVAM_MODEL` default: `sarvam-m`
- `POLICY_DATASET_PATH` optional path override for the spreadsheet
- `FALLBACK_THRESHOLD` default: `0.25`
- `LOG_PATH` default: `logs/requests.jsonl`
- `CORS_ORIGINS` default: `http://localhost:5173`

## Notes

- If `SARVAM_API_KEY` is not set, the backend uses an offline fallback response generator so the app still runs for local testing.
- The backend writes JSONL logs containing the query, retrieved docs, prompt, and generation parameters.

