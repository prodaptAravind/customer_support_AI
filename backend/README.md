# Backend

FastAPI service for the AI-assisted customer support response generator.

## What it does

- Loads the local `Complaint Dataset.xlsx` file
- Retrieves the top policy matches with BM25
- Builds strict or friendly prompts
- Calls Sarvam chat completions when `SARVAM_API_KEY` is set
- Falls back to a human-escalation response when no good match is found
- Logs every request in JSONL format

## Run

```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

## Environment

- `SARVAM_API_KEY`
- `SARVAM_BASE_URL` default: `https://api.sarvam.ai`
- `SARVAM_MODEL` default: `sarvam-m`
- `POLICY_DATASET_PATH` optional override for the spreadsheet path
- `FALLBACK_THRESHOLD` default: `0.25`
- `LOG_PATH` default: `logs/requests.jsonl`

