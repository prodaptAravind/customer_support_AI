from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    dataset_path: Path
    sarvam_api_key: str | None
    sarvam_base_url: str
    sarvam_model: str
    log_path: Path
    fallback_threshold: float
    cors_origins: list[str]


def _default_dataset_path() -> Path:
    env_path = os.getenv("POLICY_DATASET_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "Complaint Dataset.xlsx"


def load_config() -> AppConfig:
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    return AppConfig(
        dataset_path=_default_dataset_path(),
        sarvam_api_key=os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_SUBSCRIPTION_KEY"),
        sarvam_base_url=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai"),
        sarvam_model=os.getenv("SARVAM_MODEL", "sarvam-m"),
        log_path=Path(os.getenv("LOG_PATH", Path(__file__).resolve().parents[2] / "logs" / "requests.jsonl")),
        fallback_threshold=float(os.getenv("FALLBACK_THRESHOLD", "0.25")),
        cors_origins=cors_origins,
    )

