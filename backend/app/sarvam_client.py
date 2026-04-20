from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMResult:
    content: str
    raw_response: dict[str, Any]


class SarvamClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def provider_name(self) -> str:
        return "sarvam"

    def generate(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "api-subscription-key": self.api_key,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        return LLMResult(content=content, raw_response=data)


class OfflineFallbackClient:
    @property
    def provider_name(self) -> str:
        return "offline-fallback"

    def generate(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> LLMResult:
        system_message = next((item["content"] for item in messages if item["role"] == "system"), "")
        user_message = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
        content = self._summarize_context(system_message, user_message)
        return LLMResult(
            content=content,
            raw_response={
                "provider": self.provider_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    def _summarize_context(self, system_message: str, user_message: str) -> str:
        if "Please escalate this issue to a human support agent." in user_message:
            return "Please escalate this issue to a human support agent."
        company_response = ""
        solution = ""
        for block in re.split(r"\n\s*\n", user_message):
            if not company_response and "Company Response:" in block:
                company_response = self._extract_field(block, "Company Response")
            if not solution and "Solution:" in block:
                solution = self._extract_field(block, "Solution")
            if company_response and solution:
                break

        reply = company_response or solution or "Please escalate this issue to a human support agent."
        if reply == "Please escalate this issue to a human support agent.":
            return reply
        if "empathetic" in system_message.lower() or "friendly" in system_message.lower():
            return f"Thanks for reaching out. {reply}"
        return reply

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        match = re.search(rf"{re.escape(field_name)}:\s*(.*?)(?:\s*\|\s*|\s*$)", text)
        return match.group(1).strip() if match else ""
