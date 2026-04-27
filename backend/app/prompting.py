from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievedDocument


STRICT_DEFAULTS = {"temperature": 0.2, "max_tokens": 150}
FRIENDLY_DEFAULTS = {"temperature": 0.7, "max_tokens": 200}


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


def mode_defaults(mode: str) -> dict[str, float | int]:
    if mode == "friendly":
        return FRIENDLY_DEFAULTS
    return STRICT_DEFAULTS


def format_context(docs: list[RetrievedDocument]) -> str:
    lines: list[str] = []
    for index, doc in enumerate(docs, start=1):
        lines.append(
            f"[{index}] Title: {doc.title} (chunk {doc.chunk_index}/{doc.chunk_count})\n"
            f"Source ID: {doc.source_id}\n"
            f"Category: {doc.category}\n"
            f"Policy: {doc.company_response}\n"
            f"Solution: {doc.solution}\n"
            f"Alternate: {doc.alternate_solution}\n"
            f"Source text: {doc.content}"
        )
    return "\n\n".join(lines) if lines else "No policy context retrieved."


def build_prompt(mode: str, query: str, docs: list[RetrievedDocument]) -> PromptBundle:
    context = format_context(docs)
    if mode == "friendly":
        system = (
            "You are a polite and empathetic customer support assistant. "
            "Use only the policy context provided. "
            "Stay accurate, avoid inventing rules, and keep the tone warm."
        )
        user = (
            "Use the retrieved policy context to answer the customer. "
            "If the policy does not clearly answer the issue, ask for escalation.\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Customer issue:\n{query}\n\n"
            "Write a short helpful reply."
        )
        return PromptBundle(system=system, user=user)

    system = (
        "You are a professional customer support assistant. "
        "Use ONLY the provided policy context. "
        "Do not add assumptions, and keep the answer concise and direct."
    )
    user = (
        "Answer the customer using the policy context only.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Customer issue:\n{query}\n\n"
        "Return a clear, policy-based reply."
    )
    return PromptBundle(system=system, user=user)


def fallback_prompt() -> PromptBundle:
    return PromptBundle(
        system="No relevant policy found.",
        user='Respond with: "Please escalate this issue to a human support agent."',
    )
