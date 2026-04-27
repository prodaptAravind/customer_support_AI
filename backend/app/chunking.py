from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecursiveTextChunker:
    chunk_size: int = 500
    chunk_overlap: int = 80
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def split_text(self, text: str) -> list[str]:
        normalized = " ".join((text or "").split())
        if not normalized:
            return []
        fragments = self._recursive_fragments(normalized, self.separators)
        return self._merge_fragments(fragments)

    def _recursive_fragments(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text.strip()]

        separator = separators[0]
        parts = list(text) if separator == "" else text.split(separator)
        fragments: list[str] = []

        for part in parts:
            item = part.strip()
            if not item:
                continue
            if len(item) > self.chunk_size and len(separators) > 1:
                fragments.extend(self._recursive_fragments(item, separators[1:]))
            else:
                fragments.append(item)

        return fragments

    def _merge_fragments(self, fragments: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue

            prospective_length = current_length + len(fragment) + (1 if current else 0)
            if current and prospective_length > self.chunk_size:
                chunk = " ".join(current).strip()
                if chunk:
                    chunks.append(chunk)
                current = self._carry_overlap(chunk)
                current_length = len(" ".join(current))

            if len(fragment) > self.chunk_size:
                chunks.extend(self._force_split(fragment))
                current = []
                current_length = 0
                continue

            current.append(fragment)
            current_length = len(" ".join(current))

        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def _carry_overlap(self, chunk: str) -> list[str]:
        if self.chunk_overlap <= 0 or not chunk:
            return []
        tail = chunk[-self.chunk_overlap :].strip()
        return [tail] if tail else []

    def _force_split(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        step = max(self.chunk_size - self.chunk_overlap, 1)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start += step
        return chunks

