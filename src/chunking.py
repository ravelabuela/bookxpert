"""Sentence-aware overlapping chunk construction."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import DocumentPage


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def chunk_pages(pages: list[DocumentPage], target_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    """Keep sentences together, then overlap the final words into the next chunk."""
    chunks: list[Chunk] = []
    for page in pages:
        current: list[str] = []
        count = 0
        for sentence in _sentences(page.text):
            words = sentence.split()
            if current and count + len(words) > target_words:
                chunks.append(Chunk(f"chunk-{len(chunks):04d}", " ".join(current), page.source, page.page))
                trailing: list[str] = []
                trailing_count = 0
                for previous in reversed(current):
                    trailing.insert(0, previous)
                    trailing_count += len(previous.split())
                    if trailing_count >= overlap_words:
                        break
                current = trailing
                count = sum(len(item.split()) for item in current)
            current.append(sentence)
            count += len(words)
        if current:
            chunks.append(Chunk(f"chunk-{len(chunks):04d}", " ".join(current), page.source, page.page))
    return chunks

