"""Local document loading with page-aware metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


@dataclass(frozen=True)
class DocumentPage:
    text: str
    source: str
    page: int


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove common standalone page-number lines."""
    text = text.replace("\u00ad", "")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not re.fullmatch(r"(?:page\s*)?\d+", line, re.I)]
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def load_documents(data_dir: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    paths = sorted(path for path in data_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No supported documents found in {data_dir}")

    for path in paths:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            for number, pdf_page in enumerate(reader.pages, start=1):
                text = _clean_text(pdf_page.extract_text() or "")
                if text:
                    pages.append(DocumentPage(text, path.name, number))
        else:
            text = _clean_text(path.read_text(encoding="utf-8"))
            if text:
                pages.append(DocumentPage(text, path.name, 1))
    return pages

