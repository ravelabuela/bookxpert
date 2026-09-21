"""Build and persist the FAISS vector index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import chunk_pages
from .config import get_settings
from .documents import load_documents


def build_index(data_dir: Path, storage_dir: Path, model_name: str, batch_size: int) -> int:
    pages = load_documents(data_dir)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No readable text was found in the knowledge base.")

    print(f"Loaded {len(pages)} page(s) from {len({page.source for page in pages})} document(s).")
    print(f"Created {len(chunks)} sentence-aware chunks with overlap.")
    model = SentenceTransformer(model_name)
    # One batched call, rather than an embedding request for each chunk.
    vectors = model.encode(
        [chunk.text for chunk in chunks], batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    storage_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(storage_dir / "index.faiss"))
    metadata = {
        "embedding_model": model_name,
        "chunk_count": len(chunks),
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
    (storage_dir / "chunks.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved FAISS index and metadata to {storage_dir}")
    return len(chunks)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index local documents for Document QA Bot.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--storage-dir", type=Path, default=settings.storage_dir)
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    args = parser.parse_args()
    build_index(args.data_dir, args.storage_dir, settings.embedding_model, args.batch_size)


if __name__ == "__main__":
    main()

