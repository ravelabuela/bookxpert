from src.chunking import chunk_pages
from src.documents import DocumentPage


def test_chunks_keep_document_metadata_and_overlap() -> None:
    text = " ".join(
        f"Sentence {number} has enough words to form useful context."
        for number in range(45)
    )
    chunks = chunk_pages(
        [DocumentPage(text, "source.txt", 1)], target_words=80, overlap_words=20
    )
    assert len(chunks) > 1
    assert all(chunk.source == "source.txt" and chunk.page == 1 for chunk in chunks)
