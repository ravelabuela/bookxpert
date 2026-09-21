# Grounded Document Q&A Bot

A command-line Retrieval-Augmented Generation (RAG) application that answers questions over a local document collection. It indexes five AI-engineering learning documents, retrieves the most relevant evidence, and uses an OpenAI model to compose a concise answer with page-level source citations. The bot is deliberately evidence-bound: when the retrieved documents do not support an answer, it declines rather than relying on model training data.

## Features

- Loads local `.pdf` and `.txt` documents while preserving filename and page metadata.
- Uses sentence-aware, overlapping chunks and batched local embeddings.
- Persists a FAISS similarity index to disk, keeping indexing separate from querying.
- Shows the final answer, inline citations, and the exact retrieved chunks with similarity scores.
- Provides an explicit, grounded fallback for unsupported questions.

## Tech stack

| Tool | Version | Role |
|---|---:|---|
| Python | 3.11+ | Application runtime |
| sentence-transformers | 5.3.0 | Local semantic embeddings |
| all-MiniLM-L6-v2 | model | English embedding model |
| FAISS CPU | 1.13.2 | Persisted vector search |
| Ollama | current desktop app | Default local grounded answer generation |
| OpenAI Python SDK | 2.29.0 | Optional cloud answer generation |
| Ollama `tinyllama` | local model | Default chat model |
| pypdf | 6.19.0 | PDF text extraction |
| python-dotenv | 1.2.2 | Environment-variable loading |

## Architecture

```text
data/*.pdf, data/*.txt
        |
  text extraction + source/page metadata
        |
  sentence-aware chunks (180 words, 40-word overlap)
        |
  all-MiniLM-L6-v2 embeddings (batched)
        |
  FAISS IndexFlatIP + chunks.json  <-- persisted in storage/
        |
question -> embedding -> top-k similarity search -> cited context
        |
OpenAI chat model (context-only instruction) -> answer + inline citations
```

## Document collection

The knowledge base under `data/` contains five non-trivial AI-engineering documents: four original text learning notes and one PDF assignment brief. Each text document exceeds 500 words. The PDF also exercises the required PDF ingestion path.

## Chunking strategy

The indexer uses sentence-aware chunks with a target size of 180 words and a 40-word trailing overlap. Maintaining sentence boundaries avoids severing a claim midway through a sentence; overlap carries definitions and qualifications across boundaries. The sizes are intentionally modest so a top-k result set can retain several independent pieces of evidence in the chat prompt.

## Embedding model and vector database

`sentence-transformers/all-MiniLM-L6-v2` produces normalized, local embeddings. It is compact and practical for a small English prototype. All chunks are passed to `model.encode()` as a batch (default batch size: 32), not embedded one by one.

FAISS `IndexFlatIP` stores those normalized vectors and performs exact inner-product search, equivalent to cosine-similarity ranking in this setup. The application saves `storage/index.faiss` and `storage/chunks.json`, so querying never silently rebuilds the index. Re-run indexing whenever a source document or embedding model changes.

## Setup

1. Clone the repository and enter it.

   ```bash
   git clone <YOUR_PUBLIC_REPOSITORY_URL>
   cd document-qa-bot
   ```

2. Create and activate a Python 3.11+ virtual environment.

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the environment file.

   ```bash
   cp .env.example .env
   ```

   Configure `.env` for the local Ollama option:

   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=tinyllama
   ```

   Install [Ollama for Windows](https://ollama.com/download/windows), open a new PowerShell terminal, and download the local answer model:

   ```powershell
   ollama pull tinyllama
   ```

   To use OpenAI instead, set `LLM_PROVIDER=openai` and add `OPENAI_API_KEY` to `.env`; an API account with available credits is required.

5. Build the vector index. The first run downloads the embedding model.

   ```bash
   python -m src.indexer
   ```

6. Start the interactive bot.

   ```bash
   python -m src.cli
   ```

   Change retrieval breadth if desired:

   ```bash
   python -m src.cli --top-k 5
   ```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `ollama` by default; set to `openai` to use the OpenAI API. |
| `OLLAMA_MODEL` | No | Local model; defaults to `tinyllama`. |
| `OLLAMA_BASE_URL` | No | Defaults to `http://localhost:11434`. |
| `OPENAI_API_KEY` | Only for OpenAI | API key used only when `LLM_PROVIDER=openai`. |
| `OPENAI_CHAT_MODEL` | No | Defaults to `gpt-4o-mini`. |
| `EMBEDDING_MODEL` | No | Defaults to `sentence-transformers/all-MiniLM-L6-v2`; re-index after changing it. |

## Example queries

1. `Why is chunk overlap useful in a RAG system?` - explains boundary context preservation.
2. `What makes cosine similarity available with an inner-product FAISS index?` - explains embedding normalization.
3. `How should a grounded bot respond when the evidence is insufficient?` - explains abstention.
4. `What should retrieval evaluation measure?` - describes recall at k and source inspection.
5. `Which technical components are required by the RAG assignment?` - cites the PDF assignment brief.
6. `What is tomorrow's weather in Hyderabad?` - demonstrates the required unsupported-question response.

## Project structure

```text
data/                 # Five source documents, including the PDF corpus item
src/documents.py      # PDF/TXT loading and cleanup
src/chunking.py       # Sentence-aware overlapping chunker
src/indexer.py        # Batched embeddings and persisted FAISS index
src/qa.py             # Similarity retrieval and evidence-bound LLM prompt
src/cli.py            # Interactive command-line interface
storage/              # Generated locally; deliberately excluded from Git
```

## Known limitations

- Retrieval quality is limited by the small general-purpose English embedding model and the size of the corpus.
- The system uses exact vector retrieval only; it has no keyword search, reranker, or metadata filtering.
- A model can still summarize evidence imperfectly, so users should verify important claims against the displayed chunks.
- PDF extraction is text-based; scanned PDFs need OCR, which this baseline intentionally does not include.
- The default Ollama model needs several GB of disk space and adequate system RAM. OpenAI is an alternative but requires API billing credits.
- If a configured LLM service is unavailable, the CLI shows a clear message and still prints the retrieved source chunks rather than terminating unexpectedly.

## Recording checklist

For the required 3-8 minute screen recording, show the project structure, run `python -m src.indexer`, then start `python -m src.cli`. Ask the first five example questions and one unsupported weather question. For each query, keep the answer and the printed source chunks visible. Explain that sentence-aware overlapping chunks preserve context and that FAISS persistence prevents unnecessary re-indexing.
