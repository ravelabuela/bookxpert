"""Retrieval and strictly grounded OpenAI answer generation."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int
    score: float


class DocumentQA:
    def __init__(self, storage_dir: Path, embedding_model: str, chat_model: str, llm_provider: str, ollama_base_url: str, ollama_model: str) -> None:
        index_path, metadata_path = storage_dir / "index.faiss", storage_dir / "chunks.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Index not found. Run `python -m src.indexer` first.")
        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata["embedding_model"] != embedding_model:
            raise ValueError("Embedding model differs from the saved index. Re-run the indexer.")
        self.embedder = SentenceTransformer(embedding_model)
        self.chat_model = chat_model
        self.llm_provider = llm_provider
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

    def search(self, question: str, top_k: int) -> list[SearchResult]:
        vector = self.embedder.encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(vector, dtype="float32"), top_k)
        results: list[SearchResult] = []
        for score, item_id in zip(scores[0], ids[0]):
            if item_id != -1:
                item = self.metadata["chunks"][int(item_id)]
                results.append(SearchResult(item["text"], item["source"], item["page"], float(score)))
        return results

    def answer(self, question: str, results: list[SearchResult]) -> str:
                # Refuse before generation when no retrieved passage is relevant enough.
        if not results or results[0].score < 0.20:
            return "I don't know based on the provided documents."
        context = "\n\n".join(
            f"[Source {i}: {item.source}, page {item.page}]\n{item.text}"
            for i, item in enumerate(results, start=1)
        )
        prompt = f"""Answer the user's question using only the supplied context.
If the context does not contain enough evidence, reply exactly: "I don't know based on the provided documents."
Do not use outside knowledge. Write at most two short sentences. Do not add facts that are not explicitly stated in the context. Include an inline citation after each sentence in exactly this form: [filename, p. N].

Context:
{context}

Question: {question}"""
        if self.llm_provider == "ollama":
            return self._format_local_answer(self._answer_with_ollama(prompt), results)
        if self.llm_provider != "openai":
            return "Invalid LLM_PROVIDER. Set it to `ollama` or `openai` in .env."
        client = OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.chat_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a careful, evidence-bound document assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        except AuthenticationError:
            return (
                "Answer generation could not run because OPENAI_API_KEY is invalid. "
                "Create a new API key, put it in .env, and restart the application."
            )
        except RateLimitError:
            return (
                "Answer generation could not run because the OpenAI API account has no "
                "available credits or has reached its rate limit. Add API billing credits, then retry. "
                "The retrieved evidence is shown below."
            )
        except (APIConnectionError, APIStatusError) as error:
            return f"Answer generation is temporarily unavailable ({type(error).__name__}). The retrieved evidence is shown below."
        return self._ensure_citations(
            response.choices[0].message.content or "I don't know based on the provided documents.",
            results,
        )

    @staticmethod
    def _ensure_citations(answer: str, results: list[SearchResult]) -> str:
        """Keep a verifiable source trail even when a small local model omits citations."""
        if answer.startswith("I don't know") or "[" in answer:
            return answer
        citations: list[str] = []
        for result in results:
            citation = f"[{result.source}, p. {result.page}]"
            if citation not in citations:
                citations.append(citation)
        return f"{answer}\n\nSources: {'; '.join(citations)}"

    @staticmethod
    def _format_local_answer(answer: str, results: list[SearchResult]) -> str:
        """Constrain small local models that ignore answer-length instructions."""
        unavailable_prefixes = ("Ollama is unavailable", "Ollama could not", "I don't know")
        if answer.startswith(unavailable_prefixes) or not results:
            return answer
        answer_without_sources = answer.split("\nSources:", 1)[0].strip()
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer_without_sources)
            if sentence.strip()
        ]
        concise_answer = " ".join(sentences[:2])
        citation = f"[{results[0].source}, p. {results[0].page}]"
        return f"{concise_answer}\n\nSource: {citation}"

    def _answer_with_ollama(self, prompt: str) -> str:
        request_body = json.dumps({"model": self.ollama_model, "prompt": prompt, "system": "You are a careful, evidence-bound document assistant.", "stream": False, "options": {"temperature": 0}}).encode("utf-8")
        request = urllib.request.Request(f"{self.ollama_base_url}/api/generate", data=request_body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                answer = json.loads(response.read().decode("utf-8")).get("response", "").strip()
                return answer or "I don't know based on the provided documents."
        except urllib.error.URLError:
            return f"Ollama is unavailable. Install and start Ollama, then run `ollama pull {self.ollama_model}`. The retrieved evidence is shown below."
        except (urllib.error.HTTPError, TimeoutError) as error:
            return f"Ollama could not generate an answer ({type(error).__name__}). The retrieved evidence is shown below."
