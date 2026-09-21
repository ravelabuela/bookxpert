from __future__ import annotations

import argparse
import os

from .config import get_settings
from .qa import DocumentQA


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Interactive Document QA Bot")
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    args = parser.parse_args()
    if settings.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
    bot = DocumentQA(settings.storage_dir, settings.embedding_model, settings.chat_model, settings.llm_provider, settings.ollama_base_url, settings.ollama_model)
    print(f"Document QA Bot ready ({settings.llm_provider}). Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue
        results = bot.search(question, args.top_k)
        print("\nAnswer:\n" + bot.answer(question, results))
        print("\nRetrieved source chunks:")
        for number, result in enumerate(results, 1):
            preview = result.text[:260] + ("..." if len(result.text) > 260 else "")
            print(f"{number}. [{result.source}, p. {result.page}; similarity {result.score:.3f}]\n   {preview}")


if __name__ == "__main__":
    main()
