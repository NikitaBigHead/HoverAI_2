import argparse
import asyncio

from book_formatter import BookFormatter
from rag_system import AppConfig, RAGBot
from ui.book_viewer import launch_viewer
from ollama import AsyncClient


async def build_document_from_query(query: str):
    cfg = AppConfig()
    llm = AsyncClient()
    bot = RAGBot(cfg, llm)
    payload = await bot.process_query(query)
    document = BookFormatter(max_chars_per_page=500).build_document(payload)
    return document, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGBot and open the answer in the book viewer")
    parser.add_argument("query", nargs="?", help="Question for the local RAG bot")
    args = parser.parse_args()

    query = args.query or input("Question for HoverAI: ").strip()
    if not query:
        raise SystemExit("Empty query")

    document, payload = asyncio.run(build_document_from_query(query))
    print(payload.to_pretty_json())
    return launch_viewer(document)


if __name__ == "__main__":
    raise SystemExit(main())
