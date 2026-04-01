import argparse
import json
from pathlib import Path

from db import RagDatabase
from embeddings import LocalTextEmbedder
from repository import KnowledgeRepository


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def ingest_jsonl(
    dataset_path: str,
    db_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    db = RagDatabase(db_path)
    db.initialize()
    repo = KnowledgeRepository(db)
    embedder = LocalTextEmbedder()

    with Path(dataset_path).open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue

            item = json.loads(raw)
            document_id = repo.upsert_document(item)
            chunks = chunk_text(
                item.get("text", ""),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            embeddings = embedder.encode(chunks, convert_to_numpy=False) if chunks else []
            repo.replace_chunks(document_id, chunks, embeddings)
            repo.replace_images(document_id, item.get("images", []))
            print(
                f"[ingest] line={line_number} id={item.get('id')} "
                f"chunks={len(chunks)} images={len(item.get('images', []))}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL dataset into SQLite")
    parser.add_argument(
        "--dataset",
        default="skoltech_events_dataset_final_spaced.jsonl",
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--db",
        default="knowledge.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="Chunk overlap in characters",
    )
    args = parser.parse_args()

    ingest_jsonl(
        dataset_path=args.dataset,
        db_path=args.db,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
