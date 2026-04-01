import json
from typing import Any

from db import RagDatabase


class KnowledgeRepository:
    def __init__(self, db: RagDatabase):
        self.db = db

    def upsert_document(self, item: dict[str, Any]) -> int:
        with self.db.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM documents WHERE external_id = ?",
                (item["id"],),
            ).fetchone()

            values = (
                item["id"],
                item.get("entity_type"),
                item.get("title") or item["id"],
                item.get("language"),
                item.get("text") or "",
                item.get("summary"),
                item.get("source_name"),
                item.get("source_url"),
                item.get("source_domain"),
                item.get("source_type"),
                item.get("collected_at"),
                item.get("last_seen_at"),
                item.get("section"),
                json.dumps(item.get("tags", []), ensure_ascii=False),
                json.dumps(item.get("metadata", {}), ensure_ascii=False),
            )

            if existing:
                conn.execute(
                    """
                    UPDATE documents
                    SET entity_type = ?, title = ?, language = ?, text = ?, summary = ?,
                        source_name = ?, source_url = ?, source_domain = ?, source_type = ?,
                        collected_at = ?, last_seen_at = ?, section = ?, tags_json = ?, metadata_json = ?
                    WHERE external_id = ?
                    """,
                    values[1:] + (values[0],),
                )
                return int(existing["id"])

            cursor = conn.execute(
                """
                INSERT INTO documents (
                    external_id, entity_type, title, language, text, summary,
                    source_name, source_url, source_domain, source_type,
                    collected_at, last_seen_at, section, tags_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return int(cursor.lastrowid)

    def replace_chunks(
        self,
        document_id: int,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM text_chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO text_chunks (document_id, chunk_index, content, embedding_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        index,
                        chunk,
                        json.dumps(embedding, ensure_ascii=False),
                    )
                    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ],
            )

    def replace_images(self, document_id: int, images: list[dict[str, Any]]) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM images WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO images (
                    document_id, image_url, caption, source_page, local_path, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        image.get("image_url"),
                        image.get("caption"),
                        image.get("source_page"),
                        image.get("local_path"),
                        image.get("content_hash"),
                    )
                    for image in images
                ],
            )

    def fetch_all_chunks(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    tc.id AS chunk_id,
                    tc.document_id AS document_id,
                    tc.chunk_index AS chunk_index,
                    tc.content AS content,
                    tc.embedding_json AS embedding_json,
                    d.title AS title,
                    d.summary AS summary,
                    d.source_url AS source_url,
                    d.source_type AS source_type,
                    d.entity_type AS entity_type,
                    d.metadata_json AS metadata_json,
                    i.image_url AS image_url,
                    i.caption AS image_caption,
                    i.local_path AS image_local_path,
                    i.source_page AS image_source_page
                FROM text_chunks tc
                JOIN documents d ON d.id = tc.document_id
                LEFT JOIN images i ON i.document_id = d.id
                ORDER BY tc.document_id, tc.chunk_index
                """
            ).fetchall()
        return [dict(row) for row in rows]
