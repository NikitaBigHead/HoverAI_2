import sqlite3
from pathlib import Path


class RagDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE,
                    entity_type TEXT,
                    title TEXT NOT NULL,
                    language TEXT,
                    text TEXT NOT NULL,
                    summary TEXT,
                    source_name TEXT,
                    source_url TEXT,
                    source_domain TEXT,
                    source_type TEXT,
                    collected_at TEXT,
                    last_seen_at TEXT,
                    section TEXT,
                    tags_json TEXT,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id),
                    UNIQUE(document_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    image_url TEXT,
                    caption TEXT,
                    source_page TEXT,
                    local_path TEXT,
                    content_hash TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );
                """
            )
