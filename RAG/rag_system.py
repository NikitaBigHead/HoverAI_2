import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ollama import AsyncClient

from db import RagDatabase
from embeddings import LocalTextEmbedder
from repository import KnowledgeRepository

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@dataclass
class LocalLlmConfig:
    model: str = "qwen3.5:9b"
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    context_length: int = 10
    system_message: str | None = None
    knowledge_file: str = "knowledge.txt"
    sqlite_db_path: str = "knowledge.db"


@dataclass
class AppConfig:
    llm: LocalLlmConfig = field(default_factory=LocalLlmConfig)


@dataclass
class RetrievalResult:
    chunk: str
    score: float
    title: str | None = None
    source_url: str | None = None
    image_url: str | None = None
    image_caption: str | None = None


@dataclass
class AnswerPayload:
    answer: str
    title: str | None = None
    source_url: str | None = None
    image_url: str | None = None
    image_caption: str | None = None
    retrieval_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "title": self.title,
            "source": {
                "url": self.source_url,
            },
            "image": {
                "url": self.image_url,
                "caption": self.image_caption,
            },
            "retrieval": {
                "score": self.retrieval_score,
            },
        }

    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class GemmaChatContext:
    USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
    MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.history: list[str] = []

    def add_user_query(self, query: str) -> None:
        self.history.append(self.USER_CHAT_TEMPLATE.format(prompt=query))

    def add_model_response(self, resp: str) -> None:
        self.history.append(self.MODEL_CHAT_TEMPLATE.format(prompt=resp))

    def reset(self) -> None:
        self.history = []

    def generate_prompt(self, knowledge_context: str | None = None) -> str:
        ctx_len = self.cfg.llm.context_length
        if ctx_len > 0:
            self.history = self.history[-ctx_len * 2 :]
        else:
            self.history = self.history[-1:]

        context = "".join(self.history)
        system_message = (self.cfg.llm.system_message or "").strip()
        sys_prompt = ""
        if system_message:
            sys_prompt = self.USER_CHAT_TEMPLATE.format(prompt=system_message)

        knowledge_context = knowledge_context or ""
        return sys_prompt + context + f"context: {knowledge_context}\n" + "<start_of_turn>model\n"


class RetrievalScorer:
    TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)

    @classmethod
    def tokenize(cls, value: str) -> set[str]:
        return {token.lower() for token in cls.TOKEN_RE.findall(value or "") if len(token) > 1}

    @classmethod
    def score(cls, query: str, title: str, content: str, vector_score: float) -> float:
        query_tokens = cls.tokenize(query)
        title_tokens = cls.tokenize(title)
        content_tokens = cls.tokenize(content)

        if not query_tokens:
            return float(vector_score)

        title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
        content_overlap = len(query_tokens & content_tokens) / len(query_tokens)
        exact_title_bonus = 0.15 if query.lower() in (title or '').lower() else 0.0
        return float(vector_score) + (0.35 * title_overlap) + (0.2 * content_overlap) + exact_title_bonus


class VectorKnowledgeBase:
    def __init__(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            self.chunks = [line.strip() for line in f if line.strip()]
        self.embedder = LocalTextEmbedder()
        print("🧠 Encoding text knowledge base...")
        self.chunk_embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        print(f"✅ Text knowledge base ready: {len(self.chunks)} chunks loaded.")

    def retrieve(self, query: str) -> RetrievalResult:
        query_emb = self.embedder.encode(query, convert_to_numpy=True)
        sims = np.dot(self.chunk_embeddings, query_emb) / (
            np.maximum(
                np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_emb),
                1e-12,
            )
        )
        best_idx = int(np.argmax(sims))
        return RetrievalResult(chunk=self.chunks[best_idx], score=float(sims[best_idx]))


class SQLiteKnowledgeBase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db = RagDatabase(str(self.db_path))
        self.repo = KnowledgeRepository(self.db)
        self.embedder = LocalTextEmbedder()
        self.records: list[dict] = []
        self.chunk_embeddings = np.array([])
        self.reload()

    def reload(self) -> None:
        if not self.db_path.exists():
            self.records = []
            self.chunk_embeddings = np.array([])
            print(f"⚠️ SQLite knowledge DB not found: {self.db_path}")
            return

        self.records = self.repo.fetch_all_chunks()
        if self.records:
            self.chunk_embeddings = np.array(
                [json.loads(record["embedding_json"]) for record in self.records],
                dtype=np.float32,
            )
        else:
            self.chunk_embeddings = np.array([])
        print(f"✅ SQLite knowledge base ready: {len(self.records)} chunks loaded.")

    def retrieve(self, query: str) -> RetrievalResult:
        if not self.records or self.chunk_embeddings.size == 0:
            return RetrievalResult(chunk="", score=0.0)

        query_emb = np.array(self.embedder.encode(query, convert_to_numpy=True), dtype=np.float32)
        vector_scores = np.dot(self.chunk_embeddings, query_emb) / (
            np.maximum(
                np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(query_emb),
                1e-12,
            )
        )

        scored_candidates = []
        for idx, record in enumerate(self.records):
            blended_score = RetrievalScorer.score(
                query=query,
                title=record.get("title") or "",
                content=record.get("content") or "",
                vector_score=float(vector_scores[idx]),
            )
            scored_candidates.append((blended_score, idx, float(vector_scores[idx])))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_idx, vector_score = scored_candidates[0]
        record = self.records[best_idx]
        context_parts = [
            f"title: {record['title']}",
            f"content: {record['content']}",
        ]
        if record.get("summary"):
            context_parts.append(f"summary: {record['summary']}")
        if record.get("source_url"):
            context_parts.append(f"source_url: {record['source_url']}")
        if record.get("image_url"):
            context_parts.append(f"image_url: {record['image_url']}")
        return RetrievalResult(
            chunk="\n".join(context_parts),
            score=best_score,
            title=record.get("title"),
            source_url=record.get("source_url"),
            image_url=record.get("image_url") or record.get("image_source_page"),
            image_caption=record.get("image_caption"),
        )


class RAGBot:
    def __init__(self, cfg: AppConfig, llm: AsyncClient):
        self.cfg = cfg
        self.llm = llm
        sqlite_db_path = Path(cfg.llm.sqlite_db_path)
        if sqlite_db_path.exists():
            self.knowledge_base = SQLiteKnowledgeBase(str(sqlite_db_path))
        else:
            self.knowledge_base = VectorKnowledgeBase(cfg.llm.knowledge_file)
        self.chat_context = GemmaChatContext(cfg)

    async def process_query(self, query: str) -> AnswerPayload:
        self.chat_context.add_user_query(query)
        retrieval = self.knowledge_base.retrieve(query)
        print(f"RAG similarity: {retrieval.score:.3f}")
        answer = await self.generate_answer_with_context(retrieval.chunk)
        self.chat_context.add_model_response(answer)
        return AnswerPayload(
            answer=answer,
            title=retrieval.title,
            source_url=retrieval.source_url,
            image_url=retrieval.image_url,
            image_caption=retrieval.image_caption,
            retrieval_score=retrieval.score,
        )

    async def generate_answer_with_context(self, knowledge_context: str) -> str:
        prompt = self.chat_context.generate_prompt(knowledge_context)
        resp = await self.llm.generate(
            model=self.cfg.llm.model,
            prompt=prompt,
            options={
                "temperature": self.cfg.llm.temperature,
                "top_k": self.cfg.llm.top_k,
                "top_p": self.cfg.llm.top_p,
            },
        )
        return resp.get("response", "").strip()


async def run_interactive_mode(bot: RAGBot) -> None:
    print("🧠 RAGBot ready!")
    print("SQLite source will be used automatically if knowledge.db exists.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("💬 You: ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break
            payload = await bot.process_query(query)
            print(f"🤖 Bot: {payload.answer}\n")
            if payload.source_url:
                print(f"🔗 Source: {payload.source_url}")
            if payload.image_url:
                print(f"🖼 Image: {payload.image_url}")
            print()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(
        "Run `python ingest.py` to create knowledge.db, then instantiate RAGBot with AppConfig and AsyncClient."
    )
