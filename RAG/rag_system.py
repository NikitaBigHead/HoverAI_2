import asyncio
import json
import os
import re
import time
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
    image_local_path: str | None = None
    image_caption: str | None = None
    verification_status: str | None = None
    verified_source_url: str | None = None


@dataclass
class AnswerPayload:
    answer: str
    title: str | None = None
    source_url: str | None = None
    image_url: str | None = None
    image_local_path: str | None = None
    image_caption: str | None = None
    verification_status: str | None = None
    verified_source_url: str | None = None
    retrieval_score: float = 0.0
    retrieval_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "title": self.title,
            "source": {
                "url": self.source_url,
                "verified_url": self.verified_source_url,
            },
            "image": {
                "url": self.image_url,
                "local_path": self.image_local_path,
                "caption": self.image_caption,
            },
            "retrieval": {
                "score": self.retrieval_score,
                "time_ms": self.retrieval_time_ms,
                "verification_status": self.verification_status,
            },
            "inference": {
                "time_ms": self.inference_time_ms,
            },
            "timing": {
                "total_time_ms": self.total_time_ms,
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
    VERIFICATION_WEIGHTS = {
        "confirmed": 0.12,
        "partially_confirmed": 0.03,
        "contradicted_fixed": -0.04,
        "unverified": -0.08,
    }

    @classmethod
    def tokenize(cls, value: str) -> set[str]:
        return {token.lower() for token in cls.TOKEN_RE.findall(value or "") if len(token) > 1}

    @classmethod
    def verification_bonus(cls, verification_status: str | None) -> float:
        return cls.VERIFICATION_WEIGHTS.get((verification_status or "").strip(), 0.0)

    @classmethod
    def score(
        cls,
        query: str,
        title: str,
        content: str,
        vector_score: float,
        verification_status: str | None = None,
    ) -> float:
        query_tokens = cls.tokenize(query)
        title_tokens = cls.tokenize(title)
        content_tokens = cls.tokenize(content)

        if not query_tokens:
            return float(vector_score) + cls.verification_bonus(verification_status)

        title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
        content_overlap = len(query_tokens & content_tokens) / len(query_tokens)
        normalized_query = " ".join(sorted(query_tokens))
        normalized_title = " ".join(sorted(title_tokens))
        exact_title_bonus = 0.2 if (query.lower() in (title or '').lower() or (title or '').lower() in query.lower()) else 0.0
        near_title_bonus = 0.18 if normalized_query == normalized_title and normalized_query else 0.0
        return (
            float(vector_score)
            + (0.45 * title_overlap)
            + (0.18 * content_overlap)
            + exact_title_bonus
            + near_title_bonus
            + cls.verification_bonus(verification_status)
        )


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
            metadata = json.loads(record.get("metadata_json") or "{}")
            verification_status = metadata.get("verification_status")
            blended_score = RetrievalScorer.score(
                query=query,
                title=record.get("title") or "",
                content=record.get("content") or "",
                vector_score=float(vector_scores[idx]),
                verification_status=verification_status,
            )
            scored_candidates.append((blended_score, idx, float(vector_scores[idx]), verification_status, metadata))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_idx, vector_score, verification_status, metadata = scored_candidates[0]
        record = self.records[best_idx]
        context_parts = [
            f"title: {record['title']}",
            f"content: {record['content']}",
        ]
        if record.get("summary"):
            context_parts.append(f"summary: {record['summary']}")
        if record.get("source_url"):
            context_parts.append(f"source_url: {record['source_url']}")
        if metadata.get("verified_source_url"):
            context_parts.append(f"verified_source_url: {metadata['verified_source_url']}")
        if verification_status:
            context_parts.append(f"verification_status: {verification_status}")
        if record.get("image_url"):
            context_parts.append(f"image_url: {record['image_url']}")
        return RetrievalResult(
            chunk="\n".join(context_parts),
            score=best_score,
            title=record.get("title"),
            source_url=record.get("source_url"),
            image_url=record.get("image_url") or record.get("image_source_page"),
            image_local_path=record.get("image_local_path"),
            image_caption=record.get("image_caption"),
            verification_status=verification_status,
            verified_source_url=metadata.get("verified_source_url"),
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
        total_started = time.perf_counter()
        self.chat_context.add_user_query(query)

        retrieval_started = time.perf_counter()
        retrieval = self.knowledge_base.retrieve(query)
        retrieval_time_ms = (time.perf_counter() - retrieval_started) * 1000.0
        print(f"RAG similarity: {retrieval.score:.3f}")

        inference_started = time.perf_counter()
        answer = await self.generate_answer_with_context(retrieval.chunk)
        inference_time_ms = (time.perf_counter() - inference_started) * 1000.0
        self.chat_context.add_model_response(answer)
        total_time_ms = (time.perf_counter() - total_started) * 1000.0

        print(
            "Timing: "
            f"retrieval={retrieval_time_ms:.0f} ms, "
            f"inference={inference_time_ms:.0f} ms, "
            f"total={total_time_ms:.0f} ms"
        )

        return AnswerPayload(
            answer=answer,
            title=retrieval.title,
            source_url=retrieval.source_url,
            image_url=retrieval.image_url,
            image_local_path=retrieval.image_local_path,
            image_caption=retrieval.image_caption,
            verification_status=retrieval.verification_status,
            verified_source_url=retrieval.verified_source_url,
            retrieval_score=retrieval.score,
            retrieval_time_ms=retrieval_time_ms,
            inference_time_ms=inference_time_ms,
            total_time_ms=total_time_ms,
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
