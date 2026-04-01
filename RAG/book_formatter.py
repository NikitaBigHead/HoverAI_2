from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from rag_system import AnswerPayload


class UiSanitizer:
    LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
    ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)")
    INLINE_CODE_RE = re.compile(r"`([^`]+)`")
    MULTISPACE_RE = re.compile(r"[ \t]+")
    MULTINEWLINE_RE = re.compile(r"\n{3,}")

    @classmethod
    def sanitize_text(cls, value: str | None) -> str:
        text = value or ""
        text = cls.LINK_RE.sub(r"\1", text)
        text = cls.BOLD_RE.sub(r"\1", text)
        text = cls.ITALIC_RE.sub(r"\1", text)
        text = cls.INLINE_CODE_RE.sub(r"\1", text)
        text = text.replace("# ", "")
        text = text.replace("## ", "")
        text = text.replace("### ", "")
        text = ''.join(ch for ch in text if not cls._is_symbol_emoji(ch))
        text = text.replace("\r", "")
        text = cls.MULTISPACE_RE.sub(" ", text)
        text = cls.MULTINEWLINE_RE.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def _is_symbol_emoji(ch: str) -> bool:
        if not ch:
            return False
        category = unicodedata.category(ch)
        return category == "So"


@dataclass
class BookPage:
    page_number: int
    title: str | None
    body: str
    image_url: str | None = None
    image_local_path: str | None = None
    image_caption: str | None = None
    source_url: str | None = None
    footer: str | None = None
    page_type: str = "text"

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "title": self.title,
            "body": self.body,
            "image": {
                "url": self.image_url,
                "local_path": self.image_local_path,
                "caption": self.image_caption,
            },
            "source": {
                "url": self.source_url,
            },
            "footer": self.footer,
            "page_type": self.page_type,
        }


@dataclass
class BookDocument:
    document_title: str
    pages: list[BookPage] = field(default_factory=list)
    source_url: str | None = None
    primary_image_url: str | None = None

    def to_dict(self) -> dict:
        return {
            "document_title": self.document_title,
            "source_url": self.source_url,
            "primary_image_url": self.primary_image_url,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class BookFormatter:
    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars_per_page: int = 900):
        self.max_chars_per_page = max_chars_per_page

    def build_document(self, payload: AnswerPayload) -> BookDocument:
        title = UiSanitizer.sanitize_text(payload.title) or "HoverAI Response"
        pages: list[BookPage] = []

        clean_answer = UiSanitizer.sanitize_text(payload.answer)
        clean_caption = UiSanitizer.sanitize_text(payload.image_caption)
        cover_body = clean_answer[: min(len(clean_answer), 220)].strip()
        pages.append(
            BookPage(
                page_number=1,
                title=title,
                body=cover_body,
                image_url=payload.image_url,
                image_caption=clean_caption,
                source_url=payload.source_url,
                footer="Cover",
                page_type="cover",
            )
        )

        text_pages = self._build_text_pages(
            title=title,
            text=clean_answer,
            source_url=payload.source_url,
            start_page_number=len(pages) + 1,
        )
        pages.extend(text_pages)

        if payload.image_url:
            pages.append(
                BookPage(
                    page_number=len(pages) + 1,
                    title="Illustration",
                    body=clean_caption or "Reference image",
                    image_url=payload.image_url,
                    image_caption=clean_caption,
                    source_url=payload.source_url,
                    footer="Image",
                    page_type="image",
                )
            )

        if payload.source_url:
            pages.append(
                BookPage(
                    page_number=len(pages) + 1,
                    title="Source",
                    body=payload.source_url,
                    source_url=payload.source_url,
                    footer="Reference",
                    page_type="source",
                )
            )

        return BookDocument(
            document_title=title,
            pages=pages,
            source_url=payload.source_url,
            primary_image_url=payload.image_url,
        )

    def _build_text_pages(
        self,
        title: str,
        text: str,
        source_url: str | None,
        start_page_number: int,
    ) -> list[BookPage]:
        sentences = [part.strip() for part in self.SENTENCE_RE.split(text or "") if part.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self.max_chars_per_page:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)

        return [
            BookPage(
                page_number=start_page_number + index,
                title=title if index == 0 else None,
                body=chunk,
                source_url=source_url,
                footer="Answer",
                page_type="text",
            )
            for index, chunk in enumerate(chunks)
        ]
