from book_formatter import BookFormatter
from rag_system import AnswerPayload
from ui.book_viewer import launch_viewer


def main() -> int:
    payload = AnswerPayload(
        answer=(
            "Commencement 2025 will take place on June 27, 2025 at 16:00. "
            "The ceremony will be held in the Skoltech Main Hall. "
            "Guests should arrive early for seating and registration. "
            "This response is shown in a book-like Linux viewer prototype."
        ),
        title="Commencement 2025",
        source_url="https://events.skoltech.ru/commencement2025",
        image_url="https://commons.wikimedia.org/wiki/File:Skoltech_University_09.jpg",
        image_caption="Photo related to Commencement 2025",
        retrieval_score=0.3269,
    )

    document = BookFormatter(max_chars_per_page=500).build_document(payload)
    return launch_viewer(document)


if __name__ == "__main__":
    raise SystemExit(main())
