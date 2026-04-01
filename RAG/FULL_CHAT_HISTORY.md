# Full Chat History

## 1. Initial RAG direction

We started by discussing how to evolve the RAG system toward the following target architecture:
- local LLM
- RAG with internet access
- local library of sources
- ability to return links to images from a local library or from the web
- Linux UI window that presents results like a PDF/book with page turning

## 2. Understanding the original `rag_system.py`

We analyzed the original `rag_system.py` and described its structure:
- `GemmaChatContext` for prompt/history management
- `VectorKnowledgeBase` for simple line-based retrieval from `knowledge.txt`
- later a web fallback was discussed conceptually
- `RAGBot` as the orchestration layer

We discussed limitations of the original file:
- one-line knowledge chunks
- top-1 retrieval only
- weak metadata handling
- no structured source output
- no image support
- no UI-oriented output structure

## 3. Concepts: normal document index and top-k retrieval

We discussed what a “normal document index” means:
- structured documents
- chunking
- metadata
- provenance
- links to source files and images

We also discussed `top-k retrieval` versus `top-1 retrieval`:
- `top-1`: return only the best chunk
- `top-k`: return several best chunks

For the current stage, we explicitly decided not to implement FAISS and not to move to top-k yet.

## 4. SQLite-only plan

You chose a simpler direction:
- no FAISS for now
- SQLite only

We then discussed a refactoring plan centered around:
- ingestion of local data into SQLite
- retrieval from SQLite
- structured payloads for UI

## 5. Skoltech dataset planning

We prepared ideas for a dataset specification for collecting open information about Skoltech.

### Agreed entity types
The dataset should include:
- general institute information
- educational programs
- research centers and labs
- faculty and key people
- campus and infrastructure
- news and events
- admissions and study FAQ
- images and their related pages
- official documents such as strategies, reports, brochures, PDFs

### Recommended sample structure
We discussed using `JSONL` as the base format for the Python collection pipeline.

Typical fields discussed included:
- `id`
- `entity_type`
- `title`
- `language`
- `text`
- `summary`
- `source_name`
- `source_url`
- `source_domain`
- `source_type`
- `collected_at`
- `last_seen_at`
- `section`
- `tags`
- `metadata`
- `images`

### Source priorities discussed
- official Skoltech website
- Skoltech subdomains for students/programs
- official Skoltech PDFs
- Wikipedia only as a secondary source
- official Skoltech news

## 6. Clarifying JSONL

We discussed what `JSONL` means:
- one JSON object per line
- useful for streaming, ingestion, and append-only processing

## 7. Evaluating dataset candidates

You asked whether the following files are suitable as a dataset:
- `skoltech_events_dataset_final_spaced.jsonl`
- `skoltech_events_final.txt`

We examined sample JSONL records.

### Conclusion
The JSONL file was accepted as:
- good enough for `dataset v0`
- suitable as raw material for ingestion

But not yet ideal as a final production-grade dataset because:
- some event descriptions were too short
- dates were not fully normalized
- some images were generic rather than directly tied to the event

Decision:
- accept the JSONL as `dataset v0`
- continue to implementation

## 8. Moving to implementation in the new repository

The working repository/path for implementation became:
- `/media/imit-learn/ISR_2T3/HoverAI_2/RAG`

We found the project was a smaller standalone module rather than the earlier larger server repo.

## 9. Implemented SQLite ingestion pipeline

We implemented the ingestion stage from `JSONL` into SQLite.

### Added files
- [db.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/db.py)
- [repository.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/repository.py)
- [embeddings.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/embeddings.py)
- [ingest.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/ingest.py)

### What they do
- `db.py` manages the SQLite database and schema
- `repository.py` manages documents, chunks, and images
- `embeddings.py` provides a lightweight local embedder using `HashingVectorizer`
- `ingest.py` reads the JSONL file, chunks text, computes embeddings, and stores everything in SQLite

### Database created
- [knowledge.db](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/knowledge.db)

We successfully ran ingestion and confirmed records were stored.

## 10. Why the embedder changed

Originally a `sentence-transformers` approach was attempted, but the environment had dependency conflicts:
- `sentence_transformers` not installed initially
- after installation, issues appeared due to incompatible `torch` / `pandas` / `numpy`

To keep momentum, we switched to a lightweight local embedder:
- `HashingVectorizer`

This gave a working retrieval baseline for `dataset v0` without heavy ML runtime dependencies.

## 11. SQLite-backed `rag_system.py`

We rewrote [rag_system.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/rag_system.py) so that:
- it prefers `knowledge.db` if it exists
- otherwise it falls back to `knowledge.txt`
- it uses the local embedder from `embeddings.py`
- it returns a structured payload rather than only plain text

### Key classes in the updated file
- `LocalLlmConfig`
- `AppConfig`
- `RetrievalResult`
- `AnswerPayload`
- `GemmaChatContext`
- `RetrievalScorer`
- `VectorKnowledgeBase`
- `SQLiteKnowledgeBase`
- `RAGBot`

## 12. Improving retrieval quality

We improved retrieval by blending:
- vector similarity
- lexical overlap with title/content
- an exact title-match bonus

This solved earlier retrieval problems and gave good matches for examples like:
- `When is Commencement 2025?`
- `Selection Days 2025`
- `teaching assistantship training`

## 13. Returning source and image in the answer payload

We extended the retrieval/result pipeline so it returns:
- title
- source URL
- image URL
- image caption
- retrieval score

`AnswerPayload` now includes:
- `answer`
- `title`
- `source_url`
- `image_url`
- `image_caption`
- `retrieval_score`

We also added JSON-like serialization helpers:
- `to_dict()`
- `to_pretty_json()`

## 14. Switching local model via Ollama

We discussed local model use through Ollama.

The project was first aligned with `qwen3.5:2b`, then later switched to:
- `qwen3.5:9b`

The current default model in [rag_system.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/rag_system.py) is:
- `qwen3.5:9b`

We also noted that end-to-end generation requires:
- `ollama serve`
- `ollama pull qwen3.5:9b`

## 15. Linux viewer planning

We discussed how to implement the Linux UI “book” view.

The chosen plan was:
1. first build a `book_formatter.py`
2. only then build the viewer
3. leave swipe gestures for later

## 16. Implemented `book_formatter.py`

We created:
- [book_formatter.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/book_formatter.py)

### Implemented structures
- `BookPage`
- `BookDocument`
- `BookFormatter`

### Formatter behavior
It builds:
- a cover page
- one or more text pages
- an image page if an image exists
- a source page if a source URL exists

It also supports JSON-like export:
- `to_dict()`
- `to_pretty_json()`

## 17. Implemented Linux book viewer without gestures

We created:
- [ui/book_viewer.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/ui/book_viewer.py)

### Viewer capabilities
- one page at a time
- `Previous` / `Next` buttons
- keyboard navigation with `Left` / `Right`
- `F11` for fullscreen toggle
- `Esc` to exit fullscreen
- book-like page container and presentation

## 18. Demo viewer

We created:
- [demo_viewer.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/demo_viewer.py)

This script:
- creates a sample `AnswerPayload`
- runs it through `BookFormatter`
- opens the UI viewer

You ran the demo successfully and shared a screenshot showing the book-like viewer window.

## 19. RAGBot end-to-end viewer launcher

We then created:
- [demo_rag_viewer.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/demo_rag_viewer.py)

This script:
- accepts a query
- constructs `AppConfig()`
- creates a real `RAGBot`
- runs retrieval + generation through Ollama
- converts the result to a `BookDocument`
- opens the viewer

## 20. Result evaluation after running `demo_rag_viewer.py`

You ran:
```bash
python demo_rag_viewer.py "When is Commencement 2025?"
```

The system returned:
- correct event match
- correct title
- correct date/time/place
- correct source URL
- image URL

### Assessment at that point
The result was technically successful end-to-end, but the UI still showed raw Markdown-like text, e.g.:
- `**June 27, 2025**`
- `[Commencement 2025](...)`
- emoji in output

We concluded that a sanitizer/render layer for the UI was needed.

## 21. Implemented sanitizer/render layer for UI

We implemented a sanitizer in [book_formatter.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/book_formatter.py):
- `UiSanitizer`

### It now cleans
- markdown links
- bold markers
- italic markers
- inline code
- extra whitespace
- emoji-like symbol characters

This means the UI now receives display-friendly text instead of raw LLM markdown.

## 22. Dark theme change

You requested a dark gray theme.

We updated [ui/book_viewer.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/ui/book_viewer.py) so that:
- the window background is dark gray
- the page background is a darker graphite tone
- the text is light
- the buttons and labels follow the dark theme

## 23. Real image rendering in the viewer

You then requested that the viewer render actual images instead of just printing the image URL.

We updated [ui/book_viewer.py](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/ui/book_viewer.py) so that it now:
- tries to load a local image from `image_local_path`
- if that is unavailable, tries to fetch the image from `image_url`
- uses `QPixmap` to render the image
- scales the image to fit the page width
- updates the image scaling on window resize
- falls back to text if the image cannot be loaded

We also noted an important caveat:
- many current `image_url` values point to Wikimedia HTML pages rather than direct image files
- for best results, the dataset should eventually store either direct image URLs or local image paths

## 24. Export files created during the session

We created/exported chat history files:
- [CHAT_HISTORY.md](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/CHAT_HISTORY.md)
- [FULL_CHAT_HISTORY.md](/media/imit-learn/ISR_2T3/HoverAI_2/RAG/FULL_CHAT_HISTORY.md)

## 25. Current state at the time of this export

The project now contains:
- dataset v0 in JSONL
- SQLite ingestion pipeline
- SQLite-backed retrieval
- blended retrieval scoring
- structured `AnswerPayload`
- source and image in payload
- `book_formatter.py`
- a Linux book viewer without gestures
- a demo viewer
- a real `RAGBot` viewer launcher
- a UI sanitizer/render layer
- a dark theme
- real image rendering support in the viewer
- default local model set to `qwen3.5:9b`

## 26. Likely next steps after this point

The most likely next engineering steps are:
- normalize image URLs to direct image files or save local copies
- improve viewer layout further
- add graceful handling for unavailable Ollama server/model
- later add swipe gestures and more polished page animations
