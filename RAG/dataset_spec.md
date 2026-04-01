# История чата

## 1. Добавление web fallback в RAG

### Запрос
Добавить в RAG-систему `augmented web answer`: если ответ не найден в базе знаний, система должна искать в интернете.

### Что было сделано
- Изучена текущая реализация [rag_system.py](/media/imit-learn/ISR_2T2/HoverAI/Server/rag/rag_system.py).
- Добавлен fallback на веб-поиск.
- Добавлен порог уверенности для локальной базы знаний.
- Добавлены параметры в конфиг:
  - `use_web_fallback`
  - `knowledge_score_threshold`
  - `web_search_timeout_sec`
  - `web_search_max_results`

### Изменённые файлы
- [rag_system.py](/media/imit-learn/ISR_2T2/HoverAI/Server/rag/rag_system.py)
- [config.py](/media/imit-learn/ISR_2T2/HoverAI/Server/server/config.py)
- [config.avatar.yaml](/media/imit-learn/ISR_2T2/HoverAI/Server/config.avatar.yaml)
- [config.avatar_local.yaml](/media/imit-learn/ISR_2T2/HoverAI/Server/config.avatar_local.yaml)
- [config.example.yaml](/media/imit-learn/ISR_2T2/HoverAI/Server/config.example.yaml)

---

## 2. Дружелюбный системный промпт

### Запрос
Написать системный промпт, который сделает модель дружелюбной.

### Предложенный вариант
```yaml
system_message: >-
  You are HoverAI, a friendly and helpful flight drone assistant.
  Speak in a warm, calm, and supportive tone, like a smart companion who is easy to talk to.
  Answer clearly and naturally, using simple words whenever possible.
  Be polite, positive, and concise.
  If the user asks about something from the knowledge base, use that information first.
  If the knowledge base does not contain the answer, use available context or web results if provided.
  If you still do not know the answer, say so honestly instead of inventing information.
  Keep responses short, useful, and easy to understand.
  Use three sentences maximum.
```

---

## 3. Как сделать ответы модели более разнообразными

### Запрос
Какие параметры можно поменять, чтобы модель выдавала более разнообразные фразы.

### Ответ
Рекомендовано менять:
- `temperature`
- `top_p`
- `top_k`

### Практический стартовый пресет
```yaml
temperature: 0.95
top_p: 0.95
top_k: 80
```

### Дополнительное замечание
Жёсткий `system_message` тоже делает стиль ответов более шаблонным.

---

## 4. Почему web RAG не работал

### Запрос
Веб-RAG не отрабатывает.

### Найденная причина
В [config.avatar.yaml](/media/imit-learn/ISR_2T2/HoverAI/Server/config.avatar.yaml) отсутствовал параметр:
```yaml
use_web_fallback: True
```

Из-за этого веб-поиск был выключен по умолчанию.

### Что было сделано
В конфиг были возвращены:
```yaml
use_web_fallback: True
knowledge_score_threshold: 0.55
web_search_timeout_sec: 8.0
web_search_max_results: 3
```

---

## 5. Анализ `rag_system.py` и архитектурные предложения

### Запрос
Изучить код `rag_system.py`, описать структуру и предложить, как адаптировать его под:
- локальную LLM
- RAG с доступом к интернету
- локальную библиотеку источников
- выдачу ссылки на картинку
- вывод результата в Linux-окне в виде PDF-книги с пролистыванием жестами

### Краткий разбор структуры `rag_system.py`
Файл состоит из частей:
- `GemmaChatContext`
- `VectorKnowledgeBase`
- `WebSearchFallback`
- `RAGBot`

### Предложенная архитектура
Выделить отдельные слои:
- `models`
- `storage`
- `retrievers`
- `services`
- `presentation`

### Предложенные будущие возможности
- локальная библиотека документов
- web text search
- image retrieval
- структурированные источники
- `AnswerPayload`
- `BookPage`
- Linux GUI в стиле книги через `PySide6`

---

## 6. Что такое “нормальный индекс документов”

### Запрос
Объяснить, что имеется в виду под “нормальным индексом документов”.

### Ответ
Это не один `knowledge.txt`, а структурированная система:
- документ
- chunk’и
- embeddings
- метаданные
- ссылки на источники
- изображения и связанные файлы

Пример структуры chunk:
```python
{
  "chunk_id": "doc_17_chunk_4",
  "document_id": "doc_17",
  "title": "Museum Guide",
  "content": "Shishkin's Morning in a Pine Forest is displayed in Hall 34...",
  "source_path": "docs/museum_guide.pdf",
  "page": 12,
  "tags": ["painting", "museum", "shishkin"],
  "embedding": [...]
}
```

---

## 7. Что такое `top-k retrieval`

### Запрос
Объяснить, что такое `top-k retrieval` вместо одного chunk.

### Ответ
Сейчас используется `top-1`:
- ищется один самый похожий chunk

`top-k retrieval` означает:
- брать не один лучший chunk, а `k` лучших chunk’ов
- например `k=3` или `k=5`

### Плюсы
- меньше промахов
- более полный контекст
- лучше ответы
- удобнее прикладывать несколько источников

---

## 8. Реализация индекса без FAISS, только на SQLite

### Запрос
Исключить `FAISS`, оставить только `SQLite`.

### Ответ
Предложена схема:
- `SQLite` для документов, chunk’ов, изображений и метаданных
- embeddings хранить прямо в `SQLite`
- поиск similarity выполнять в Python

### Пример таблиц
- `documents`
- `text_chunks`
- `images`

---

## 9. Рефакторинг `rag_system.py` почти как ТЗ

### Запрос
Предложить конкретный рефакторинг `rag_system.py` по классам и файлам.

### Предложенная структура
```text
Server/rag/
  rag_system.py
  models.py
  db.py
  repository.py
  ingest.py
  prompt_builder.py
  answer_service.py
  retrieval_service.py
  web_search.py
  image_service.py
  book_formatter.py
```

### Ключевые сущности
- `RetrievedText`
- `RetrievedImage`
- `AnswerPayload`
- `BookPage`

### Основные компоненты
- `RagDatabase`
- `KnowledgeRepository`
- `KnowledgeIngestor`
- `LocalRetrievalService`
- `WebSearchService`
- `ImageSelectionService`
- `PromptBuilder`
- `AnswerService`
- `BookFormatter`
- `RAGBot` как оркестратор

---

## 10. Подготовка ТЗ для коллеги по сбору датасета о Сколтехе

### Запрос
Подготовить идеи для ТЗ на сбор датасета из открытых источников о Сколтехе, с расчётом на Python-пайплайн.

### Что предложено собирать
- общая информация об институте
- образовательные программы
- исследовательские центры и лаборатории
- факультет и ключевые персоны
- кампус и инфраструктура
- новости и события
- FAQ по поступлению и обучению
- изображения и связанные страницы
- официальные документы: стратегии, отчёты, брошюры, PDF

---

## 11. Что такое JSONL

### Запрос
Что такое `JSONL`.

### Ответ
`JSONL` = `JSON Lines`:
- одна строка файла = один JSON-объект

Пример:
```json
{"id":"1","title":"Skoltech Admissions","text":"Admission info..."}
{"id":"2","title":"Skoltech Research","text":"Research centers..."}
{"id":"3","title":"Campus","text":"Campus information..."}
```

### Почему удобно
- легко читать построчно
- удобно дозаписывать
- хорошо подходит для пайплайнов и RAG

---

## 12. Markdown-адаптация ТЗ

### Были подготовлены Markdown-блоки для:

#### 12.1. ТЗ
```md
# ТЗ

## Цель
Собрать открытые данные о Сколтехе из официальных и вторичных открытых источников.

## Требования к формату хранения
Все результаты необходимо сохранять в формате `JSONL`.

## Требования к сбору данных

### Веб-страницы
Для каждой страницы необходимо сохранять:
- очищенный текст
- URL
- тип сущности
- теги
- дату сбора

### PDF-документы
Для каждого PDF-документа необходимо извлекать и сохранять:
- текст
- метаданные документа
- номер страницы

### Изображения
Для каждого изображения необходимо сохранять:
- URL изображения
- `caption`
- страницу-источник
- локальную копию, если это возможно

### Длинные документы
Для длинных документов необходимо дополнительно формировать `chunks` фиксированного размера.

## Требования к происхождению данных
Для каждого семпла необходимо сохранять информацию о происхождении данных так, чтобы в дальнейшем можно было показать пользователю исходный источник.
```

#### 12.2. Что собирать
```md
## Что собирать

Нужно собирать не просто страницы про Сколтех, а сущности нескольких типов:

- общая информация об институте
- образовательные программы
- исследовательские центры и лаборатории
- факультет и ключевые персоны
- кампус и инфраструктура
- новости и события
- FAQ по поступлению и обучению
- изображения и страницы, к которым они относятся
- официальные документы: стратегии, отчёты, брошюры, PDF
```

#### 12.3. Рекомендуемая структура одного семпла
```md
## Рекомендуемая структура одного семпла

Для Python-пайплайна удобнее всего использовать формат `JSONL`: одна запись на строку.

### Минимальный универсальный формат
```

С примером JSON-объекта.

#### 12.4. Обязательные поля и типы `entity_type`
```md
## Какие поля обязательно попросить собрать

### Обязательные поля
- `id`
- `entity_type`
- `title`
- `text`
- `language`
- `source_url`
- `source_domain`
- `source_type`
- `collected_at`
- `tags`

### Желательные поля
- `summary`
- `section`
- `images`
- `person_names`
- `program_names`
- `pdf_url`
- `local_file_path` если PDF или изображение скачано локально

## Типы `entity_type`

- `about`
- `admissions`
- `program`
- `research_center`
- `laboratory`
- `faculty`
- `person`
- `campus`
- `news`
- `event`
- `official_document`
- `faq`
- `image_asset`
```

#### 12.5. Источники для сбора
```md
## Какие источники использовать

### 1. Официальный сайт Сколтеха
- <https://www.skoltech.ru/en/>
- <https://www.skoltech.ru/en/admissions/>
- <https://www.skoltech.ru/en/research>

### 2. Сайты и поддомены Сколтеха
- <https://student.skoltech.ru/>

### 3. Официальные PDF-документы Сколтеха
- Strategy 2021–2025: <https://back.skoltech.ru/storage/app/media/About%20Skoltech%20-%20Documents/Strategy%202021-2025/strategy-40-eng-09-01-2025.pdf>
- Strategy 2026–2030: <https://back.skoltech.ru/storage/app/media/%D1%81%D1%82%D1%80%D0%B0%D1%82%D0%B5%D0%B3%D0%B8%D1%8F/skoltech-strategy-2026-2030-online.pdf>
- Annual report 2023: <https://back.skoltech.ru/storage/app/media/%D0%B3%D0%BE%D0%B4%D0%BE%D0%B2%D1%8B%D0%B5%20%D0%BE%D1%82%D1%87%D0%B5%D1%82%D1%8B/skoltech-annual-report-2023-eng.pdf>
- Brochure: <https://back.skoltech.ru/storage/app/media/archive/2022/10/SKOLTECH_brochure_50_pages_Grafica_ENGL_singles_for_site_CORRECT.pdf>

### 4. Википедия как вторичный справочный источник
- EN: <https://en.wikipedia.org/wiki/Skolkovo_Institute_of_Science_and_Technology>
- RU: <https://ru.wikipedia.org/wiki/%D0%A1%D0%BA%D0%BE%D0%BB%D0%BA%D0%BE%D0%B2%D1%81%D0%BA%D0%B8%D0%B9_%D0%B8%D0%BD%D1%81%D1%82%D0%B8%D1%82%D1%83%D1%82_%D0%BD%D0%B0%D1%83%D0%BA%D0%B8_%D0%B8_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B9>

### 5. Официальные новости Сколтеха
- <https://www.skoltech.ru/en/news/prestigious-ai-index-report-stanford-university-highlighted-skoltech-research>
```

---

## 13. Как поделиться ссылкой на чат

### Запрос
Как поделиться ссылкой на этот чат.

### Ответ
Предложены варианты:
- использовать кнопку `Share` или `Copy link`, если она есть в интерфейсе;
- если её нет, скопировать содержимое чата;
- удобнее собрать результат в `.md` файл и делиться уже им.

---

Если хочешь, следующим сообщением я могу собрать уже не “историю чата”, а **чистый итоговый документ без диалогов**, в виде одного аккуратного `README.md` или `dataset_spec.md`.