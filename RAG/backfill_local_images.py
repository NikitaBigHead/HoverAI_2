from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import requests


def resolve_wikimedia_image(page_url: str) -> str | None:
    response = requests.get(page_url, headers={'User-Agent': 'Mozilla/5.0 HoverAI/1.0'}, timeout=20)
    response.raise_for_status()
    text = response.text
    marker = 'https://upload.wikimedia.org/wikipedia/commons/thumb/'
    start = text.find(marker)
    if start == -1:
        return None
    end = text.find('"', start)
    if end == -1:
        return None
    return text[start:end]


def download_image(url: str, output_dir: Path) -> tuple[str | None, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 HoverAI/1.0'}, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
    if not content_type.startswith('image/'):
        return None, None
    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    ext = '.jpg'
    file_path = output_dir / f'{digest}{ext}'
    if not file_path.exists():
        file_path.write_bytes(content)
    return str(file_path), digest


def main() -> None:
    db_path = Path('knowledge.db')
    image_dir = Path('assets/images')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('select id, source_page, image_url from images').fetchall()

    cache: dict[str, tuple[str | None, str | None, str | None]] = {}

    for row in rows:
        source_page = row['source_page'] or row['image_url']
        if not source_page:
            continue

        if source_page in cache:
            resolved_url, local_path, content_hash = cache[source_page]
        else:
            try:
                resolved_url = resolve_wikimedia_image(source_page)
                if resolved_url:
                    local_path, content_hash = download_image(resolved_url, image_dir)
                else:
                    local_path, content_hash = None, None
            except Exception:
                resolved_url, local_path, content_hash = None, None, None
            cache[source_page] = (resolved_url, local_path, content_hash)

        conn.execute(
            'update images set image_url = ?, local_path = ?, content_hash = ? where id = ?',
            (resolved_url or row['image_url'], local_path, content_hash, row['id'])
        )

    conn.commit()
    conn.close()
    print(f'updated_rows={len(rows)}')


if __name__ == '__main__':
    main()
