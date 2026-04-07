from __future__ import annotations

import hashlib
import mimetypes
import urllib.parse
from pathlib import Path

import requests


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}


def resolve_image_url(url: str | None, timeout: int = 10) -> str | None:
    if not url:
        return None

    if "commons.wikimedia.org/wiki/File:" in url:
        file_name = url.split("/wiki/File:", 1)[1]
        return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{file_name}"

    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return url

    return url


def download_image(url: str | None, output_dir: str | Path, timeout: int = 15) -> tuple[str | None, str | None]:
    if not url:
        return None, None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "HoverAI/1.0"}, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            return None, None
        content = response.content
    except Exception:
        return None, None

    digest = hashlib.sha256(content).hexdigest()
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        mime_suffix = mimetypes.guess_extension(response.headers.get('Content-Type', '').split(';')[0].strip() or '')
        suffix = mime_suffix if mime_suffix in IMAGE_EXTENSIONS else '.jpg'

    file_path = output_path / f"{digest}{suffix}"
    if not file_path.exists():
        file_path.write_bytes(content)
    return str(file_path), digest


def normalize_image_record(image: dict, output_dir: str | Path) -> dict:
    record = dict(image)
    original_url = record.get('image_url')
    resolved_url = resolve_image_url(original_url)
    local_path, content_hash = download_image(resolved_url, output_dir)
    record['source_page'] = record.get('source_page') or original_url
    record['image_url'] = resolved_url
    record['local_path'] = local_path
    record['content_hash'] = content_hash
    return record
