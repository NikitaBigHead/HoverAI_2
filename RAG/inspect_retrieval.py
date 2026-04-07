import argparse
import json
from pathlib import Path

import numpy as np

from rag_system import RetrievalScorer
from db import RagDatabase
from embeddings import LocalTextEmbedder
from repository import KnowledgeRepository


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect top retrieval candidates without running the LLM')
    parser.add_argument('query', help='Query to inspect')
    parser.add_argument('--db', default='knowledge.db', help='Path to SQLite knowledge base')
    parser.add_argument('--top-k', type=int, default=5, help='How many candidates to print')
    args = parser.parse_args()

    db_path = Path(args.db)
    db = RagDatabase(str(db_path))
    repo = KnowledgeRepository(db)
    records = repo.fetch_all_chunks()
    if not records:
        print('No records found in knowledge base.')
        return 1

    embedder = LocalTextEmbedder()
    chunk_embeddings = np.array([json.loads(record['embedding_json']) for record in records], dtype=np.float32)
    query_emb = np.array(embedder.encode(args.query, convert_to_numpy=True), dtype=np.float32)
    vector_scores = np.dot(chunk_embeddings, query_emb) / (
        np.maximum(np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb), 1e-12)
    )

    scored = []
    for idx, record in enumerate(records):
        metadata = json.loads(record.get('metadata_json') or '{}')
        verification_status = metadata.get('verification_status')
        score = RetrievalScorer.score(
            query=args.query,
            title=record.get('title') or '',
            content=record.get('content') or '',
            vector_score=float(vector_scores[idx]),
            verification_status=verification_status,
        )
        scored.append({
            'rank_score': round(score, 4),
            'vector_score': round(float(vector_scores[idx]), 4),
            'verification_status': verification_status,
            'title': record.get('title'),
            'source_url': record.get('source_url'),
            'verified_source_url': metadata.get('verified_source_url'),
            'content_preview': (record.get('content') or '')[:180],
        })

    scored.sort(key=lambda item: item['rank_score'], reverse=True)
    for index, item in enumerate(scored[: args.top_k], start=1):
        print(f'#{index}')
        print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
