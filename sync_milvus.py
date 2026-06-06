from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from config import get_settings
from llm_service import embed_texts
from milvus_store import delete_document_vectors, ensure_milvus_collection, upsert_leaf_chunks


def sync_existing_chunks_to_milvus() -> int:
    settings = get_settings()
    ensure_milvus_collection()
    total = 0

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT document_name
                FROM document_chunks
                WHERE chunk_level = 3
                ORDER BY document_name
                """
            )
            document_names = [row["document_name"] for row in cur.fetchall()]

            for document_name in document_names:
                delete_document_vectors(document_name)
                cur.execute(
                    """
                    SELECT
                        chunk_uid,
                        parent_uid,
                        chunk_level,
                        document_name,
                        section_title,
                        position_hint,
                        content
                    FROM document_chunks
                    WHERE document_name = %s
                      AND chunk_level = 3
                    ORDER BY chunk_index
                    """,
                    (document_name,),
                )
                rows = cur.fetchall()
                for start in range(0, len(rows), settings.embedding_batch_size):
                    batch = rows[start : start + settings.embedding_batch_size]
                    embeddings = embed_texts([row["content"] for row in batch])
                    records = []
                    for row, embedding in zip(batch, embeddings):
                        records.append(
                            {
                                "chunk_uid": row["chunk_uid"],
                                "parent_uid": row["parent_uid"] or "",
                                "chunk_level": row["chunk_level"],
                                "document_name": row["document_name"],
                                "section_title": row["section_title"][:1024],
                                "position_hint": row["position_hint"][:256],
                                "content": row["content"][: settings.milvus_text_max_length],
                                "dense": embedding,
                            }
                        )
                    upsert_leaf_chunks(records)
                    total += len(records)
                    print(f"{document_name}: synced {total} chunks")

    return total


if __name__ == "__main__":
    count = sync_existing_chunks_to_milvus()
    print(f"Milvus sync done: {count} chunks")
