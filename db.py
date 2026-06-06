from collections.abc import Iterator

from fastapi import HTTPException
import psycopg

from config import get_settings


def get_conn() -> Iterator[psycopg.Connection]:
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(status_code=400, detail="请先在 .env 中配置 DATABASE_URL")

    with psycopg.connect(settings.database_url) as conn:
        yield conn


def init_db() -> None:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("请先在 .env 中配置 DATABASE_URL")

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    chunk_uid TEXT NOT NULL DEFAULT '',
                    parent_uid TEXT,
                    chunk_level INTEGER NOT NULL DEFAULT 3,
                    document_name TEXT NOT NULL,
                    section_title TEXT NOT NULL DEFAULT '',
                    position_hint TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    tokenized_content TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_uid TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_uid TEXT")
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_level INTEGER NOT NULL DEFAULT 3")
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS position_hint TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS tokenized_content TEXT NOT NULL DEFAULT ''")
            cur.execute("DROP INDEX IF EXISTS idx_document_chunks_embedding")
            cur.execute("DROP INDEX IF EXISTS idx_document_chunks_search_vector")
            cur.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding")
            cur.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS search_vector")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_message_id BIGINT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS summarized_message_id BIGINT")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    rag_trace JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS rag_trace JSONB")

            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_document_chunks_chunk_uid
                ON document_chunks (chunk_uid)
                WHERE chunk_uid <> ''
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_parent_uid ON document_chunks (parent_uid)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_level ON document_chunks (chunk_level)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages (session_id, id DESC)
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS folders (
                    folder_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    folder_name TEXT NOT NULL,
                    parent_folder_id UUID REFERENCES folders(folder_id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS folder_id UUID REFERENCES folders(folder_id) ON DELETE SET NULL")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_folder_id ON document_chunks (folder_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_folder_doc ON document_chunks (folder_id, document_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders (parent_folder_id)")

        conn.commit()
