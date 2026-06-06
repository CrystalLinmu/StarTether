from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from config import get_settings
from memory_summarizer import summarize_chat_memory, summarize_session_title
from schemas import ChatMessageRecord, MemoryContext, SessionDetail, SessionListItem


def ensure_session(conn: psycopg.Connection, session_id: str | None) -> str:
    resolved_session_id = session_id or str(uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_sessions (session_id)
            VALUES (%s)
            ON CONFLICT (session_id)
            DO UPDATE SET updated_at = now()
            """,
            (resolved_session_id,),
        )
    conn.commit()
    return resolved_session_id


def load_memory_context(
    conn: psycopg.Connection,
    session_id: str,
) -> MemoryContext:
    settings = get_settings()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT summary FROM chat_sessions WHERE session_id = %s",
            (session_id,),
        )
        session_row = cur.fetchone()
        summary = session_row[0] if session_row else ""

        cur.execute(
            """
            SELECT id, role, content
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (session_id, settings.history_message_limit),
        )
        rows = cur.fetchall()

    recent_messages = [
        ChatMessageRecord(id=row_id, role=role, content=content)
        for row_id, role, content in reversed(rows)
    ]
    return MemoryContext(summary=summary or "", recent_messages=recent_messages)


def save_message(
    conn: psycopg.Connection,
    session_id: str,
    role: str,
    content: str,
    rag_trace: dict | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, rag_trace)
            VALUES (%s, %s, %s, %s)
            """,
            (session_id, role, content, Jsonb(rag_trace) if rag_trace is not None else None),
        )
        cur.execute(
            """
            UPDATE chat_sessions
            SET updated_at = now()
            WHERE session_id = %s
            """,
            (session_id,),
        )
    conn.commit()


def maybe_summarize_session(conn: psycopg.Connection, session_id: str) -> None:
    settings = get_settings()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT summary, summarized_message_id
            FROM chat_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        session_row = cur.fetchone()
        if not session_row:
            return

        old_summary, summarized_message_id = session_row
        if summarized_message_id is None:
            cur.execute(
                """
                SELECT id, role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY id ASC
                """,
                (session_id,),
            )
        else:
            cur.execute(
                """
                SELECT id, role, content
                FROM chat_messages
                WHERE session_id = %s
                  AND id > %s
                ORDER BY id ASC
                """,
                (session_id, summarized_message_id),
            )
        rows = cur.fetchall()

    if len(rows) < settings.summary_trigger_messages:
        return

    summarize_rows = rows[: -settings.summary_keep_recent_messages]
    if not summarize_rows:
        return

    messages_to_summarize = [
        ChatMessageRecord(id=row_id, role=role, content=content)
        for row_id, role, content in summarize_rows
    ]
    new_summary = summarize_chat_memory(old_summary or "", messages_to_summarize)
    new_summarized_message_id = summarize_rows[-1][0]

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE chat_sessions
            SET summary = %s,
                summarized_message_id = %s,
                updated_at = now()
            WHERE session_id = %s
            """,
            (new_summary, new_summarized_message_id, session_id),
        )
    conn.commit()


def maybe_generate_session_title(conn: psycopg.Connection, session_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT title FROM chat_sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row or row[0]:
            return

        cur.execute(
            """
            SELECT id, role, content
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY id ASC
            LIMIT 4
            """,
            (session_id,),
        )
        rows = cur.fetchall()

    if len(rows) < 2:
        return

    messages = [
        ChatMessageRecord(id=row_id, role=role, content=content)
        for row_id, role, content in rows
    ]
    try:
        title = summarize_session_title(messages)
    except Exception:
        title = ""
    if not title:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE chat_sessions
            SET title = %s,
                updated_at = now()
            WHERE session_id = %s
            """,
            (title, session_id),
        )
    conn.commit()


def list_sessions(conn: psycopg.Connection) -> list[SessionListItem]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                s.session_id,
                COALESCE(
                    NULLIF(s.title, ''),
                    NULLIF(left(first_user.content, 30), ''),
                    s.session_id
                ) AS title,
                count(m.id) AS message_count,
                s.updated_at
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.session_id
            LEFT JOIN LATERAL (
                SELECT content
                FROM chat_messages
                WHERE session_id = s.session_id
                  AND role = 'user'
                ORDER BY id ASC
                LIMIT 1
            ) first_user ON true
            GROUP BY s.session_id, s.title, first_user.content, s.updated_at
            ORDER BY s.updated_at DESC
            """
        )
        rows = cur.fetchall()

    return [
        SessionListItem(
            session_id=session_id,
            title=title,
            message_count=int(message_count),
            updated_at=updated_at.isoformat(),
        )
        for session_id, title, message_count, updated_at in rows
    ]


def load_session_detail(conn: psycopg.Connection, session_id: str) -> SessionDetail:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT summary
            FROM chat_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        session_row = cur.fetchone()
        if not session_row:
            return SessionDetail(session_id=session_id, messages=[])

        cur.execute(
            """
            SELECT id, role, content, rag_trace
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY id ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()

    return SessionDetail(
        session_id=session_id,
        summary=session_row[0] or "",
        messages=[
            ChatMessageRecord(id=row_id, role=role, content=content, rag_trace=rag_trace)
            for row_id, role, content, rag_trace in rows
        ],
    )
