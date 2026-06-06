"""文件夹业务逻辑。"""

import psycopg


def create_folder(
    conn: psycopg.Connection,
    folder_name: str,
    parent_folder_id: str | None = None,
) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO folders (folder_name, parent_folder_id)
               VALUES (%s, %s)
               RETURNING folder_id, folder_name, parent_folder_id, created_at""",
            (folder_name.strip(), parent_folder_id),
        )
        row = cur.fetchone()
        conn.commit()
    return {
        "folder_id": str(row[0]),
        "folder_name": row[1],
        "parent_folder_id": str(row[2]) if row[2] else None,
        "created_at": row[3].isoformat(),
    }


def _count_documents(conn: psycopg.Connection, folder_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(DISTINCT document_name) FROM document_chunks WHERE folder_id = %s",
            (folder_id,),
        )
        return cur.fetchone()[0]


def _count_entities(conn: psycopg.Connection, folder_id: str) -> int:
    """通过 document_chunks 估算实体数量——精确数量由 Neo4j 查询提供。"""
    # 粗略估算：每个文档有一批实体
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE folder_id = %s AND chunk_level = 3",
                (folder_id,),
            )
            leaf_count = cur.fetchone()[0]
        return max(1, leaf_count // 3)  # 粗略估：每3个L3块1个实体
    except Exception:
        return 0


def list_folders(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT folder_id, folder_name, parent_folder_id, created_at FROM folders ORDER BY created_at")
        rows = cur.fetchall()
    return [
        {
            "folder_id": str(r[0]),
            "folder_name": r[1],
            "parent_folder_id": str(r[2]) if r[2] else None,
            "document_count": _count_documents(conn, str(r[0])),
            "entity_count": _count_entities(conn, str(r[0])),
            "created_at": r[3].isoformat(),
            "children": [],
        }
        for r in rows
    ]


def build_folder_tree(conn: psycopg.Connection) -> list[dict]:
    folders = list_folders(conn)
    folder_map = {f["folder_id"]: f for f in folders}
    roots: list[dict] = []
    for f in folders:
        parent = f.get("parent_folder_id")
        if parent and parent in folder_map:
            folder_map[parent]["children"].append(f)
        else:
            roots.append(f)
    return roots


def get_folder(conn: psycopg.Connection, folder_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT folder_id, folder_name, parent_folder_id, created_at FROM folders WHERE folder_id = %s",
            (folder_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "folder_id": str(row[0]),
        "folder_name": row[1],
        "parent_folder_id": str(row[2]) if row[2] else None,
        "document_count": _count_documents(conn, folder_id),
        "entity_count": _count_entities(conn, folder_id),
        "created_at": row[3].isoformat(),
        "children": [],
    }


def update_folder(conn: psycopg.Connection, folder_id: str, folder_name: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE folders SET folder_name = %s, updated_at = now() WHERE folder_id = %s RETURNING folder_id, folder_name, parent_folder_id, created_at",
            (folder_name.strip(), folder_id),
        )
        row = cur.fetchone()
        conn.commit()
    if row is None:
        return None
    return {
        "folder_id": str(row[0]),
        "folder_name": row[1],
        "parent_folder_id": str(row[2]) if row[2] else None,
        "created_at": row[3].isoformat(),
    }


def delete_folder(conn: psycopg.Connection, folder_id: str) -> bool:
    """删除文件夹；级联将 document_chunks 的 folder_id 置空。"""
    with conn.cursor() as cur:
        cur.execute("UPDATE document_chunks SET folder_id = NULL WHERE folder_id = %s", (folder_id,))
        cur.execute("DELETE FROM folders WHERE folder_id = %s", (folder_id,))
        deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def ensure_default_folder(conn: psycopg.Connection) -> str:
    """确保存在默认文件夹，返回其 folder_id。"""
    with conn.cursor() as cur:
        cur.execute("SELECT folder_id FROM folders WHERE folder_name = '默认' LIMIT 1")
        row = cur.fetchone()
        if row:
            return str(row[0])
        cur.execute(
            "INSERT INTO folders (folder_name) VALUES ('默认') RETURNING folder_id",
        )
        fid = str(cur.fetchone()[0])
        conn.commit()
    return fid


def move_document_to_folder(
    conn: psycopg.Connection, document_name: str, folder_id: str
) -> int:
    """将文档的所有 chunks 移动到指定文件夹。返回更新行数。"""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE document_chunks SET folder_id = %s WHERE document_name = %s",
            (folder_id, document_name),
        )
        count = cur.rowcount
        conn.commit()
    return count


def list_documents_in_folder(
    conn: psycopg.Connection, folder_id: str
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT document_name,
                      COUNT(*) AS chunk_count,
                      COUNT(*) FILTER (WHERE chunk_level = 3) AS leaf_chunk_count,
                      MIN(created_at) AS created_at
               FROM document_chunks
               WHERE folder_id = %s
               GROUP BY document_name
               ORDER BY document_name""",
            (folder_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "document_name": r[0],
            "chunk_count": r[1],
            "leaf_chunk_count": r[2],
            "created_at": r[3].isoformat() if r[3] else "",
        }
        for r in rows
    ]
