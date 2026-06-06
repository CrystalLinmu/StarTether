from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import psycopg
from docx import Document
from fastapi import UploadFile
from langsmith import traceable
from pypdf import PdfReader

from config import get_settings
from llm_service import embed_texts
from milvus_store import delete_document_vectors, upsert_leaf_chunks
from tokenizer import tokenize_for_search

ProgressCallback = Callable[[str, str, dict | None], None]


@dataclass
class ChunkNode:
    chunk_uid: str
    parent_uid: str | None
    chunk_level: int
    section_title: str
    position_hint: str
    chunk_index: int
    content: str


@dataclass
class LeafDraft:
    chunk_uid: str
    l1_uid: str
    section_title: str
    position_hint: str
    content: str
    embedding: list[float]


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[page:{page_index}]\n{text}")
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    lines: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            style = paragraph.style.name if paragraph.style else ""
            prefix = "# " if style.startswith("Heading") else ""
            lines.append(prefix + text)
    return "\n".join(lines)


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    raise ValueError("只支持 docx、pdf、txt 文件")


def split_by_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "正文"
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        is_page_marker = line.startswith("[page:")
        is_title = line.startswith("#") or _looks_like_title(line)
        if is_title and current_lines:
            sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = []
        elif is_title:
            current_title = line.lstrip("#").strip()
        else:
            current_lines.append(line if not is_page_marker else f"\n{line}")

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines)) for title, lines in sections]


def _looks_like_title(line: str) -> bool:
    if len(line) > 100:
        return False
    patterns = [
        r"^第[一二三四五六七八九十]+[章节条部分]",
        r"^[一二三四五六七八九十]+[、.．]",
        r"^\d+(\.\d+){0,3}\s+.+",
        r"^\d+(\.\d+){0,3}[、.．]\s*.+",
    ]
    return any(re.match(pattern, line) for pattern in patterns)


def detect_nearest_title(text: str, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("[page:") and _looks_like_title(line):
            return line.lstrip("#").strip()

    inline_match = re.search(r"(?<!\d)(\d{1,2}(?:\.\d+){0,3})\s+([^。；;\n]{2,60})", text)
    if inline_match:
        return f"{inline_match.group(1)} {inline_match.group(2).strip()}"
    return fallback


def detect_position_hint(text: str, fallback: str) -> str:
    match = re.search(r"\[page:(\d+)\]", text)
    if match:
        return f"page {match.group(1)}"
    return fallback


def clean_content(text: str) -> str:
    return re.sub(r"\[page:\d+\]\s*", "", text).strip()


def document_uid_prefix(document_name: str) -> str:
    digest = hashlib.sha1(document_name.encode("utf-8")).hexdigest()[:10]
    return f"doc-{digest}"


def prefixed_uid(prefix: str, uid: str | None) -> str | None:
    if not uid:
        return None
    return f"{prefix}-{uid}"


def split_l1_by_structure(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    buffer = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        next_text = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(next_text) <= max_chars:
            buffer = next_text
            continue

        if buffer:
            parts.append(buffer)
        buffer = paragraph

    if buffer:
        parts.append(buffer)

    return parts or [text]


def recursive_split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    separators = ["\n\n", "\n", "。", "；", "，", "、", ".", " ", ""]
    chunks = _recursive_split(text, max_chars, separators)
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    previous = ""
    for chunk in chunks:
        prefix = previous[-overlap_chars:] if previous else ""
        overlapped.append(f"{prefix}{chunk}" if prefix else chunk)
        previous = chunk

    return overlapped


def _recursive_split(text: str, max_chars: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    separator = separators[0]
    rest = separators[1:]
    if separator == "":
        return [text[start : start + max_chars] for start in range(0, len(text), max_chars)]

    pieces = [piece.strip() for piece in text.split(separator) if piece.strip()]
    if len(pieces) == 1:
        return _recursive_split(text, max_chars, rest)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        piece_with_separator = piece + separator if separator.strip() else piece
        next_text = piece_with_separator if not buffer else buffer + piece_with_separator
        if len(next_text) <= max_chars:
            buffer = next_text
            continue

        if buffer:
            chunks.extend(_recursive_split(buffer, max_chars, rest))
        buffer = piece_with_separator

    if buffer:
        chunks.extend(_recursive_split(buffer, max_chars, rest))

    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def group_leaf_drafts_by_semantic(leaf_drafts: list[LeafDraft]) -> list[list[LeafDraft]]:
    settings = get_settings()
    if not leaf_drafts:
        return []

    groups: list[list[LeafDraft]] = []
    buffer: list[LeafDraft] = [leaf_drafts[0]]

    for draft in leaf_drafts[1:]:
        previous = buffer[-1]
        similarity = cosine_similarity(previous.embedding, draft.embedding)
        next_text = "".join(item.content for item in buffer + [draft])
        should_start_new_group = (
            similarity < settings.semantic_merge_similarity
            or len(next_text) > settings.l2_chunk_max_chars
        )

        if should_start_new_group:
            groups.append(buffer)
            buffer = [draft]
        else:
            buffer.append(draft)

    if buffer:
        groups.append(buffer)

    return groups


@traceable(name="split_chunks", run_type="chain")
def split_chunks(
    text: str,
    emit: ProgressCallback | None = None,
) -> tuple[list[ChunkNode], dict[str, list[float]]]:
    settings = get_settings()
    nodes: list[ChunkNode] = []
    embedding_by_uid: dict[str, list[float]] = {}
    leaf_count = 0

    sections = split_by_sections(text)

    for section_index, (title, section_text) in enumerate(sections):
        l1_parts = split_l1_by_structure(section_text, settings.l1_chunk_max_chars)
        for l1_index, l1_content in enumerate(l1_parts):
            position_hint = detect_position_hint(l1_content, f"section {section_index + 1}")
            l1_title = detect_nearest_title(l1_content, title)
            l1_uid = f"l1-{section_index}-{l1_index}"
            nodes.append(
                ChunkNode(
                    chunk_uid=l1_uid,
                    parent_uid=None,
                    chunk_level=1,
                    section_title=l1_title,
                    position_hint=position_hint,
                    chunk_index=len(nodes),
                    content=clean_content(l1_content),
                )
            )

            l3_parts = recursive_split_with_overlap(
                l1_content,
                settings.l3_chunk_max_chars,
                settings.l3_chunk_overlap_chars,
            )
            if leaf_count + len(l3_parts) > settings.max_chunks_per_doc:
                l3_parts = l3_parts[: settings.max_chunks_per_doc - leaf_count]
            if not l3_parts:
                return nodes, embedding_by_uid

            clean_l3_parts = [clean_content(part) for part in l3_parts]
            embeddings = embed_texts(clean_l3_parts)
            leaf_drafts = [
                LeafDraft(
                    chunk_uid=f"{l1_uid}-l3-draft-{index}",
                    l1_uid=l1_uid,
                    section_title=detect_nearest_title(content, l1_title),
                    position_hint=detect_position_hint(content, position_hint),
                    content=clean_content(content),
                    embedding=embedding,
                )
                for index, (content, embedding) in enumerate(zip(l3_parts, embeddings))
            ]

            l2_groups = group_leaf_drafts_by_semantic(leaf_drafts)
            for l2_index, group in enumerate(l2_groups):
                l2_uid = f"{l1_uid}-l2-{l2_index}"
                l2_content = "".join(item.content for item in group)
                l2_title = detect_nearest_title(l2_content, group[0].section_title)
                l2_position = group[0].position_hint
                nodes.append(
                    ChunkNode(
                        chunk_uid=l2_uid,
                        parent_uid=l1_uid,
                        chunk_level=2,
                        section_title=l2_title,
                        position_hint=l2_position,
                        chunk_index=len(nodes),
                        content=l2_content,
                    )
                )

                for l3_index, draft in enumerate(group):
                    l3_uid = f"{l2_uid}-l3-{l3_index}"
                    nodes.append(
                        ChunkNode(
                            chunk_uid=l3_uid,
                            parent_uid=l2_uid,
                            chunk_level=3,
                            section_title=draft.section_title,
                            position_hint=draft.position_hint,
                            chunk_index=len(nodes),
                            content=draft.content,
                        )
                    )
                    embedding_by_uid[l3_uid] = draft.embedding
                    leaf_count += 1

            if leaf_count >= settings.max_chunks_per_doc:
                return nodes, embedding_by_uid

    return nodes, embedding_by_uid


async def save_upload_file(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        return Path(tmp.name)


@traceable(name="ingest_document", run_type="chain")
async def ingest_upload(file: UploadFile, conn: psycopg.Connection, folder_id: str = "") -> int:
    return await ingest_upload_with_progress(file, conn, None, folder_id)


@traceable(name="ingest_document_with_progress", run_type="chain")
async def ingest_upload_with_progress(
    file: UploadFile,
    conn: psycopg.Connection,
    emit: ProgressCallback | None = None,
    folder_id: str = "",
) -> int:
    path = await save_upload_file(file)
    try:
        leaf_count = ingest_path_with_progress(path, file.filename or "unknown", conn, emit)
        # 将 chunks 关联到文件夹
        if folder_id and leaf_count > 0:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE document_chunks SET folder_id = %s WHERE document_name = %s AND folder_id IS NULL",
                    (folder_id, file.filename),
                )
                conn.commit()
        return leaf_count
    finally:
        path.unlink(missing_ok=True)


def ingest_path_with_progress(
    path: Path,
    document_name: str,
    conn: psycopg.Connection,
    emit: ProgressCallback | None = None,
) -> int:
    def progress(code: str, label: str, data: dict | None = None) -> None:
        if emit:
            emit(code, label, data)

    progress("file_saved", "上传文件已保存到临时目录", {"filename": document_name})
    progress("parse_start", "文档解析中", None)
    text = read_file(path)
    progress("parse_done", "文档解析完成", {"chars": len(text)})

    progress("split_start", "三级分块中", None)
    nodes, embedding_by_uid = split_chunks(text, progress)
    leaf_nodes = [node for node in nodes if node.chunk_level == 3]
    if not leaf_nodes:
        progress("empty", "没有生成可入库的 L3 分块", None)
        return 0
    l1_count = sum(1 for n in nodes if n.chunk_level == 1)
    l2_count = sum(1 for n in nodes if n.chunk_level == 2)
    l3_count = len(leaf_nodes)
    progress(
        "split_done",
        f"三级分块完成（L1 {l1_count} / L2 {l2_count} / L3 {l3_count}）",
        {"total_chunks": len(nodes), "l1_chunks": l1_count, "l2_chunks": l2_count, "l3_chunks": l3_count},
    )

    settings = get_settings()
    uid_prefix = document_uid_prefix(document_name)
    milvus_records: list[dict] = []
    delete_document_vectors(document_name)

    with conn.cursor() as cur:
        progress("insert_start", "写入数据库中", {"document_name": document_name})
        cur.execute(
            "DELETE FROM document_chunks WHERE document_name = %s",
            (document_name,),
        )

        for index, node in enumerate(nodes, start=1):
            chunk_uid = prefixed_uid(uid_prefix, node.chunk_uid) or ""
            parent_uid = prefixed_uid(uid_prefix, node.parent_uid)
            embedding = embedding_by_uid.get(node.chunk_uid)
            cur.execute(
                """
                INSERT INTO document_chunks
                (
                    chunk_uid,
                    parent_uid,
                    chunk_level,
                    document_name,
                    section_title,
                    position_hint,
                    chunk_index,
                    content,
                    tokenized_content
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    chunk_uid,
                    parent_uid,
                    node.chunk_level,
                    document_name,
                    node.section_title,
                    node.position_hint,
                    node.chunk_index,
                    node.content,
                    "",
                ),
            )
            if node.chunk_level == 3 and embedding is not None:
                milvus_records.append(
                    {
                        "chunk_uid": chunk_uid,
                        "parent_uid": parent_uid or "",
                        "chunk_level": node.chunk_level,
                        "document_name": document_name,
                        "section_title": node.section_title[:1024],
                        "position_hint": node.position_hint[:256],
                        "content": node.content[: settings.milvus_text_max_length],
                        "tokenized_content": tokenize_for_search(node.content)[: settings.milvus_text_max_length],
                        "dense": embedding,
                    }
                )

    progress("milvus_start", "写入 Milvus 向量库中", {"leaf_chunks": len(milvus_records)})
    upsert_leaf_chunks(milvus_records)
    progress(
        "milvus_done",
        f"Milvus 向量入库完成（dense + BM25 sparse，L3 {len(milvus_records)} 块）",
        {"leaf_chunks": len(milvus_records)},
    )
    conn.commit()
    progress(
        "commit_done",
        f"入库完成（L1 {l1_count} / L2 {l2_count} / L3 {l3_count}，{len(text)} 字）",
        {
            "l1_chunks": l1_count, "l2_chunks": l2_count, "l3_chunks": l3_count,
            "total_chunks": len(nodes), "chars": len(text), "document_name": document_name,
        },
    )
    return len(leaf_nodes)
