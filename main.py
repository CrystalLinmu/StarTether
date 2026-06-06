import asyncio
import json
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tempfile import NamedTemporaryFile

from encoding_bootstrap import force_utf8
import psycopg
from pydantic import BaseModel
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langsmith import traceable

from chat_memory import (
    ensure_session,
    list_sessions,
    load_memory_context,
    load_session_detail,
    maybe_summarize_session,
    maybe_generate_session_title,
    save_message,
)
from config import get_settings
from db import get_conn, init_db
from entity_extractor import run_entity_extraction
from graph_store import load_graph, save_graph
from ingest import ingest_upload, ingest_path_with_progress
from llm_service import answer_question, stream_answer_question
from milvus_store import delete_document_vectors, ensure_milvus_collection
from neo4j_store import (
    delete_document_graph,
    delete_folder_graph,
    get_neo4j_driver,
    query_subgraph,
)
from retriever import retrieve, retrieve_with_progress
from schemas import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    DocumentChunkItem,
    DocumentListItem,
    FolderCreate,
    FolderItem,
    FolderUpdate,
    GraphEdge,
    GraphNode,
    IngestResponse,
    SessionDetail,
    SessionListItem,
    SubgraphData,
)

logger = logging.getLogger(__name__)

force_utf8()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def fix_filename_mojibake(name: str) -> str:
    """修复文件名乱码（UTF-8 或 GBK 字节被误当作 Latin-1 解码）。"""
    for ch in name:
        if '一' <= ch <= '鿿':
            return name
    for encoding in ('utf-8', 'gbk'):
        try:
            fixed = name.encode('latin-1').decode(encoding)
            if any('一' <= ch <= '鿿' for ch in fixed):
                return fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return name

app = FastAPI(title="Enterprise RAG")

# 全局 Neo4j 驱动（可选）
_neo4j_driver = None


@app.on_event("startup")
def startup() -> None:
    global _neo4j_driver
    settings = get_settings()
    if settings.database_url:
        init_db()
        # 确保默认文件夹存在
        try:
            with psycopg.connect(settings.database_url) as conn:
                from folder_service import ensure_default_folder
                ensure_default_folder(conn)
        except Exception as exc:
            logger.warning("默认文件夹初始化失败: %s", exc)
    ensure_milvus_collection()
    # 初始化 Neo4j（可选）
    _neo4j_driver = get_neo4j_driver()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@traceable(name="langsmith_test", run_type="chain")
def make_trace_test() -> dict[str, str]:
    return {"status": "trace_sent"}


@app.get("/trace-test")
def trace_test() -> dict[str, str]:
    return make_trace_test()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    folder_id: str = Form(""),
    conn=Depends(get_conn),
) -> IngestResponse:
    filename = fix_filename_mojibake(file.filename or "unknown")
    chunks = await ingest_upload(file, conn, folder_id)
    # 无文件夹时自动归入默认文件夹
    if not folder_id and chunks > 0:
        from folder_service import ensure_default_folder
        folder_id = ensure_default_folder(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE document_chunks SET folder_id = %s WHERE document_name = %s AND folder_id IS NULL",
                (folder_id, filename),
            )
            conn.commit()
    # 异步触发实体提取
    if folder_id and chunks > 0:
        _trigger_entity_extraction(filename, folder_id)
    return IngestResponse(filename=filename, chunks=chunks, folder_id=folder_id)


@app.post("/ingest/stream")
async def ingest_stream(
    file: UploadFile = File(...),
    folder_id: str = Form(""),
) -> StreamingResponse:
    filename = fix_filename_mojibake(file.filename or "unknown")
    suffix = Path(filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    async def event_generator():

        settings = get_settings()
        events: Queue[dict] = Queue()
        result_holder: dict = {}

        def emit(code: str, label: str, data: dict | None = None) -> None:
            event = {"type": "step", "code": code, "label": label}
            if data:
                event.update(data)
            events.put(event)

        def ingest_worker() -> None:
            nonlocal folder_id
            try:
                with psycopg.connect(settings.database_url) as conn:
                    chunks = ingest_path_with_progress(temp_path, filename, conn, emit)
                    # 无文件夹时自动归入默认文件夹
                    if not folder_id and chunks > 0:
                        from folder_service import ensure_default_folder
                        folder_id = ensure_default_folder(conn)
                    # 关联文件夹
                    if folder_id and chunks > 0:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE document_chunks SET folder_id = %s WHERE document_name = %s AND folder_id IS NULL",
                                (folder_id, filename),
                            )
                            conn.commit()
                    result_holder["chunks"] = chunks
                    result_holder["folder_id"] = folder_id

                    # GraphRAG 实体提取 + 社区检测（同步）
                    if folder_id and chunks > 0:
                        try:
                            from entity_extractor import extract_entities_from_document
                            from graph_store import load_graph, save_graph
                            emit("graph_extract_start", "GraphRAG 实体提取中…")
                            extract_entities_from_document(filename, folder_id, conn, _neo4j_driver)
                            G = load_graph(folder_id)
                            node_count = G.number_of_nodes()
                            edge_count = G.number_of_edges()
                            result_holder["graph_stats"] = {"entities": node_count, "relationships": edge_count}
                            emit("graph_extract_done",
                                 f"GraphRAG 提取完成（{node_count} 实体, {edge_count} 关系）",
                                 {"entities": node_count, "relationships": edge_count})

                            # 社区检测 + 摘要
                            if node_count >= 3:
                                try:
                                    from community_detector import (
                                        detect_communities_hierarchical,
                                        generate_community_reports,
                                    )
                                    emit("community_start", "GraphRAG 社区检测 + 摘要生成中…")
                                    partitions = detect_communities_hierarchical(G)
                                    generate_community_reports(folder_id, G, partitions)
                                    save_graph(folder_id, G)
                                    community_count = len(set(partitions[0].values()))
                                    emit("community_done",
                                         f"GraphRAG 社区摘要完成（{len(partitions)} 层, {community_count} 个社区）",
                                         {"levels": len(partitions), "level0_communities": community_count})
                                except Exception as exc2:
                                    logger.warning("社区检测失败: %s", exc2)
                        except Exception as exc:
                            logger.warning("GraphRAG 提取失败: %s", exc)
                            emit("graph_extract_done", f"GraphRAG 提取异常: {str(exc)[:60]}")
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                temp_path.unlink(missing_ok=True)
                events.put({"type": "ingest_finished"})

        worker = Thread(target=ingest_worker, daemon=True)
        worker.start()

        while True:
            try:
                event = events.get(timeout=0.2)
            except Empty:
                if not worker.is_alive():
                    break
                continue

            if event["type"] == "ingest_finished":
                break

            # 捕获统计信息
            if event.get("code") == "commit_done":
                result_holder["stats"] = {
                    k: v for k, v in event.items()
                    if k in ("l1_chunks", "l2_chunks", "l3_chunks", "total_chunks", "chars", "document_name")
                }
            if event.get("code") == "graph_extract_done":
                result_holder["graph_stats"] = {
                    k: v for k, v in event.items()
                    if k in ("entities", "relationships", "nodes")
                }

            yield stream_event(event)

        if "error" in result_holder:
            yield stream_event({"type": "error", "message": result_holder["error"]})
            return

        chunks_result = result_holder.get("chunks", 0)
        ingested_folder_id = result_holder.get("folder_id", "")
        stats = result_holder.get("stats", {})
        graph_stats = result_holder.get("graph_stats", {})

        yield stream_event(
            {
                "type": "done",
                "filename": filename,
                "chunks": chunks_result,
                "folder_id": ingested_folder_id,
                "stats": stats,
                "graph_stats": graph_stats,
            }
        )

        # GraphRAG 已在 worker 内同步完成，无需再触发

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/documents", response_model=list[DocumentListItem])
def documents(folder_id: str | None = None, conn=Depends(get_conn)) -> list[DocumentListItem]:
    with conn.cursor() as cur:
        if folder_id:
            cur.execute(
                """
                SELECT document_name, count(*) AS chunk_count,
                       count(*) FILTER (WHERE chunk_level = 3) AS leaf_chunk_count,
                       min(created_at) AS created_at
                FROM document_chunks WHERE folder_id = %s
                GROUP BY document_name ORDER BY created_at DESC
                """,
                (folder_id,),
            )
        else:
            cur.execute(
                """
                SELECT document_name, count(*) AS chunk_count,
                       count(*) FILTER (WHERE chunk_level = 3) AS leaf_chunk_count,
                       min(created_at) AS created_at
                FROM document_chunks
                GROUP BY document_name ORDER BY created_at DESC
                """
            )
        rows = cur.fetchall()

    return [
        DocumentListItem(
            document_name=document_name,
            chunk_count=int(chunk_count),
            leaf_chunk_count=int(leaf_chunk_count),
            created_at=created_at.isoformat(),
        )
        for document_name, chunk_count, leaf_chunk_count, created_at in rows
    ]


@app.get("/documents/{document_name}/chunks", response_model=list[DocumentChunkItem])
def document_chunks(document_name: str, conn=Depends(get_conn)) -> list[DocumentChunkItem]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                chunk_uid,
                parent_uid,
                chunk_level,
                document_name,
                section_title,
                position_hint,
                chunk_index,
                content
            FROM document_chunks
            WHERE document_name = %s
            ORDER BY chunk_index ASC
            """,
            (document_name,),
        )
        rows = cur.fetchall()

    return [
        DocumentChunkItem(
            id=row_id,
            chunk_uid=chunk_uid,
            parent_uid=parent_uid,
            chunk_level=chunk_level,
            document_name=row_document_name,
            section_title=section_title,
            position_hint=position_hint or "",
            chunk_index=chunk_index,
            content=content,
            has_embedding=False,
            has_search_vector=False,
        )
        for (
            row_id,
            chunk_uid,
            parent_uid,
            chunk_level,
            row_document_name,
            section_title,
            position_hint,
            chunk_index,
            content,
        ) in rows
    ]


@app.delete("/documents/{document_name}", response_model=DeleteDocumentResponse)
def delete_document(document_name: str, conn=Depends(get_conn)) -> DeleteDocumentResponse:
    # 获取文档的 folder_id 用于图谱清理
    folder_id = ""
    with conn.cursor() as cur:
        cur.execute("SELECT folder_id FROM document_chunks WHERE document_name = %s AND folder_id IS NOT NULL LIMIT 1", (document_name,))
        row = cur.fetchone()
        if row:
            folder_id = str(row[0]) if row[0] else ""

    delete_document_vectors(document_name)
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_chunks WHERE document_name = %s",
            (document_name,),
        )
        deleted_chunks = cur.rowcount
    conn.commit()

    # 清理 Neo4j 图谱
    if folder_id and _neo4j_driver:
        try:
            delete_document_graph(_neo4j_driver, document_name, folder_id)
        except Exception as exc:
            logger.warning("Neo4j 文档图谱清理失败: %s", exc)

    return DeleteDocumentResponse(document_name=document_name, deleted_chunks=deleted_chunks)


@app.get("/sessions", response_model=list[SessionListItem])
def sessions(conn=Depends(get_conn)) -> list[SessionListItem]:
    return list_sessions(conn)


@app.get("/sessions/{session_id}", response_model=SessionDetail)
def session_detail(session_id: str, conn=Depends(get_conn)) -> SessionDetail:
    return load_session_detail(conn, session_id)


class RenameSessionBody(BaseModel):
    title: str


@app.put("/sessions/{session_id}")
def rename_session(
    session_id: str,
    body: RenameSessionBody,
    conn=Depends(get_conn),
) -> dict:
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "title 不能为空")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE chat_sessions SET title = %s, updated_at = now() WHERE session_id = %s",
            (title[:50], session_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "会话不存在")
        conn.commit()
    return {"session_id": session_id, "title": title}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, conn=Depends(get_conn)) -> dict:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "会话不存在")
        conn.commit()
    return {"session_id": session_id, "deleted": True}


def _get_graph_for_chat(folder_id: str | None, enable_graph: bool) -> tuple:
    """加载图谱并处理全局问题路由。返回 (graph, default_folder_id)。"""
    import networkx as nx
    if not enable_graph:
        return None, None
    if not folder_id:
        # GraphRAG 勾选但未选文件夹 → 自动用默认文件夹
        try:
            with psycopg.connect(get_settings().database_url) as c:
                from folder_service import ensure_default_folder
                folder_id = ensure_default_folder(c)
        except Exception:
            return None, None
    graph = load_graph(folder_id)
    if graph.number_of_nodes() == 0:
        return None, None
    return graph, folder_id


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, conn=Depends(get_conn)) -> ChatResponse:
    session_id = ensure_session(conn, request.session_id)
    memory = load_memory_context(conn, session_id)
    save_message(conn, session_id, "user", request.question)

    # 如果启用 GraphRAG，加载图谱
    graph, resolved_folder_id = _get_graph_for_chat(request.folder_id,
                                                     request.enable_graph)
    effective_folder = request.folder_id or resolved_folder_id

    retrieval = retrieve(
        request.question, conn, memory, debug=True,
        folder_id=effective_folder, enable_graph=request.enable_graph,
        graph=graph, neo4j_driver=_neo4j_driver,
    )
    if retrieval.can_answer:
        answer = answer_question(
            request.question, retrieval.contexts,
            memory_summary=memory.summary,
            recent_messages=[{"role": m.role, "content": m.content} for m in memory.recent_messages],
        )
    else:
        answer = "资料中没有找到相关信息。"

    save_message(conn, session_id, "assistant", answer, rag_trace=retrieval.debug_info)
    maybe_generate_session_title(conn, session_id)
    maybe_summarize_session(conn, session_id)
    updated_memory = load_memory_context(conn, session_id)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        contexts=retrieval.contexts,
        retrieval_query=retrieval.retrieval_query,
        contextualized_question=retrieval.contextualized_question,
        used_contextual_rewrite=retrieval.used_contextual_rewrite,
        rewrite_strategy=retrieval.rewrite_strategy,
        rewritten_query=retrieval.rewritten_query,
        rewrite_attempts=retrieval.rewrite_attempts,
        can_answer=retrieval.can_answer,
        grade_reason=retrieval.grade_reason,
        missing_information=retrieval.missing_information,
        memory_summary=updated_memory.summary,
        debug_info=retrieval.debug_info if request.debug else None,
    )


def stream_event(event: dict) -> str:
    payload = json.dumps(event, ensure_ascii=False)
    # SSE framing is less likely to be buffered by browsers than raw JSON lines.
    return f"data: {payload}\n\n"


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def event_generator():
        settings = get_settings()
        events: Queue[dict] = Queue()
        result_holder: dict = {}

        def retrieval_worker() -> None:
            try:
                with psycopg.connect(settings.database_url) as conn:
                    session_id = ensure_session(conn, request.session_id)
                    memory = load_memory_context(conn, session_id)
                    save_message(conn, session_id, "user", request.question)
                    events.put({"type": "meta", "session_id": session_id})

                    def emit_step(step: dict) -> None:
                        events.put({"type": "step", **step})

                    # 加载图谱
                    graph, resolved_folder_id = _get_graph_for_chat(request.folder_id, request.enable_graph)
                    effective_folder = request.folder_id or resolved_folder_id

                    retrieval = retrieve_with_progress(
                        request.question, conn, memory, emit_step,
                        folder_id=effective_folder, enable_graph=request.enable_graph,
                        graph=graph, neo4j_driver=_neo4j_driver,
                    )
                    result_holder["session_id"] = session_id
                    result_holder["retrieval"] = retrieval
                    result_holder["memory"] = memory
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                events.put({"type": "retrieval_finished"})

        worker = Thread(target=retrieval_worker, daemon=True)
        worker.start()

        while True:
            try:
                event = events.get(timeout=0.2)
            except Empty:
                if not worker.is_alive():
                    break
                continue

            if event["type"] == "retrieval_finished":
                break
            yield stream_event(event)

        if "error" in result_holder:
            yield stream_event({"type": "error", "message": result_holder["error"]})
            return

        try:
            session_id = result_holder["session_id"]
            retrieval = result_holder["retrieval"]
            memory = result_holder.get("memory")
            mem_summary = memory.summary if memory else ""
            mem_recent = [{"role": m.role, "content": m.content} for m in (memory.recent_messages if memory else [])]

            if retrieval.can_answer:
                yield stream_event({"type": "step", "code": "answer_start", "label": "答案生成中"})
                answer_parts: list[str] = []
                for delta in stream_answer_question(
                    request.question, retrieval.contexts,
                    memory_summary=mem_summary,
                    recent_messages=mem_recent,
                ):
                    answer_parts.append(delta)
                    yield stream_event({"type": "answer_delta", "content": delta})
                answer = "".join(answer_parts)
                yield stream_event({"type": "step", "code": "answer_done", "label": "答案生成通过"})
            else:
                answer = "资料中没有找到相关信息。"
                yield stream_event({"type": "answer_delta", "content": answer})

            with psycopg.connect(settings.database_url) as conn:
                save_message(conn, session_id, "assistant", answer, rag_trace=retrieval.debug_info)
                maybe_generate_session_title(conn, session_id)
                maybe_summarize_session(conn, session_id)
                updated_memory = load_memory_context(conn, session_id)

            yield stream_event(
                {
                    "type": "done",
                    "session_id": session_id,
                    "answer": answer,
                    "contexts": [chunk.model_dump() for chunk in retrieval.contexts],
                    "retrieval_query": retrieval.retrieval_query,
                    "contextualized_question": retrieval.contextualized_question,
                    "used_contextual_rewrite": retrieval.used_contextual_rewrite,
                    "rewrite_strategy": retrieval.rewrite_strategy,
                    "rewritten_query": retrieval.rewritten_query,
                    "rewrite_attempts": retrieval.rewrite_attempts,
                    "can_answer": retrieval.can_answer,
                    "grade_reason": retrieval.grade_reason,
                    "missing_information": retrieval.missing_information,
                    "memory_summary": updated_memory.summary,
                    "debug_info": retrieval.debug_info if request.debug else None,
                }
            )
        except Exception as exc:
            yield stream_event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 文件夹 API
# ============================================================


def _trigger_entity_extraction(document_name: str, folder_id: str) -> None:
    """异步触发实体提取任务。"""
    settings = get_settings()

    async def _run():
        try:
            conn_params = {"conninfo": settings.database_url}
            count = await run_entity_extraction(
                document_name, folder_id,
                conn_params=conn_params,
                neo4j_driver=_neo4j_driver,
            )
            logger.info("实体提取完成: %s → %d 实体", document_name, count)

            # 实体提取后自动运行社区检测
            if count > 0:
                try:
                    from community_detector import (
                        detect_communities_hierarchical,
                        generate_community_reports,
                        has_community_reports,
                    )
                    from graph_store import load_graph, save_graph
                    G = load_graph(folder_id)
                    partitions = detect_communities_hierarchical(G)
                    generate_community_reports(folder_id, G, partitions)
                    save_graph(folder_id, G)  # 保存社区ID到图
                    logger.info("社区检测+摘要完成: %s", folder_id)
                except Exception as exc2:
                    logger.warning("社区检测失败: %s", exc2)
        except Exception as exc:
            logger.warning("实体提取失败 (%s): %s", document_name, exc)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        # 没有运行中的事件循环（如测试环境），跳过
        pass


@app.post("/folders")
def create_folder(body: FolderCreate, conn=Depends(get_conn)) -> dict:
    from folder_service import create_folder as svc_create
    return svc_create(conn, body.folder_name, body.parent_folder_id)


@app.get("/folders")
def folders(conn=Depends(get_conn)) -> list[dict]:
    from folder_service import list_folders
    return list_folders(conn)


@app.get("/folders/tree")
def folder_tree(conn=Depends(get_conn)) -> list[dict]:
    from folder_service import build_folder_tree
    return build_folder_tree(conn)


@app.get("/folders/{folder_id}")
def get_folder(folder_id: str, conn=Depends(get_conn)) -> dict:
    from folder_service import get_folder as svc_get
    result = svc_get(conn, folder_id)
    if result is None:
        raise HTTPException(404, "文件夹不存在")
    return result


@app.put("/folders/{folder_id}")
def update_folder(folder_id: str, body: FolderUpdate, conn=Depends(get_conn)) -> dict:
    from folder_service import update_folder as svc_update
    result = svc_update(conn, folder_id, body.folder_name)
    if result is None:
        raise HTTPException(404, "文件夹不存在")
    return result


@app.delete("/folders/{folder_id}")
def delete_folder(folder_id: str, conn=Depends(get_conn)) -> dict:
    from folder_service import delete_folder as svc_delete
    # 先删 Milvus 中该文件夹的全部文档向量
    from folder_service import list_documents_in_folder
    docs = list_documents_in_folder(conn, folder_id)
    for doc in docs:
        delete_document_vectors(doc["document_name"])
    ok = svc_delete(conn, folder_id)
    if not ok:
        raise HTTPException(404, "文件夹不存在")
    # 清理图谱
    from graph_store import delete_graph
    delete_graph(folder_id)
    if _neo4j_driver:
        try:
            delete_folder_graph(_neo4j_driver, folder_id)
        except Exception as exc:
            logger.warning("Neo4j 文件夹图谱清理失败: %s", exc)
    return {"deleted": True, "folder_id": folder_id}


@app.get("/folders/{folder_id}/documents")
def folder_documents(folder_id: str, conn=Depends(get_conn)) -> list[dict]:
    from folder_service import list_documents_in_folder
    return list_documents_in_folder(conn, folder_id)


@app.post("/folders/{folder_id}/ingest/stream")
async def folder_ingest_stream(
    folder_id: str,
    file: UploadFile = File(...),
) -> StreamingResponse:
    """上传文档到指定文件夹（Streaming）。"""
    filename = fix_filename_mojibake(file.filename or "unknown")
    suffix = Path(filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    async def event_generator():

        settings = get_settings()
        events: Queue[dict] = Queue()
        result_holder: dict = {}

        def emit(code: str, label: str, data: dict | None = None) -> None:
            event = {"type": "step", "code": code, "label": label}
            if data:
                event.update(data)
            events.put(event)

        def ingest_worker() -> None:
            try:
                with psycopg.connect(settings.database_url) as conn:
                    chunks = ingest_path_with_progress(temp_path, filename, conn, emit)
                    if chunks > 0:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE document_chunks SET folder_id = %s WHERE document_name = %s AND folder_id IS NULL",
                                (folder_id, filename),
                            )
                            conn.commit()
                    result_holder["chunks"] = chunks
                    result_holder["folder_id"] = folder_id

                    # GraphRAG 实体提取
                    if chunks > 0:
                        try:
                            from entity_extractor import extract_entities_from_document
                            from graph_store import load_graph, save_graph
                            emit("graph_extract_start", "GraphRAG 实体提取中…")
                            extract_entities_from_document(filename, folder_id, conn, _neo4j_driver)
                            G = load_graph(folder_id)
                            node_count = G.number_of_nodes()
                            edge_count = G.number_of_edges()
                            result_holder["graph_stats"] = {"entities": node_count, "relationships": edge_count}
                            emit("graph_extract_done",
                                 f"GraphRAG 提取完成（{node_count} 实体, {edge_count} 关系）",
                                 {"entities": node_count, "relationships": edge_count})
                            # 社区检测 + 摘要
                            if node_count >= 3:
                                try:
                                    from community_detector import (
                                        detect_communities_hierarchical,
                                        generate_community_reports,
                                    )
                                    emit("community_start", "GraphRAG 社区检测 + 摘要生成中…")
                                    partitions = detect_communities_hierarchical(G)
                                    generate_community_reports(folder_id, G, partitions)
                                    save_graph(folder_id, G)
                                    community_count = len(set(partitions[0].values()))
                                    emit("community_done",
                                         f"GraphRAG 社区摘要完成（{len(partitions)} 层, {community_count} 个社区）",
                                         {"levels": len(partitions), "level0_communities": community_count})
                                except Exception as exc2:
                                    logger.warning("社区检测失败: %s", exc2)
                        except Exception as exc:
                            logger.warning("GraphRAG 提取失败: %s", exc)
                            emit("graph_extract_done", f"GraphRAG 提取异常: {str(exc)[:60]}")
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                temp_path.unlink(missing_ok=True)
                events.put({"type": "ingest_finished"})

        worker = Thread(target=ingest_worker, daemon=True)
        worker.start()

        while True:
            try:
                event = events.get(timeout=0.2)
            except Empty:
                if not worker.is_alive():
                    break
                continue
            if event["type"] == "ingest_finished":
                break
            if event.get("code") == "commit_done":
                result_holder["stats"] = {k: v for k, v in event.items()
                    if k in ("l1_chunks", "l2_chunks", "l3_chunks", "total_chunks", "chars", "document_name")}
            if event.get("code") == "graph_extract_done":
                result_holder["graph_stats"] = {k: v for k, v in event.items()
                    if k in ("entities", "relationships")}
            yield stream_event(event)

        if "error" in result_holder:
            yield stream_event({"type": "error", "message": result_holder["error"]})
            return

        chunks_result = result_holder.get("chunks", 0)
        graph_stats = result_holder.get("graph_stats", {})
        yield stream_event({
            "type": "done", "filename": filename, "chunks": chunks_result,
            "folder_id": result_holder.get("folder_id", ""),
            "stats": result_holder.get("stats", {}),
            "graph_stats": graph_stats,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Graph API
# ============================================================


@app.get("/folders/{folder_id}/graph")
def get_folder_graph(
    folder_id: str,
    entity_names: str | None = None,
) -> SubgraphData:
    """获取文件夹的知识图谱子图，供前端可视化。"""
    if not _neo4j_driver:
        raise HTTPException(503, "Neo4j 未配置")

    names = [n.strip() for n in entity_names.split(",")] if entity_names else None
    data = query_subgraph(_neo4j_driver, folder_id, names)
    return SubgraphData(
        nodes=[GraphNode(**n) for n in data["nodes"]],
        edges=[GraphEdge(**e) for e in data["edges"]],
    )


@app.post("/folders/{folder_id}/graph/extract")
def trigger_graph_extraction(folder_id: str, conn=Depends(get_conn)) -> dict:
    """手动触发整个文件夹的图谱重新提取。"""
    from folder_service import list_documents_in_folder
    docs = list_documents_in_folder(conn, folder_id)
    if not docs:
        return {"message": "文件夹无文档", "triggered": 0}
    count = 0
    for doc in docs:
        _trigger_entity_extraction(doc["document_name"], folder_id)
        count += 1
    return {"message": f"已触发 {count} 个文档的实体提取", "triggered": count}


# ---- Document Graph API ----

@app.get("/documents/{document_name}/graph")
def doc_graph(document_name: str) -> dict:
    """返回文档关联的实体+关系子图 + 统计信息。"""
    import json, pathlib
    from graph_store import load_graph
    import psycopg as _psycopg
    from config import get_settings as _gs

    s = _gs()
    stats = {"l1": 0, "l2": 0, "l3": 0, "total": 0, "chars": 0}

    # 从 PG 取文档统计
    with _psycopg.connect(s.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_level, count(*), sum(length(content)) FROM document_chunks WHERE document_name=%s GROUP BY chunk_level", (document_name,))
            for row in cur.fetchall():
                lv, cnt, ch = row
                stats[f"l{lv}"] = cnt
                stats["total"] += cnt
                stats["chars"] += (ch or 0)
            # 取 folder_id
            cur.execute("SELECT folder_id FROM document_chunks WHERE document_name=%s AND folder_id IS NOT NULL LIMIT 1", (document_name,))
            row = cur.fetchone()
            fid = str(row[0]) if row and row[0] else None

    graph_data = {"nodes": [], "edges": []}
    if fid:
        G = load_graph(fid)
        # 提取该文档相关的实体和关系
        doc_nodes = []
        doc_node_ids = set()
        for nid, data in G.nodes(data=True):
            # 实体来源文档匹配 或 chunk_uids 包含该文档的 chunk
            node_doc = data.get("document_name", "")
            in_chunks = any(document_name in str(c) for c in data.get("chunk_uids", []))
            if node_doc == document_name or in_chunks:
                doc_nodes.append({
                    "id": nid, "label": data.get("name", nid),
                    "type": data.get("type", "OTHER"),
                    "group": data.get("type", "OTHER"),
                    "description": data.get("description", "")[:120],
                    "confidence": data.get("confidence", 0.0),
                    "chunk_count": len(data.get("chunk_uids", [])),
                })
                doc_node_ids.add(nid)

        # 收集这些实体之间的关系
        edge_set = set()
        for u, v, d in G.edges(data=True):
            if u in doc_node_ids and v in doc_node_ids:
                eid = f"{u}-{v}"
                if eid not in edge_set:
                    edge_set.add(eid)
                    graph_data["edges"].append({
                        "id": eid, "source": u, "target": v,
                        "label": d.get("relation", ""),
                        "description": d.get("description", ""),
                    })

        graph_data["nodes"] = doc_nodes

    return {
        "document_name": document_name,
        "stats": stats,
        "graph": graph_data,
        "folder_id": fid or "",
    }


# ---- Settings API ----

@app.get("/api/settings")
def api_get_settings() -> list[dict]:
    from config import export_settings
    return export_settings()


@app.put("/api/settings")
def api_update_settings(updates: dict) -> dict:
    from config import set_runtime
    applied = set_runtime(updates)
    return {"applied": len(applied), "keys": list(applied.keys())}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
