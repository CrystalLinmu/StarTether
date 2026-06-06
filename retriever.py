from collections.abc import Callable
from time import perf_counter
import logging

import networkx as nx
import psycopg
from langsmith import traceable
from neo4j import Driver
from psycopg.rows import dict_row

from config import get_runtime, get_settings
from grader import grade_contexts
from graph_retriever import (
    fuse_with_vector,
    global_search,
    graph_retrieve,
    is_global_question,
    should_escalate_to_graph,
)
from llm_service import embed_texts
from milvus_store import dense_search, sparse_search
from query_rewriter import (
    choose_rewrite_order,
    contextualize_question,
    explain_contextual_rewrite,
    explain_rewrite_order,
    explain_should_rewrite,
    needs_contextual_rewrite,
    rewrite_query,
    should_rewrite,
)
from reranker import rerank_chunks
from schemas import MemoryContext, RetrievedChunk, RetrievalResult


def rrf(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def merge_by_rrf(
    dense_chunks: list[RetrievedChunk],
    keyword_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    merged: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        if chunk.id not in merged:
            merged[chunk.id] = chunk.model_copy()
            merged[chunk.id].rrf_score = 0

        merged[chunk.id].dense_score = chunk.dense_score
        merged[chunk.id].rrf_score = (merged[chunk.id].rrf_score or 0) + rrf(rank)

    for rank, chunk in enumerate(keyword_chunks, start=1):
        if chunk.id not in merged:
            merged[chunk.id] = chunk.model_copy()
            merged[chunk.id].rrf_score = 0

        merged[chunk.id].sparse_score = chunk.sparse_score
        merged[chunk.id].rrf_score = (merged[chunk.id].rrf_score or 0) + rrf(rank)

    return sorted(
        merged.values(),
        key=lambda chunk: chunk.rrf_score or 0,
        reverse=True,
    )


def fetch_chunks_by_uid(
    conn: psycopg.Connection,
    chunk_uids: list[str],
    score_by_uid: dict[str, float],
    score_field: str,
) -> list[RetrievedChunk]:
    if not chunk_uids:
        return []

    with conn.cursor(row_factory=dict_row) as cur:
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
                content
            FROM document_chunks
            WHERE chunk_uid = ANY(%s)
            """,
            (chunk_uids,),
        )
        rows_by_uid = {row["chunk_uid"]: row for row in cur.fetchall()}

    chunks: list[RetrievedChunk] = []
    for chunk_uid in chunk_uids:
        row = rows_by_uid.get(chunk_uid)
        if not row:
            continue

        score = score_by_uid.get(chunk_uid)
        chunks.append(
            RetrievedChunk(
                id=row["id"],
                chunk_uid=row["chunk_uid"],
                parent_uid=row["parent_uid"],
                chunk_level=row["chunk_level"],
                document_name=row["document_name"],
                section_title=row["section_title"],
                position_hint=row["position_hint"],
                content=row["content"],
                dense_score=score if score_field == "dense" else None,
                sparse_score=score if score_field == "sparse" else None,
            )
        )
    return chunks


@traceable(name="auto_merge_parent_chunks", run_type="retriever")
def auto_merge_parent_chunks(
    chunks: list[RetrievedChunk],
    conn: psycopg.Connection,
) -> list[RetrievedChunk]:
    settings = get_settings()
    parent_hits: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        if chunk.parent_uid:
            parent_hits.setdefault(chunk.parent_uid, []).append(chunk)

    if not parent_hits:
        return chunks

    parent_uids = list(parent_hits)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT parent_uid, count(*) AS total_children
            FROM document_chunks
            WHERE chunk_level = 3
              AND parent_uid = ANY(%s)
            GROUP BY parent_uid
            """,
            (parent_uids,),
        )
        total_children_by_parent = {
            row["parent_uid"]: int(row["total_children"]) for row in cur.fetchall()
        }

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
                content
            FROM document_chunks
            WHERE chunk_level = 2
              AND chunk_uid = ANY(%s)
            """,
            (parent_uids,),
        )
        parents_by_uid = {row["chunk_uid"]: row for row in cur.fetchall()}

    merged_parent_uids: set[str] = set()
    merged_chunks: list[RetrievedChunk] = []
    for parent_uid, hit_children in parent_hits.items():
        total_children = total_children_by_parent.get(parent_uid, 0)
        if not total_children:
            continue

        merge_ratio = len(hit_children) / total_children
        parent = parents_by_uid.get(parent_uid)
        if (
            parent
            and len(hit_children) >= settings.auto_merge_min_children
            and merge_ratio >= settings.auto_merge_child_ratio
            and len(parent["content"]) <= get_runtime("auto_merge_max_parent_chars", settings.auto_merge_max_parent_chars)
        ):
            best_child = max(
                hit_children,
                key=lambda child: child.rerank_score
                if child.rerank_score is not None
                else child.rrf_score or 0,
            )
            merged_chunks.append(
                RetrievedChunk(
                    id=parent["id"],
                    chunk_uid=parent["chunk_uid"],
                    parent_uid=parent["parent_uid"],
                    chunk_level=parent["chunk_level"],
                    document_name=parent["document_name"],
                    section_title=parent["section_title"],
                    position_hint=parent["position_hint"],
                    content=parent["content"],
                    dense_score=best_child.dense_score,
                    sparse_score=best_child.sparse_score,
                    rrf_score=best_child.rrf_score,
                    rerank_score=best_child.rerank_score,
                    merge_ratio=merge_ratio,
                    merged_from_children=[child.chunk_uid for child in hit_children],
                    merge_info={
                        "merge_applied": True,
                        "parent_uid": parent_uid,
                        "hit_children_count": len(hit_children),
                        "total_children_count": total_children,
                        "merge_ratio": merge_ratio,
                        "merge_threshold": settings.auto_merge_child_ratio,
                        "min_children": settings.auto_merge_min_children,
                        "merged_from_children": [child.chunk_uid for child in hit_children],
                        "merge_reason": "同一父块下命中子块数量和命中比例达到阈值",
                    },
                )
            )
            merged_parent_uids.add(parent_uid)

    output_chunks = merged_chunks[:]
    for chunk in chunks:
        if chunk.parent_uid not in merged_parent_uids:
            output_chunks.append(chunk)

    return output_chunks


@traceable(name="dense_retrieve_context", run_type="retriever")
def dense_retrieve(
    question: str,
    conn: psycopg.Connection,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    limit = limit or settings.top_k
    query_embedding = embed_texts([question])[0]
    hits = dense_search(query_embedding, limit)
    return fetch_chunks_by_uid(
        conn,
        [hit.chunk_uid for hit in hits],
        {hit.chunk_uid: hit.score for hit in hits},
        "dense",
    )


@traceable(name="keyword_retrieve_context", run_type="retriever")
def keyword_retrieve(
    question: str,
    conn: psycopg.Connection,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    limit = limit or settings.top_k
    hits = sparse_search(question, limit)
    return fetch_chunks_by_uid(
        conn,
        [hit.chunk_uid for hit in hits],
        {hit.chunk_uid: hit.score for hit in hits},
        "sparse",
    )


@traceable(name="hybrid_retrieve_context", run_type="retriever")
def hybrid_retrieve(question: str, conn: psycopg.Connection) -> list[RetrievedChunk]:
    settings = get_settings()

    dense_chunks = dense_retrieve(question, conn, settings.candidate_top_k)
    keyword_chunks = keyword_retrieve(question, conn, settings.candidate_top_k)

    merged_chunks = merge_by_rrf(dense_chunks, keyword_chunks)
    reranked_chunks = rerank_chunks(question, merged_chunks)
    auto_merged_chunks = auto_merge_parent_chunks(reranked_chunks, conn)
    return auto_merged_chunks[: settings.top_k]


def hybrid_retrieve_with_debug(
    question: str,
    conn: psycopg.Connection,
) -> tuple[list[RetrievedChunk], dict]:
    settings = get_settings()
    timings: dict[str, int] = {}

    start = perf_counter()
    dense_chunks = dense_retrieve(question, conn, settings.candidate_top_k)
    timings["dense_ms"] = elapsed_ms(start)
    start = perf_counter()
    keyword_chunks = keyword_retrieve(question, conn, settings.candidate_top_k)
    timings["keyword_ms"] = elapsed_ms(start)
    start = perf_counter()
    rrf_chunks = merge_by_rrf(dense_chunks, keyword_chunks)
    timings["rrf_ms"] = elapsed_ms(start)
    start = perf_counter()
    reranked_chunks = rerank_chunks(question, rrf_chunks)
    timings["rerank_ms"] = elapsed_ms(start)
    start = perf_counter()
    auto_merged_chunks = auto_merge_parent_chunks(reranked_chunks, conn)
    timings["auto_merge_ms"] = elapsed_ms(start)
    final_chunks = auto_merged_chunks[: settings.top_k]
    merged_count = sum(1 for chunk in auto_merged_chunks if chunk.merge_info)

    return final_chunks, {
        "dense_results": [chunk.model_dump() for chunk in dense_chunks],
        "keyword_results": [chunk.model_dump() for chunk in keyword_chunks],
        "rrf_results": [chunk.model_dump() for chunk in rrf_chunks],
        "rerank_results": [chunk.model_dump() for chunk in reranked_chunks],
        "auto_merged_results": [chunk.model_dump() for chunk in auto_merged_chunks],
        "final_results": [chunk.model_dump() for chunk in final_chunks],
        "timings": timings,
        "summary": {
            "dense_count": len(dense_chunks),
            "keyword_count": len(keyword_chunks),
            "rrf_count": len(rrf_chunks),
            "rerank_count": len(reranked_chunks),
            "auto_merge_count": len(auto_merged_chunks),
            "merged_parent_count": merged_count,
            "final_count": len(final_chunks),
        },
    }


def hybrid_retrieve_with_progress(
    question: str,
    conn: psycopg.Connection,
    emit: Callable[[dict], None],
) -> tuple[list[RetrievedChunk], dict]:
    settings = get_settings()
    timings: dict[str, int] = {}

    start = perf_counter()
    emit({"code": "dense_start", "label": "Milvus 稠密向量检索中"})
    dense_chunks = dense_retrieve(question, conn, settings.candidate_top_k)
    timings["dense_ms"] = elapsed_ms(start)
    emit({"code": "dense_done", "label": f"Milvus 稠密向量检索通过（检索到 {len(dense_chunks)} 块，用时 {timings['dense_ms']}ms）", "count": len(dense_chunks), "elapsed_ms": timings["dense_ms"]})

    start = perf_counter()
    emit({"code": "keyword_start", "label": "Milvus BM25 稀疏向量检索中"})
    keyword_chunks = keyword_retrieve(question, conn, settings.candidate_top_k)
    timings["keyword_ms"] = elapsed_ms(start)
    emit({"code": "keyword_done", "label": f"Milvus BM25 稀疏向量检索通过（检索到 {len(keyword_chunks)} 块，用时 {timings['keyword_ms']}ms）", "count": len(keyword_chunks), "elapsed_ms": timings["keyword_ms"]})

    start = perf_counter()
    emit({"code": "rrf_start", "label": "RRF 融合排序中"})
    rrf_chunks = merge_by_rrf(dense_chunks, keyword_chunks)
    timings["rrf_ms"] = elapsed_ms(start)
    emit({"code": "rrf_done", "label": f"RRF 融合通过（融合后 {len(rrf_chunks)} 块，用时 {timings['rrf_ms']}ms）", "count": len(rrf_chunks), "elapsed_ms": timings["rrf_ms"]})

    start = perf_counter()
    emit({"code": "rerank_start", "label": "Rerank 精排中"})
    reranked_chunks = rerank_chunks(question, rrf_chunks)
    timings["rerank_ms"] = elapsed_ms(start)
    emit({"code": "rerank_done", "label": f"Rerank 精排通过（返回 {len(reranked_chunks)} 块，用时 {timings['rerank_ms']}ms）", "count": len(reranked_chunks), "elapsed_ms": timings["rerank_ms"]})

    start = perf_counter()
    emit({"code": "auto_merge_start", "label": "父块自动合并中"})
    auto_merged_chunks = auto_merge_parent_chunks(reranked_chunks, conn)
    timings["auto_merge_ms"] = elapsed_ms(start)
    merged_count = sum(1 for chunk in auto_merged_chunks if chunk.merge_info)
    emit({"code": "auto_merge_done", "label": f"父块自动合并通过（得到 {len(auto_merged_chunks)} 块，合并 {merged_count} 个父块，用时 {timings['auto_merge_ms']}ms）", "count": len(auto_merged_chunks), "merged_count": merged_count, "elapsed_ms": timings["auto_merge_ms"]})

    final_chunks = auto_merged_chunks[: settings.top_k]
    return final_chunks, {
        "dense_results": [chunk.model_dump() for chunk in dense_chunks],
        "keyword_results": [chunk.model_dump() for chunk in keyword_chunks],
        "rrf_results": [chunk.model_dump() for chunk in rrf_chunks],
        "rerank_results": [chunk.model_dump() for chunk in reranked_chunks],
        "auto_merged_results": [chunk.model_dump() for chunk in auto_merged_chunks],
        "final_results": [chunk.model_dump() for chunk in final_chunks],
        "timings": timings,
        "summary": {
            "dense_count": len(dense_chunks),
            "keyword_count": len(keyword_chunks),
            "rrf_count": len(rrf_chunks),
            "rerank_count": len(reranked_chunks),
            "auto_merge_count": len(auto_merged_chunks),
            "merged_parent_count": merged_count,
            "final_count": len(final_chunks),
        },
    }

def retrieve_once(query: str, conn: psycopg.Connection) -> list[RetrievedChunk]:
    return hybrid_retrieve(query, conn)


def retrieve_once_with_debug(
    query: str,
    conn: psycopg.Connection,
) -> tuple[list[RetrievedChunk], dict]:
    return hybrid_retrieve_with_debug(query, conn)


def retrieve_once_with_progress(
    query: str,
    conn: psycopg.Connection,
    emit: Callable[[dict], None],
) -> tuple[list[RetrievedChunk], dict]:
    return hybrid_retrieve_with_progress(query, conn, emit)


@traceable(name="retrieve_with_query_rewrite", run_type="retriever")
def retrieve(
    question: str,
    conn: psycopg.Connection,
    memory: MemoryContext | None = None,
    debug: bool = False,
    folder_id: str | None = None,
    enable_graph: bool = True,
    graph: nx.Graph | None = None,
    neo4j_driver: Driver | None = None,
) -> RetrievalResult:
    memory = memory or MemoryContext()
    context_reason = explain_contextual_rewrite(question, memory)
    context_start = perf_counter()
    used_contextual_rewrite = needs_contextual_rewrite(question, memory)
    base_query = contextualize_question(question, memory)
    context_ms = elapsed_ms(context_start)
    debug_runs: list[dict] = []

    if debug:
        original_contexts, original_debug = retrieve_once_with_debug(base_query, conn)
    else:
        original_contexts = retrieve_once(base_query, conn)
        original_debug = {}

    grade_start = perf_counter()
    original_grade = grade_contexts(question, original_contexts)
    original_grade_ms = elapsed_ms(grade_start)
    if debug:
        original_debug.setdefault("timings", {})["grade_ms"] = original_grade_ms
        original_debug["rewrite_trigger_reason"] = explain_should_rewrite(original_contexts)
    if debug:
        debug_runs.append(
            {
                "strategy": "original",
                "query": base_query,
                **original_debug,
                "grade": original_grade.model_dump(),
            }
        )

    if original_grade.can_answer and not should_rewrite(original_contexts):
        # 质量不佳时触发 GraphRAG 升级
        escalate = should_escalate_to_graph(
            RetrievalResult(
                contexts=original_contexts, retrieval_query=base_query,
                can_answer=True, grade_reason=original_grade.reason,
            )
        ) if enable_graph and graph and folder_id else None
        if not escalate:
            return RetrievalResult(
                contexts=original_contexts,
                retrieval_query=base_query,
                contextualized_question=base_query if used_contextual_rewrite else None,
                used_contextual_rewrite=used_contextual_rewrite,
                can_answer=True,
                grade_reason=original_grade.reason,
                missing_information=original_grade.missing_information,
                debug_info={
                    "original_question": question,
                    "contextualized_question": base_query,
                    "used_contextual_rewrite": used_contextual_rewrite,
                    "contextual_rewrite_reason": context_reason,
                    "contextual_rewrite_elapsed_ms": context_ms,
                    "selected_strategy": "original",
                    "selected_query": base_query,
                    "runs": debug_runs,
                }
                if debug
                else None,
            )

    attempts: list[str] = []
    best_contexts = original_contexts
    best_query = base_query
    best_strategy = "none"
    best_grade = original_grade
    rewrite_order_reason = explain_rewrite_order(base_query)

    for strategy in choose_rewrite_order(base_query):
        rewrite_start = perf_counter()
        rewritten_query = rewrite_query(base_query, strategy)
        rewrite_ms = elapsed_ms(rewrite_start)
        attempts.append(f"{strategy}: {rewritten_query}")
        if debug:
            rewritten_contexts, rewritten_debug = retrieve_once_with_debug(rewritten_query, conn)
        else:
            rewritten_contexts = retrieve_once(rewritten_query, conn)
            rewritten_debug = {}

        if debug:
            rewritten_debug.setdefault("timings", {})["rewrite_ms"] = rewrite_ms
        grade_start = perf_counter()
        rewritten_grade = grade_contexts(question, rewritten_contexts)
        grade_ms = elapsed_ms(grade_start)
        if debug:
            rewritten_debug.setdefault("timings", {})["grade_ms"] = grade_ms
            rewritten_debug["rewrite_trigger_reason"] = explain_should_rewrite(rewritten_contexts)
        if debug:
            debug_runs.append(
                {
                    "strategy": strategy,
                    "query": rewritten_query,
                    **rewritten_debug,
                    "grade": rewritten_grade.model_dump(),
                }
            )

        if rewritten_grade.can_answer and not should_rewrite(rewritten_contexts):
            return RetrievalResult(
                contexts=rewritten_contexts,
                retrieval_query=rewritten_query,
                contextualized_question=base_query if used_contextual_rewrite else None,
                used_contextual_rewrite=used_contextual_rewrite,
                rewrite_strategy=strategy,
                rewritten_query=rewritten_query,
                rewrite_attempts=attempts,
                can_answer=True,
                grade_reason=rewritten_grade.reason,
                missing_information=rewritten_grade.missing_information,
                debug_info={
                    "original_question": question,
                    "contextualized_question": base_query,
                    "used_contextual_rewrite": used_contextual_rewrite,
                    "contextual_rewrite_reason": context_reason,
                    "contextual_rewrite_elapsed_ms": context_ms,
                    "selected_strategy": strategy,
                    "selected_query": rewritten_query,
                    "rewrite_order_reason": rewrite_order_reason,
                    "rewrite_attempts": attempts,
                    "runs": debug_runs,
                }
                if debug
                else None,
            )

        if len(rewritten_contexts) > len(best_contexts):
            best_contexts = rewritten_contexts
            best_query = rewritten_query
            best_strategy = strategy
            best_grade = rewritten_grade

    # ============================================================
    # GraphRAG 升级: RAG + 重写都不行 → Local Search → Global Search
    # ============================================================
    settings = get_settings()
    logger = logging.getLogger(__name__)
    graph_data: dict | None = None
    graph_contexts = best_contexts

    if enable_graph and graph and folder_id and settings.graph_retrieval_enabled:
        # L2: GraphRAG Local Search
        try:
            local_result = graph_retrieve(
                question=base_query,
                folder_id=folder_id,
                graph=graph,
                max_hops=settings.graph_max_hops,
                top_k=settings.graph_top_k_chunks,
                emit=emit,
            )
            if local_result["chunk_uids"]:
                fused = fuse_with_vector(
                    best_contexts, local_result["chunk_uids"], conn, base_query,
                )
                fused_grade = grade_contexts(question, fused)
                graph_data = {
                    "entities_matched": local_result["entities_matched"],
                    "graph_chunks_found": len(local_result["chunk_uids"]),
                    "overlap_with_vector": len(
                        set(c.chunk_uid for c in best_contexts)
                        & set(local_result["chunk_uids"])
                    ),
                    "subgraph": local_result["subgraph"],
                }
                if fused_grade.can_answer:
                    return RetrievalResult(
                        contexts=fused,
                        retrieval_query=base_query,
                        contextualized_question=base_query if used_contextual_rewrite else None,
                        used_contextual_rewrite=used_contextual_rewrite,
                        rewrite_strategy=best_strategy if best_strategy != "none" else "graph_local",
                        rewritten_query=best_query if best_strategy != "none" else None,
                        rewrite_attempts=attempts + [f"graph_local: {len(local_result['chunk_uids'])} chunks"],
                        can_answer=True,
                        grade_reason=fused_grade.reason,
                        missing_information=fused_grade.missing_information,
                        debug_info={
                            "original_question": question,
                            "contextualized_question": base_query,
                            "used_contextual_rewrite": used_contextual_rewrite,
                            "contextual_rewrite_reason": context_reason,
                            "contextual_rewrite_elapsed_ms": context_ms,
                            "selected_strategy": "graph_local",
                            "selected_query": base_query,
                            "rewrite_order_reason": rewrite_order_reason,
                            "rewrite_attempts": attempts + [f"graph_local: {len(local_result['chunk_uids'])} chunks"],
                            "runs": debug_runs,
                            "graph_data": graph_data,
                        } if debug else None,
                    )
                graph_contexts = fused
        except Exception as exc:
            logger.warning("GraphRAG local search 失败: %s", exc)

        # L3: Global Search 兜底
        try:
            gs_context = global_search(question, folder_id, graph)
            if gs_context:
                # 将社区摘要作为特殊 context 追加
                if graph_data is None:
                    graph_data = {}
                graph_data["global_context"] = gs_context
        except Exception as exc:
            logger.warning("GraphRAG global search 失败: %s", exc)

    return RetrievalResult(
        contexts=graph_contexts,
        retrieval_query=best_query,
        contextualized_question=base_query if used_contextual_rewrite else None,
        used_contextual_rewrite=used_contextual_rewrite,
        rewrite_strategy=best_strategy,
        rewritten_query=best_query if best_strategy != "none" else None,
        rewrite_attempts=attempts,
        can_answer=best_grade.can_answer,
        grade_reason=best_grade.reason,
        missing_information=best_grade.missing_information,
        debug_info={
            "original_question": question,
            "contextualized_question": base_query,
            "used_contextual_rewrite": used_contextual_rewrite,
            "contextual_rewrite_reason": context_reason,
            "contextual_rewrite_elapsed_ms": context_ms,
            "selected_strategy": best_strategy,
            "selected_query": best_query,
            "rewrite_order_reason": rewrite_order_reason,
            "rewrite_attempts": attempts,
            "runs": debug_runs,
            **({"graph_data": graph_data} if graph_data else {}),
        } if debug else None,
    )


def retrieve_with_progress(
    question: str,
    conn: psycopg.Connection,
    memory: MemoryContext | None,
    emit: Callable[[dict], None],
    folder_id: str | None = None,
    enable_graph: bool = True,
    graph: nx.Graph | None = None,
    neo4j_driver: Driver | None = None,
) -> RetrievalResult:
    settings = get_settings()
    logger = logging.getLogger(__name__)
    memory = memory or MemoryContext()

    # 前置：全局问题检测 / GraphRAG 强制模式
    global_context = ""
    graph_force = enable_graph and graph and graph.number_of_nodes() > 0 and settings.graph_retrieval_enabled

    if graph_force:
        if is_global_question(question):
            emit({"code": "global_start", "label": "全局问题，启动图谱摘要检索"})
            try:
                global_context = global_search(question, folder_id, graph)
                emit({"code": "global_done", "label": "图谱摘要检索完成" if global_context else "图谱未找到相关摘要"})
            except Exception as exc:
                logger.warning("Global search failed: %s", exc)

        # 向量检索
        vector_chunks, vector_debug = retrieve_once_with_progress(question, conn, emit)

        # 构建 runs（含各阶段结果列表）
        def _dump(c):
            return c.model_dump() if hasattr(c, 'model_dump') else c
        vector_run = {
            "strategy": "graph_vector",
            "query": question,
            "final_results": [_dump(c) for c in vector_chunks],
            "summary": vector_debug.get("summary", {}),
            "dense_results": [_dump(c) for c in vector_debug.get("dense_results", [])],
            "keyword_results": [_dump(c) for c in vector_debug.get("keyword_results", [])],
            "rrf_results": [_dump(c) for c in vector_debug.get("rrf_results", [])],
            "rerank_results": [_dump(c) for c in vector_debug.get("rerank_results", [])],
            "auto_merged_results": [_dump(c) for c in vector_debug.get("auto_merged_results", [])],
            "grade": {},
        }

        # 图谱检索 + 融合
        graph_data_result = None
        try:
            local_result = graph_retrieve(question=question, folder_id=folder_id, graph=graph,
                                          max_hops=settings.graph_max_hops, top_k=settings.graph_top_k_chunks, emit=emit)
            graph_uids = local_result["chunk_uids"] if local_result else []

            if graph_uids:
                fused = fuse_with_vector(vector_chunks, graph_uids, conn, question)
                overlap = len(set(c.chunk_uid for c in vector_chunks) & set(graph_uids))
                graph_data_result = {
                    "entities_matched": local_result["entities_matched"],
                    "graph_chunks_found": len(graph_uids),
                    "overlap_with_vector": overlap,
                    "subgraph": local_result["subgraph"],
                }
                if global_context:
                    graph_data_result["global_context"] = global_context
                emit({"code": "graph_done",
                      "label": f"GraphRAG 融合（向量{len(vector_chunks)}+图谱{len(graph_uids)}→{len(fused)}）",
                      "graph_data": graph_data_result})

                fused_grade = grade_contexts(question, fused)
                emit({"code": "grade_done", "label": "资料充分性判断通过" if fused_grade.can_answer else "资料仍不足",
                      "can_answer": fused_grade.can_answer})
                if fused_grade.can_answer:
                    vector_run["grade"] = fused_grade.model_dump()
                    return RetrievalResult(
                        contexts=fused, retrieval_query=question, rewrite_strategy="graph_local",
                        can_answer=True, grade_reason=fused_grade.reason,
                        missing_information=fused_grade.missing_information,
                        debug_info={"original_question": question, "selected_strategy": "graph_local",
                                    "runs": [vector_run], "graph_data": graph_data_result},
                    )
            else:
                emit({"code": "graph_done", "label": "GraphRAG 未匹配实体，使用向量检索结果"})
        except Exception as exc:
            logger.warning("GraphRAG failed: %s", exc)

        # 回退：纯向量结果 grade
        v_grade = grade_contexts(question, vector_chunks)
        emit({"code": "grade_done", "label": "资料充分性判断通过" if v_grade.can_answer else "资料仍不足",
              "can_answer": v_grade.can_answer})
        if v_grade.can_answer:
            vector_run["grade"] = v_grade.model_dump()
            return RetrievalResult(
                contexts=vector_chunks, retrieval_query=question, rewrite_strategy="graph_vector_fallback",
                can_answer=True, grade_reason=v_grade.reason, missing_information=v_grade.missing_information,
                debug_info={"original_question": question, "selected_strategy": "graph_vector_fallback",
                            "runs": [vector_run], "graph_data": graph_data_result},
            )
        # 都不行 → 继续走 L1 重写流程

    context_reason = explain_contextual_rewrite(question, memory)
    start = perf_counter()
    emit({"code": "context_start", "label": "上下文判断中"})
    used_contextual_rewrite = needs_contextual_rewrite(question, memory)
    base_query = contextualize_question(question, memory)
    context_ms = elapsed_ms(start)
    emit(
        {
            "code": "context_done",
            "label": ("上下文判断通过" if not used_contextual_rewrite else "上下文改写通过") + f"（用时 {context_ms}ms）",
            "query": base_query,
            "reason": context_reason,
            "elapsed_ms": context_ms,
        }
    )

    debug_runs: list[dict] = []

    emit({"code": "query_start", "label": "原问题检索中", "strategy": "original", "query": base_query})
    original_contexts, original_debug = retrieve_once_with_progress(base_query, conn, emit)
    start = perf_counter()
    emit({"code": "grade_start", "label": "资料充分性判断中"})
    original_grade = grade_contexts(question, original_contexts)
    grade_ms = elapsed_ms(start)
    original_debug.setdefault("timings", {})["grade_ms"] = grade_ms
    original_debug["rewrite_trigger_reason"] = explain_should_rewrite(original_contexts)
    emit(
        {
            "code": "grade_done",
            "label": ("资料充分性判断通过" if original_grade.can_answer else "资料不足，准备改写检索") + f"（用时 {grade_ms}ms）",
            "can_answer": original_grade.can_answer,
            "reason": original_grade.reason,
            "elapsed_ms": grade_ms,
        }
    )
    debug_runs.append(
        {
            "strategy": "original",
            "query": base_query,
            **original_debug,
            "grade": original_grade.model_dump(),
        }
    )

    if original_grade.can_answer and not should_rewrite(original_contexts):
        return RetrievalResult(
            contexts=original_contexts,
            retrieval_query=base_query,
            contextualized_question=base_query if used_contextual_rewrite else None,
            used_contextual_rewrite=used_contextual_rewrite,
            can_answer=True,
            grade_reason=original_grade.reason,
            missing_information=original_grade.missing_information,
            debug_info={
                "original_question": question,
                "contextualized_question": base_query,
                "used_contextual_rewrite": used_contextual_rewrite,
                "contextual_rewrite_reason": context_reason,
                "contextual_rewrite_elapsed_ms": context_ms,
                "selected_strategy": "original",
                "selected_query": base_query,
                "runs": debug_runs,
            },
        )

    attempts: list[str] = []
    best_contexts = original_contexts
    best_query = base_query
    best_strategy = "none"
    best_grade = original_grade
    rewrite_order_reason = explain_rewrite_order(base_query)

    for strategy in choose_rewrite_order(base_query):
        start = perf_counter()
        emit({"code": "rewrite_start", "label": f"{strategy} 改写中", "strategy": strategy, "reason": rewrite_order_reason})
        rewritten_query = rewrite_query(base_query, strategy)
        rewrite_ms = elapsed_ms(start)
        attempts.append(f"{strategy}: {rewritten_query}")
        emit({"code": "rewrite_done", "label": f"{strategy} 改写通过（用时 {rewrite_ms}ms）", "strategy": strategy, "query": rewritten_query, "reason": rewrite_order_reason, "elapsed_ms": rewrite_ms})

        rewritten_contexts, rewritten_debug = retrieve_once_with_progress(rewritten_query, conn, emit)
        rewritten_debug.setdefault("timings", {})["rewrite_ms"] = rewrite_ms
        start = perf_counter()
        emit({"code": "grade_start", "label": "资料充分性判断中"})
        rewritten_grade = grade_contexts(question, rewritten_contexts)
        grade_ms = elapsed_ms(start)
        rewritten_debug.setdefault("timings", {})["grade_ms"] = grade_ms
        rewritten_debug["rewrite_trigger_reason"] = explain_should_rewrite(rewritten_contexts)
        emit(
            {
                "code": "grade_done",
                "label": ("资料充分性判断通过" if rewritten_grade.can_answer else "资料仍不足") + f"（用时 {grade_ms}ms）",
                "can_answer": rewritten_grade.can_answer,
                "reason": rewritten_grade.reason,
                "elapsed_ms": grade_ms,
            }
        )

        debug_runs.append(
            {
                "strategy": strategy,
                "query": rewritten_query,
                **rewritten_debug,
                "grade": rewritten_grade.model_dump(),
            }
        )

        if rewritten_grade.can_answer and not should_rewrite(rewritten_contexts):
            return RetrievalResult(
                contexts=rewritten_contexts,
                retrieval_query=rewritten_query,
                contextualized_question=base_query if used_contextual_rewrite else None,
                used_contextual_rewrite=used_contextual_rewrite,
                rewrite_strategy=strategy,
                rewritten_query=rewritten_query,
                rewrite_attempts=attempts,
                can_answer=True,
                grade_reason=rewritten_grade.reason,
                missing_information=rewritten_grade.missing_information,
                debug_info={
                    "original_question": question,
                    "contextualized_question": base_query,
                    "used_contextual_rewrite": used_contextual_rewrite,
                    "contextual_rewrite_reason": context_reason,
                    "contextual_rewrite_elapsed_ms": context_ms,
                    "selected_strategy": strategy,
                    "selected_query": rewritten_query,
                    "rewrite_order_reason": rewrite_order_reason,
                    "rewrite_attempts": attempts,
                    "runs": debug_runs,
                },
            )

        if len(rewritten_contexts) > len(best_contexts):
            best_contexts = rewritten_contexts
            best_query = rewritten_query
            best_strategy = strategy
            best_grade = rewritten_grade

    # GraphRAG 升级
    graph_data: dict | None = None
    graph_contexts = best_contexts
    settings = get_settings()

    if enable_graph and graph and folder_id and settings.graph_retrieval_enabled:
        emit({"code": "graph_start", "label": "RAG不足，升级知识图谱检索"})
        try:
            local_result = graph_retrieve(
                question=base_query, folder_id=folder_id, graph=graph,
                max_hops=settings.graph_max_hops, top_k=settings.graph_top_k_chunks,
            )
            if local_result["chunk_uids"]:
                fused = fuse_with_vector(best_contexts, local_result["chunk_uids"], conn, base_query)
                fused_grade = grade_contexts(question, fused)
                graph_data = {
                    "entities_matched": local_result["entities_matched"],
                    "graph_chunks_found": len(local_result["chunk_uids"]),
                    "subgraph": local_result["subgraph"],
                }
                emit({
                    "code": "graph_done",
                    "label": f"知识图谱检索完成，找到 {len(local_result['chunk_uids'])} 个关联块",
                    "graph_data": graph_data,
                })
                if fused_grade.can_answer:
                    emit({"code": "grade_done", "label": f"知识图谱增强后资料充分", "can_answer": True})
                    return RetrievalResult(
                        contexts=fused, retrieval_query=base_query,
                        contextualized_question=base_query if used_contextual_rewrite else None,
                        used_contextual_rewrite=used_contextual_rewrite,
                        rewrite_strategy=best_strategy if best_strategy != "none" else "graph_local",
                        rewritten_query=best_query if best_strategy != "none" else None,
                        rewrite_attempts=attempts + [f"graph_local: {len(local_result['chunk_uids'])} chunks"],
                        can_answer=True, grade_reason=fused_grade.reason,
                        missing_information=fused_grade.missing_information,
                        debug_info={
                            "original_question": question,
                            "contextualized_question": base_query,
                            "used_contextual_rewrite": used_contextual_rewrite,
                            "contextual_rewrite_reason": context_reason,
                            "contextual_rewrite_elapsed_ms": context_ms,
                            "selected_strategy": "graph_local", "selected_query": base_query,
                            "rewrite_order_reason": rewrite_order_reason,
                            "rewrite_attempts": attempts + [f"graph_local: {len(local_result['chunk_uids'])} chunks"],
                            "runs": debug_runs,
                            "graph_data": graph_data,
                        },
                    )
                graph_contexts = fused
            else:
                emit({"code": "graph_done", "label": "知识图谱未匹配到相关实体"})
        except Exception as exc:
            logger.warning("GraphRAG 失败: %s", exc)
            emit({"code": "graph_done", "label": f"知识图谱检索异常: {str(exc)[:50]}"})

        # Global Search 兜底
        try:
            gs_context = global_search(question, folder_id, graph)
            if gs_context:
                if graph_data is None:
                    graph_data = {}
                graph_data["global_context"] = gs_context
        except Exception as exc:
            logger.warning("Global Search 失败: %s", exc)

    return RetrievalResult(
        contexts=graph_contexts,
        retrieval_query=best_query,
        contextualized_question=base_query if used_contextual_rewrite else None,
        used_contextual_rewrite=used_contextual_rewrite,
        rewrite_strategy=best_strategy,
        rewritten_query=best_query if best_strategy != "none" else None,
        rewrite_attempts=attempts,
        can_answer=False,
        grade_reason=best_grade.reason,
        missing_information=best_grade.missing_information,
        debug_info={
            "original_question": question,
            "contextualized_question": base_query,
            "used_contextual_rewrite": used_contextual_rewrite,
            "contextual_rewrite_reason": context_reason,
            "contextual_rewrite_elapsed_ms": context_ms,
            "selected_strategy": best_strategy,
            "selected_query": best_query,
            "rewrite_order_reason": rewrite_order_reason,
            "rewrite_attempts": attempts,
            "runs": debug_runs,
            **({"graph_data": graph_data} if graph_data else {}),
        },
    )
