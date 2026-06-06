"""GraphRAG 检索层 — Local Search + Global Search + 多级路由。

Local Search:  实体感知检索，LLM 提取问题实体 → NetworkX 图遍历 → 返回关联 chunks
Global Search: 社区摘要检索，Map-Reduce 筛选相关社区摘要 → 拼入 prompt
多级路由:      RAG → 重写 → GraphRAG local → GraphRAG global
"""

from __future__ import annotations

import logging
import re

import networkx as nx
import psycopg
from langsmith import traceable
from openai import OpenAI

from config import get_runtime, get_settings
from entity_embedder import embed_text, embed_texts, similarity
from graph_store import load_graph
from llm_service import get_chat_client
from schemas import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

# ============================================================
# 全局问题判断（规则，零成本）
# ============================================================

GLOBAL_SIGNALS = (
    "总结", "概括", "归纳", "梳理", "综述", "回顾",
    "概览", "全景", "全貌", "框架", "脉络",
    "主题", "专题", "主线", "结构",
    "讲了什么", "说了什么", "都有什么", "包含什么",
    "有哪些", "主要讲什么", "介绍", "简介",
)

OPEN_ENDED_PATTERNS = (
    "怎么样", "是什么", "有哪些", "什么是",
    "如何", "怎么", "为什么",
)


def is_global_question(question: str) -> bool:
    """规则判断是否为全局/宏问题。"""
    q = question.strip()

    if any(signal in q for signal in GLOBAL_SIGNALS):
        return True

    has_proper = bool(re.search(r'《|》|"|"|"|\d+年|\d+条|\d+款', q))
    if not has_proper and len(q) <= 15:
        if any(p in q for p in OPEN_ENDED_PATTERNS):
            return True

    return False


# ============================================================
# 升级判断（RAG 结果质量检验）
# ============================================================


def should_escalate_to_graph(result: RetrievalResult) -> str | None:
    """判断是否需要从 RAG 升级到 GraphRAG local。返回升级原因，None = 不需要。"""
    if not result.can_answer:
        return "grade_rejected"

    if len(result.contexts) <= 1:
        return "low_recall"

    scores = [c.rerank_score or c.rrf_score or 0 for c in result.contexts]
    best = max(scores) if scores else 0
    if best < 0.3:
        return "low_confidence"

    if result.rewrite_attempts and len(result.rewrite_attempts) >= 2:
        return "multiple_rewrites"

    return None


# ============================================================
# Local Search — 实体感知检索
# ============================================================


@traceable(name="extract_query_entities", run_type="llm")
def extract_query_entities(question: str, client: OpenAI | None = None) -> list[str]:
    """LLM 从问题中提取关键实体名。"""
    settings = get_settings()
    if client is None:
        client = get_chat_client()

    prompt = f"""从以下问题中提取关键实体名称（人物、地点、概念、组织、技术、产品等），每行一个。
只输出实体名称，不要解释，不要编号。
如果没有实体，输出 NONE。

问题：{question}
实体："""

    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.upper() == "NONE":
        return []

    return [line.strip("- •1234567890. ") for line in text.split("\n") if line.strip() and line.strip().upper() != "NONE"]


def _match_entities_in_graph(
    entity_names: list[str],
    G: nx.Graph,
    min_confidence: float = 0.5,
    embedding_threshold: float = 0.50,
) -> list[str]:
    """混合匹配：精确+包含+embedding 相似度。每个实体名单独算 embedding。"""
    matched_ids: list[str] = []
    matched_names: set[str] = set()

    if not entity_names:
        return []

    for name in entity_names:
        name_lower = name.lower()
        best_id: str | None = None
        best_score: float = 0

        # 为当前实体名单独算 embedding
        name_emb = embed_text(name)

        for nid, data in G.nodes(data=True):
            node_name = str(data.get("name", ""))
            node_lower = node_name.lower()
            conf = data.get("confidence", 0.0)
            if conf < min_confidence:
                continue

            # 精确匹配
            if node_lower == name_lower:
                best_id = nid
                break

            # 包含匹配
            if len(name_lower) >= 2 and (name_lower in node_lower or node_lower in name_lower):
                if best_score < 0.9:
                    best_id = nid
                    best_score = 0.9

            # embedding 语义匹配
            if best_score < 0.8:
                node_emb = data.get("description_embedding")
                if node_emb and name_emb:
                    sim = similarity(name_emb, node_emb)
                    if sim > embedding_threshold and sim > best_score:
                        best_id = nid
                        best_score = sim

        if best_id:
            matched_node_name = G.nodes[best_id].get("name", "")
            if matched_node_name not in matched_names:
                matched_ids.append(best_id)
                matched_names.add(matched_node_name)

    return matched_ids


def _traverse_and_collect(
    matched_ids: list[str],
    G: nx.Graph,
    max_hops: int = 2,
    top_k: int = 5,
) -> tuple[list[str], dict]:
    """从匹配实体出发做 n-hop 图遍历，收集关联 chunk_uids + 子图数据。"""
    if not matched_ids:
        return [], {"nodes": [], "edges": []}

    # BFS 遍历
    visited: set[str] = set()
    current = set(matched_ids)
    for _ in range(max_hops + 1):
        visited |= current
        neighbors: set[str] = set()
        for nid in current:
            for neighbor in G.neighbors(nid):
                if neighbor not in visited:
                    neighbors.add(neighbor)
        current = neighbors

    # 收集 chunk_uids（按实体置信度初排）
    chunk_scores: dict[str, float] = {}
    for nid in visited:
        chunk_uids = G.nodes[nid].get("chunk_uids", [])
        node_confidence = G.nodes[nid].get("confidence", 0.5)
        for cid in chunk_uids:
            chunk_scores[cid] = max(chunk_scores.get(cid, 0), node_confidence)

    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: -x[1])
    # 取 top_k * 2 候选，后续可语义重排
    candidates = [c[0] for c in sorted_chunks[: min(top_k * 3, len(sorted_chunks))]]
    chunk_uids = candidates[:top_k]

    # 子图数据（给前端可视化）
    sub_nodes = [
        {
            "id": nid,
            "label": G.nodes[nid].get("name", nid),
            "type": G.nodes[nid].get("type", "OTHER"),
            "group": G.nodes[nid].get("type", "OTHER"),
            "description": G.nodes[nid].get("description", ""),
            "confidence": G.nodes[nid].get("confidence", 0.0),
            "chunk_count": len(G.nodes[nid].get("chunk_uids", [])),
        }
        for nid in visited
    ]
    sub_edges = [
        {
            "id": f"{u}-{d.get('relation', '')}-{v}",
            "source": u,
            "target": v,
            "label": d.get("relation", ""),
            "description": d.get("description", ""),
        }
        for u, v, d in G.edges(data=True)
        if u in visited and v in visited
    ]

    return chunk_uids, {"nodes": sub_nodes, "edges": sub_edges}


@traceable(name="graph_retrieve", run_type="retriever")
def graph_retrieve(
    question: str,
    folder_id: str,
    graph: nx.Graph | None = None,
    max_hops: int = 2,
    top_k: int = 5,
    emit: callable | None = None,
) -> dict:
    """Local Search: 实体感知检索 + 图遍历。返回 {chunk_uids, subgraph, entities_matched}。"""
    settings = get_settings()
    if max_hops <= 0:
        max_hops = settings.graph_max_hops
    if top_k <= 0:
        top_k = get_runtime("graph_top_k_chunks", settings.graph_top_k_chunks)

    if graph is None:
        graph = load_graph(folder_id)

    if graph.number_of_nodes() == 0:
        return {"chunk_uids": [], "subgraph": {"nodes": [], "edges": []}, "entities_matched": []}

    # ① 提取问题实体
    t0 = __import__('time').perf_counter()
    if emit: emit({"code": "graph_entity_start", "label": "① LLM 提取问题实体"})
    client = get_chat_client()
    entity_names = extract_query_entities(question, client)
    t1 = __import__('time').perf_counter()

    if not entity_names:
        if emit: emit({"code": "graph_entity_done", "label": "未识别出实体名", "elapsed_ms": int((t1-t0)*1000)})
        return {"chunk_uids": [], "subgraph": {"nodes": [], "edges": []}, "entities_matched": []}

    if emit: emit({"code": "graph_entity_done", "label": f"识别实体: {', '.join(entity_names[:8])}", "elapsed_ms": int((t1-t0)*1000)})

    # ② 图中匹配实体
    t2 = __import__('time').perf_counter()
    if emit: emit({"code": "graph_match_start", "label": "② 知识图谱中匹配实体"})
    matched_ids = _match_entities_in_graph(entity_names, graph)
    t3 = __import__('time').perf_counter()
    if emit:
        matched_names = [graph.nodes[mid]["name"] for mid in matched_ids]
        emit({"code": "graph_match_done", "label": f"命中 {len(matched_ids)} 个实体: {', '.join(matched_names[:5])}" if matched_ids else "未匹配到图中实体", "elapsed_ms": int((t3-t2)*1000)})

    # ③ 图遍历收集邻居
    t4 = __import__('time').perf_counter()
    if emit: emit({"code": "graph_traverse_start", "label": f"③ 从 {len(matched_ids)} 个实体出发做 {max_hops}-hop 图遍历"})
    chunk_uids, subgraph = _traverse_and_collect(matched_ids, graph, max_hops, top_k)
    t5 = __import__('time').perf_counter()
    if emit:
        neighbor_count = len(subgraph["nodes"]) - len(matched_ids)
        emit({"code": "graph_traverse_done", "label": f"遍历到 {neighbor_count} 个邻居实体，收集 {len(chunk_uids)} 个关联 chunk", "elapsed_ms": int((t5-t4)*1000)})

    logger.info(
        "GraphRAG local: 问题实体=%s, 匹配=%d, chunks=%d",
        entity_names, len(matched_ids), len(chunk_uids),
    )

    return {
        "chunk_uids": chunk_uids,
        "subgraph": subgraph,
        "entities_matched": [
            {"name": graph.nodes[mid]["name"], "type": graph.nodes[mid].get("type", "")}
            for mid in matched_ids
        ],
    }


# ============================================================
# Global Search — 社区摘要检索
# ============================================================


@traceable(name="global_search", run_type="retriever")
def global_search(
    question: str,
    folder_id: str,
    graph: nx.Graph | None = None,
    communities: dict | None = None,
    community_reports: dict | None = None,
) -> str:
    """Global Search: 筛选相关社区摘要，拼成上下文文本。"""
    if community_reports is None:
        from community_detector import load_community_reports
        community_reports = load_community_reports(folder_id)

    if not community_reports:
        return ""

    # 用 LLM 筛选相关社区摘要
    settings = get_settings()
    client = get_chat_client()

    # 构建候选列表（兼容新旧格式）
    candidates = []
    # 新格式: {level: {cid: {summary, key_entities}}}
    for level_key, level_reports in community_reports.items():
        if not isinstance(level_reports, dict):
            continue
        for cid, info in level_reports.items():
            if isinstance(info, dict):
                candidates.append({
                    "community_id": f"L{level_key}-{cid}",
                    "summary": info.get("summary", ""),
                    "key_entities": info.get("key_entities", []),
                })
    # 旧格式: {cid: {summary, key_entities}}（兼容）
    if not candidates:
        for cid, info in community_reports.items():
            if isinstance(info, dict):
                candidates.append({
                    "community_id": str(cid),
                    "summary": info.get("summary", ""),
                    "key_entities": info.get("key_entities", []),
                })

    if len(candidates) > 20:
        # 太多则只取 key_entities 含问题关键词的
        filtered = []
        for c in candidates:
            if any(e.lower() in question.lower() for e in c["key_entities"]):
                filtered.append(c)
        candidates = filtered[:20] if filtered else candidates[:20]

    # Map: 逐条评估相关性
    relevant_texts = []
    for c in candidates:
        prompt = f"""评估以下社区摘要与用户问题的相关性。如果相关，输出 RELEVANT 和评分 1-10；如果不相关，输出 IRRELEVANT。

社区关键词：{', '.join(c['key_entities'][:10])}
社区摘要：{c['summary'][:500]}

用户问题：{question}

判断（RELEVANT 分数 或 IRRELEVANT）："""
        resp = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("RELEVANT") or answer.startswith("相关"):
            relevant_texts.append(f"【社区{c['community_id']}】\n{c['summary']}")

    if not relevant_texts:
        return ""

    return "\n\n---\n\n".join(relevant_texts[:5])


# ============================================================
# 结果融合
# ============================================================


def fetch_chunks_by_uid(conn: psycopg.Connection, uids: list[str]) -> list[RetrievedChunk]:
    """从 PG 根据 chunk_uid 列表获取完整 chunk 数据。"""
    if not uids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, chunk_uid, parent_uid, chunk_level, document_name,
                      section_title, position_hint, content
               FROM document_chunks
               WHERE chunk_uid = ANY(%s)""",
            (uids,),
        )
        rows = cur.fetchall()
    return [
        RetrievedChunk(
            id=r[0],
            chunk_uid=r[1],
            parent_uid=r[2],
            chunk_level=r[3],
            document_name=r[4],
            section_title=r[5],
            position_hint=r[6],
            content=r[7],
        )
        for r in rows
    ]


def fuse_with_vector(
    vector_chunks: list[RetrievedChunk],
    graph_chunk_uids: list[str],
    conn: psycopg.Connection,
    question: str = "",
    top_k: int | None = None,
    boost_overlap: float = 1.2,
) -> list[RetrievedChunk]:
    """融合向量检索和图谱检索结果。图谱独有 chunk 走 Rerank 获取真实分数。"""
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k

    if not graph_chunk_uids:
        return vector_chunks[:top_k]

    # 图谱独有 chunk
    graph_uids_set = set(graph_chunk_uids)
    graph_only_uids = graph_uids_set - {c.chunk_uid for c in vector_chunks}
    graph_only = fetch_chunks_by_uid(conn, list(graph_only_uids))

    # 交集加权
    for chunk in vector_chunks:
        if chunk.chunk_uid in graph_uids_set:
            if chunk.rerank_score:
                chunk.rerank_score *= boost_overlap
            elif chunk.dense_score:
                chunk.dense_score *= boost_overlap
            chunk.merge_info = {"source": "vector_graph_overlap"}

    # 图谱独有的走 Rerank 获取真实分数（不再用 0.3）
    if graph_only and question:
        try:
            from reranker import rerank_chunks
            graph_only = rerank_chunks(question, graph_only)
        except Exception:
            for chunk in graph_only:
                chunk.rerank_score = chunk.rrf_score or chunk.dense_score or 0.3
                chunk.merge_info = {"source": "graph_only"}
    else:
        for chunk in graph_only:
            chunk.rerank_score = chunk.rrf_score or chunk.dense_score or 0.3
            chunk.merge_info = {"source": "graph_only"}

    merged = list(vector_chunks) + graph_only
    merged.sort(
        key=lambda c: c.rerank_score or c.rrf_score or c.dense_score or 0,
        reverse=True,
    )

    return merged[:top_k]
