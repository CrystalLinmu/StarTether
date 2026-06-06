"""NetworkX 图存储层——内存图 CRUD + JSON 持久化。

每个文件夹一个图，存为 data/graphs/{folder_id}.json。
检索使用 NetworkX（不依赖 Neo4j），Neo4j 仅用于前端可视化。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import networkx as nx

from config import get_settings

logger = logging.getLogger(__name__)

GRAPHS_DIR = Path("data/graphs")


def _ensure_dir() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def _graph_path(folder_id: str) -> Path:
    return GRAPHS_DIR / f"{folder_id}.json"


def _graph_to_dict(G: nx.Graph) -> dict:
    """序列化 NetworkX 图为 JSON-safe dict。"""
    return {
        "nodes": [
            {
                "entity_id": nid,
                **{k: v for k, v in data.items()},
            }
            for nid, data in G.nodes(data=True)
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                **{k: val for k, val in data.items()},
            }
            for u, v, data in G.edges(data=True)
        ],
        "version": 1,
    }


def _dict_to_graph(data: dict) -> nx.Graph:
    """从 JSON-safe dict 还原 NetworkX 图。"""
    G = nx.Graph()
    for node in data.get("nodes", []):
        nid = node.pop("entity_id")
        G.add_node(nid, **node)
    for edge in data.get("edges", []):
        src = edge.pop("source")
        tgt = edge.pop("target")
        G.add_edge(src, tgt, **edge)
    return G


# ---- 持久化 ----

def save_graph(folder_id: str, G: nx.Graph) -> None:
    _ensure_dir()
    path = _graph_path(folder_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_graph_to_dict(G), f, ensure_ascii=False, indent=2)
    logger.info("图谱已保存: %s (%d 节点, %d 边)", path, G.number_of_nodes(), G.number_of_edges())


def load_graph(folder_id: str) -> nx.Graph:
    path = _graph_path(folder_id)
    if not path.exists():
        logger.info("图谱文件不存在，返回空图: %s", path)
        return nx.Graph()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return _dict_to_graph(data)


def delete_graph(folder_id: str) -> None:
    path = _graph_path(folder_id)
    if path.exists():
        path.unlink()
        logger.info("图谱已删除: %s", path)


# ---- 增量更新 ----

def upsert_entities(G: nx.Graph, entities: list[dict], document_name: str, folder_id: str) -> nx.Graph:
    """将新实体合并到现有图中，返回更新后的图。

    entities: [{name, type, description, confidence, chunk_uids: [str]}]
    """
    for ent in entities:
        entity_id = ent.get("entity_id") or str(uuid.uuid4())
        existing = None
        # 按 name 精确匹配
        for nid, data in G.nodes(data=True):
            if data.get("name") == ent["name"]:
                existing = nid
                break

        if existing:
            # 合并：更新描述 + 置信度 + 追加 source chunks
            old_conf = G.nodes[existing].get("confidence") or 0
            if ent["confidence"] > old_conf:
                G.nodes[existing]["description"] = ent.get("description", "")
                G.nodes[existing]["confidence"] = ent["confidence"]
            old_chunks: list = G.nodes[existing].get("chunk_uids", [])
            new_chunks = [c for c in ent.get("chunk_uids", []) if c not in old_chunks]
            G.nodes[existing]["chunk_uids"] = old_chunks + new_chunks
            ent["entity_id"] = existing
        else:
            G.add_node(
                entity_id,
                name=ent["name"],
                type=ent.get("type", "OTHER"),
                description=ent.get("description", ""),
                description_embedding=ent.get("description_embedding"),
                confidence=ent.get("confidence", 0.5),
                document_name=document_name,
                folder_id=folder_id,
                chunk_uids=ent.get("chunk_uids", []),
                aliases=[],
                community_id=-1,
                community_ids={},
            )
            ent["entity_id"] = entity_id

    return G


def upsert_relationships(G: nx.Graph, relationships: list[dict]) -> nx.Graph:
    """将关系批量写入图。自动解析 entity_name → entity_id。

    relationships: [{source, target, relation, description, confidence}]
    """
    # 构建 name → id 映射
    name_to_id: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        name_to_id[data.get("name", "")] = nid

    for rel in relationships:
        src_id = name_to_id.get(rel["source"]) or rel.get("source_id")
        tgt_id = name_to_id.get(rel["target"]) or rel.get("target_id")
        if not src_id or not tgt_id:
            continue
        if not G.has_node(src_id) or not G.has_node(tgt_id):
            continue
        G.add_edge(
            src_id,
            tgt_id,
            relation=rel.get("relation", "关联"),
            description=rel.get("description", ""),
            confidence=rel.get("confidence", 0.5),
        )

    return G
