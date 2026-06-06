"""层次化 Leiden 社区检测 + LLM 社区摘要生成。

对标 Microsoft GraphRAG:
- Level 0 = 最粗（大主题）→ Level N-1 = 最细（子主题）
- 每层每个社区生成 LLM 摘要
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import community as community_louvain
import networkx as nx
from langsmith import traceable
from openai import OpenAI

from config import get_settings
from graph_store import load_graph
from llm_service import get_chat_client

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/community_reports")


def _ensure_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _reports_path(folder_id: str) -> Path:
    return REPORTS_DIR / f"{folder_id}.json"


# ============================================================
# 层次化社区检测
# ============================================================


@traceable(name="detect_communities_hierarchical", run_type="chain")
def detect_communities_hierarchical(G: nx.Graph, max_levels: int = 3) -> dict[int, dict[str, int]]:
    """层次化 Leiden 聚类。返回 {level: {entity_id: community_id}}。

    Level 0: 在最粗粒度图上跑
    Level 1: 每个 Level 0 社区内部再聚类
    Level 2: 每个 Level 1 社区内部再聚类
    """
    if G.number_of_nodes() == 0:
        return {}

    settings = get_settings()
    min_size = settings.graph_community_min_size

    # 转无向图
    G_undirected = G.to_undirected() if G.is_directed() else G

    all_partitions: dict[int, dict[str, int]] = {}

    # Level 0: 全图聚类
    partition_l0 = community_louvain.best_partition(
        G_undirected, weight="confidence", random_state=42
    )
    all_partitions[0] = _merge_small(partition_l0, G_undirected, min_size)
    logger.info("Level 0: %d communities", len(set(all_partitions[0].values())))

    # Level 1+: 每个上级社区内部再聚类
    for level in range(1, min(max_levels, 3)):
        sub_partition: dict[str, int] = {}
        prev_partition = all_partitions[level - 1]
        community_groups: dict[int, list[str]] = {}
        for nid, cid in prev_partition.items():
            community_groups.setdefault(cid, []).append(nid)

        global_offset = 0
        for cid, members in community_groups.items():
            if len(members) < min_size * 2:
                # 太小不细分
                for nid in members:
                    sub_partition[nid] = cid + global_offset
                global_offset += 1
                continue

            # 提取子图
            subgraph = G_undirected.subgraph(members)
            if subgraph.number_of_edges() == 0:
                for nid in members:
                    sub_partition[nid] = cid + global_offset
                global_offset += 1
                continue

            try:
                sub_result = community_louvain.best_partition(
                    subgraph, weight="confidence", random_state=42 + level
                )
                for nid, sub_cid in sub_result.items():
                    sub_partition[nid] = global_offset + sub_cid
                global_offset += max(sub_result.values()) + 1
            except Exception:
                for nid in members:
                    sub_partition[nid] = cid + global_offset
                global_offset += 1

        sub_partition = _merge_small(sub_partition, G_undirected, min_size)
        all_partitions[level] = sub_partition
        total = len(set(sub_partition.values()))
        logger.info("Level %d: %d communities", level, total)
        if total <= len(set(prev_partition.values())) * 0.3:
            # 社区数大幅下降 → 已足够细，停止
            break

    # 存入 NetworkX 节点
    for level, partition in all_partitions.items():
        for nid, cid in partition.items():
            if nid in G.nodes:
                G.nodes[nid].setdefault("community_ids", {})[level] = cid
                if level == 0:
                    G.nodes[nid]["community_id"] = cid

    return all_partitions


def _merge_small(
    partition: dict[str, int], G: nx.Graph, min_size: int
) -> dict[str, int]:
    """合并过小的社区到邻居大社区。"""
    sizes: dict[int, int] = {}
    for cid in partition.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    small = {cid for cid, s in sizes.items() if s < min_size}
    if not small:
        return partition

    result = dict(partition)
    for nid, cid in list(result.items()):
        if cid in small:
            for neighbor in G.neighbors(nid):
                nc = partition.get(neighbor)
                if nc is not None and nc not in small:
                    result[nid] = nc
                    break
    return result


# ============================================================
# 社区成员
# ============================================================


def get_community_members(G: nx.Graph, partition: dict) -> dict[int, list[dict]]:
    communities: dict[int, list[dict]] = {}
    for nid, cid in partition.items():
        data = G.nodes[nid]
        communities.setdefault(cid, []).append({
            "entity_id": nid,
            "name": data.get("name", ""),
            "type": data.get("type", "OTHER"),
            "description": data.get("description", ""),
            "confidence": data.get("confidence", 0.0),
        })
    return communities


# ============================================================
# 社区摘要
# ============================================================


@traceable(name="generate_community_reports", run_type="chain")
def generate_community_reports(
    folder_id: str,
    G: nx.Graph | None = None,
    partitions: dict[int, dict[str, int]] | None = None,
    max_levels: int = 3,
) -> dict:
    """为每层每个社区生成 LLM 摘要。返回 {level: {community_id: {summary, key_entities}}}。"""
    if G is None:
        G = load_graph(folder_id)
    if G.number_of_nodes() == 0:
        return {}

    if partitions is None:
        partitions = detect_communities_hierarchical(G, max_levels)

    settings = get_settings()
    client = get_chat_client()
    all_reports: dict = {}

    for level, partition in partitions.items():
        communities = get_community_members(G, partition)
        level_reports: dict = {}

        for cid, members in communities.items():
            if len(members) < settings.graph_community_min_size:
                continue

            entities_text = "\n".join(
                f"- {m['name']}（{m['type']}）：{m['description']}"
                for m in sorted(members, key=lambda x: -x['confidence'])[:15]
            )

            prompt = f"""为以下知识图谱社区生成摘要，概括其核心主题。

社区成员实体：
{entities_text}

要求：
1. 用 2-3 句中文概括这个社区的核心主题
2. 列出 3-5 个最重要的实体名称
3. 只输出 JSON：{{"summary": "...", "key_entities": ["..."]}}"""

            try:
                resp = client.chat.completions.create(
                    model=settings.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                content = resp.choices[0].message.content or "{}"

                # 解析 JSON
                import re
                match = re.search(r"\{.*\}", content, flags=re.S)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = {"summary": content[:200], "key_entities": [m["name"] for m in members[:5]]}

                level_reports[str(cid)] = {
                    "summary": data.get("summary", ""),
                    "key_entities": data.get("key_entities", []),
                    "member_count": len(members),
                    "level": level,
                }
            except Exception as exc:
                logger.warning("社区 %d (level %d) 摘要生成失败: %s", cid, level, exc)
                level_reports[str(cid)] = {
                    "summary": f"包含 {len(members)} 个实体: {', '.join(m['name'] for m in members[:5])}",
                    "key_entities": [m["name"] for m in members[:5]],
                    "member_count": len(members),
                    "level": level,
                }

        all_reports[str(level)] = level_reports
        logger.info("Level %d: %d community reports generated", level, len(level_reports))

    # 持久化
    _ensure_dir()
    path = _reports_path(folder_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    logger.info("Community reports saved: %s (%d levels)", path, len(all_reports))

    return all_reports


def load_community_reports(folder_id: str) -> dict:
    path = _reports_path(folder_id)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def has_community_reports(folder_id: str) -> bool:
    return _reports_path(folder_id).exists()
