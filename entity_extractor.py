"""实体 / 关系提取管道。

采用 Microsoft GraphRAG 官方提示词结构（分隔符格式 + Gleaning 循环），
适配中文文档。提取结果写入 NetworkX 图 + 同步 Neo4j（可选）。
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
from langsmith import traceable
from neo4j import Driver
from openai import OpenAI

from config import get_settings
from entity_embedder import embed_texts, similarity
from graph_store import load_graph, save_graph, upsert_entities, upsert_relationships
from llm_service import get_chat_client

logger = logging.getLogger(__name__)

# ============================================================
# GraphRAG 官方提示词（中文适配版）
# 来源: microsoft/graphrag index/graph/extractors/graph/prompts.py
# ============================================================

ENTITY_TYPES = "人物, 组织, 地点, 概念, 时间, 事件, 技术, 产品, 法律法规, 其他"

GRAPH_EXTRACTION_PROMPT = """你是一个知识图谱实体与关系提取器。请从给定文本中提取所有符合类型的实体，以及它们之间明确的关系。

-目标-
给定一个文本和实体类型列表，识别所有符合这些类型的实体，以及所有已识别实体之间的关系。

-步骤-
1. 识别所有实体。对每个实体提取以下信息：
   - entity_name: 实体名称，使用原文中的规范表述
   - entity_type: 从以下类型中选择：{entity_types}
   - entity_description: 用一句话描述该实体的属性和行为
   格式：("entity"|entity_name|entity_type|entity_description)

2. 从步骤1中识别的实体中，找出彼此之间有明确语义关联的实体对。对每一对提取：
   - source_entity: 源实体名称（必须与步骤1中某个 entity_name 完全一致）
   - target_entity: 目标实体名称（必须与步骤1中某个 entity_name 完全一致）
   - relationship_description: 用简短中文描述两者之间的关系
   - relationship_strength: 关系强度数值 1-10（1=很弱, 10=极强）
   格式：("relationship"|source_entity|target_entity|relationship_description|relationship_strength)

3. 识别所有"名称相同但描述不同"的实体，进行合并。只保留最关键的那个。

4. 将所有 entity 和 relationship 作为单个列表返回，每条独占一行。

5. 输出结束后返回 completion 标记：{completion_delimiter}

-示例-
文本：
刘秀，字文叔，东汉开国皇帝。他带领云台二十八将北伐，其中寇恂担任副将。
寇恂在军中发明了十二转运法，大幅降低粮草损耗。
十二转运法在宛城前线首次使用，节约了约30%的军粮。

输出：
("entity"|刘秀|人物|东汉开国皇帝，北伐统帅)
("entity"|寇恂|人物|云台二十八将之一，北伐副将)
("entity"|十二转运法|概念|寇恂发明的后勤运输方法，降低粮草损耗)
("entity"|宛城|地点|北伐前线战场)
("relationship"|刘秀|寇恂|刘秀任命寇恂为北伐副将|8)
("relationship"|寇恂|十二转运法|寇恂发明了十二转运法|9)
("relationship"|十二转运法|宛城|十二转运法在宛城首次使用|6)
{completion_delimiter}

-真实数据-
实体类型：{entity_types}
文本：{input_text}
输出："""

CONTINUE_PROMPT = """上文最后一次提取遗漏了很多实体。请从下文中补充遗漏的实体。

-步骤-
1. 识别所有实体。提取 entity_name / entity_type / entity_description。
   格式：("entity"|entity_name|entity_type|entity_description)

2. 识别有明确语义关联的实体对。提取 source_entity / target_entity / relationship_description / relationship_strength。
   格式：("relationship"|source_entity|target_entity|relationship_description|relationship_strength)

3. 输出结束后返回 completion 标记：{completion_delimiter}

请务必只输出上述格式的数据，不要重复之前已经提取过的实体。

实体类型：{entity_types}
文本：{input_text}
输出："""

LOOP_PROMPT = """是否还有实体未被提取？回答 YES 或 NO。"""

COMPLETION_DELIMITER = "##COMPLETE##"


def _build_extraction_prompt(chunks_text: str) -> str:
    return GRAPH_EXTRACTION_PROMPT.format(
        entity_types=ENTITY_TYPES,
        input_text=chunks_text,
        completion_delimiter=COMPLETION_DELIMITER,
    )


def _build_continue_prompt(chunks_text: str) -> str:
    return CONTINUE_PROMPT.format(
        entity_types=ENTITY_TYPES,
        input_text=chunks_text,
        completion_delimiter=COMPLETION_DELIMITER,
    )


# ============================================================
# 解析器
# ============================================================


def parse_entity_output(text: str) -> dict:
    """解析 GraphRAG 分隔符格式输出 → {entities: [...], relationships: [...]}。"""
    entities: list[dict] = []
    relationships: list[dict] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line or line == COMPLETION_DELIMITER:
            continue

        # 匹配 ("entity"|...) 或 ("relationship"|...)
        if line.startswith('("entity"') or line.startswith('("entity') or line.startswith('（"entity"'):
            parts = _split_tuple(line)
            if len(parts) >= 3:
                entities.append({
                    "name": parts[0].strip(),
                    "type": parts[1].strip(),
                    "description": parts[2].strip() if len(parts) > 2 else "",
                    "confidence": 0.8,
                })
        elif line.startswith('("relationship"') or line.startswith('（"relationship"'):
            parts = _split_tuple(line)
            if len(parts) >= 4:
                strength = 5
                try:
                    strength = int(parts[3].strip()) if len(parts) > 3 else 5
                except ValueError:
                    strength = 5
                relationships.append({
                    "source": parts[0].strip(),
                    "target": parts[1].strip(),
                    "relation": parts[2].strip() if len(parts) > 2 else "关联",
                    "description": parts[2].strip() if len(parts) > 2 else "",
                    "confidence": min(1.0, strength / 10),
                })

    return {"entities": entities, "relationships": relationships}


def _split_tuple(line: str) -> list[str]:
    """解析 ("type"|field1|field2|...) 格式，返回 [field1, field2, ...]。

    示例: ("entity"|刘秀|人物|东汉开国皇帝) → ["刘秀", "人物", "东汉开国皇帝"]
    """
    line = line.strip()
    # 匹配 ("label"|field1|field2|...)  — 兼容 ) 和 全角 ）
    m = re.match(r'\(\s*"([^"]+)"\s*\|\s*(.+)[)）]\s*$', line)
    if not m:
        return []
    inner = m.group(2)
    # 按 | 分割，但保留括号内的内容
    parts = inner.split("|")
    return [p.strip() for p in parts]


# ============================================================
# 提取引擎
# ============================================================


def _deduplicate_entities(
    new_entities: list[dict],
    existing_entities: list[dict],  # [(name, description, embedding)]
    threshold: float | None = None,
) -> list[dict]:
    """对同批实体做去重：embedding 余弦相似度 >0.82 → 同一实体。"""
    settings = get_settings()
    if threshold is None:
        threshold = settings.entity_similarity_threshold

    if not new_entities:
        return []

    # 为新实体生成 description embedding
    descs = [e.get("description", e["name"]) for e in new_entities]
    embeddings = embed_texts(descs)
    for ent, emb in zip(new_entities, embeddings):
        ent["description_embedding"] = emb

    result: list[dict] = []
    for ent, emb in zip(new_entities, embeddings):
        name = ent["name"]

        # 和已有实体比较
        is_dup = False
        for exist_name, exist_desc, exist_emb in existing_entities:
            if exist_emb is None:
                continue
            sim = similarity(emb, exist_emb)
            if sim >= threshold:
                logger.info("实体合并: '%s' → '%s' (相似度 %.3f)", name, exist_name, sim)
                is_dup = True
                break
        if is_dup:
            continue

        # 和同批已添加的比较
        for added in result:
            added_emb = added.get("description_embedding")
            if added_emb is None:
                continue
            if similarity(emb, added_emb) >= threshold:
                is_dup = True
                break
        if is_dup:
            continue

        result.append(ent)

    return result


def _collect_existing_entities(G) -> list[tuple[str, str, list[float] | None]]:
    """从现有图中收集 (name, description, embedding)。"""
    result = []
    for _, data in G.nodes(data=True):
        name = data.get("name", "")
        desc = data.get("description", "")
        emb = data.get("description_embedding")
        result.append((name, desc, emb))
    return result


def _merge_entity_descriptions(
    client: OpenAI, entities: list[dict], target_name: str
) -> str:
    """LLM 合并多个描述为一个精准描述。"""
    if len(entities) == 1:
        return entities[0].get("description", "")

    descriptions = "\n".join(
        f"{i+1}. {e['name']}: {e.get('description', '')}"
        for i, e in enumerate(entities[:5])
    )

    prompt = f"""将以下关于同一实体的多条描述合并为一句精准的描述（不超过40字）：
{descriptions}

合并后的描述："""
    resp = client.chat.completions.create(
        model=get_settings().chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or descriptions).strip()[:120]


@traceable(name="extract_entities_batch", run_type="llm")
def _extract_batch(
    client: OpenAI,
    chunks_text: str,
    existing_entities: list[tuple[str, str, list[float] | None]],
    max_gleanings: int = 1,
) -> dict:
    """单批 L2 chunks → LLM 提取 + Gleaning 循环。"""
    settings = get_settings()
    all_entities: list[dict] = []
    all_relationships: list[dict] = []

    # 首次提取
    prompt = _build_extraction_prompt(chunks_text)
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    result = parse_entity_output(content)
    all_entities.extend(result["entities"])
    all_relationships.extend(result["relationships"])

    # Gleaning 循环
    for _ in range(max_gleanings):
        fresh = _deduplicate_entities(all_entities, existing_entities)
        if not fresh:
            break

        loop_resp = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "user", "content": _build_extraction_prompt(chunks_text)},
                {"role": "assistant", "content": content},
                {"role": "user", "content": LOOP_PROMPT},
            ],
            temperature=0,
        )
        answer = (loop_resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("NO"):
            break

        continue_prompt = _build_continue_prompt(chunks_text)
        resp2 = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": continue_prompt}],
            temperature=0.2,
        )
        content = resp2.choices[0].message.content or ""
        result2 = parse_entity_output(content)
        all_entities.extend(result2["entities"])
        all_relationships.extend(result2["relationships"])

    # 最终去重（用 embedding 相似度）
    deduped = _deduplicate_entities(all_entities, existing_entities)
    return {"entities": deduped, "relationships": all_relationships}


@traceable(name="extract_entities_from_document", run_type="chain")
def extract_entities_from_document(
    document_name: str,
    folder_id: str,
    conn: psycopg.Connection,
    neo4j_driver: Driver | None = None,
) -> int:
    """从文档的 L2 chunks 中提取实体和关系，写入 NetworkX 图 + Neo4j。

    返回提取的实体数。
    """
    # 1. 加载 L2 chunks
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_uid, content FROM document_chunks
               WHERE document_name = %s AND chunk_level = 2
               ORDER BY chunk_index""",
            (document_name,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.info("文档 %s 无 L2 chunk，跳过实体提取", document_name)
        return 0

    l2_chunks = [{"chunk_uid": r[0], "content": r[1]} for r in rows]

    # 2. 加载现有图
    G = load_graph(folder_id)
    existing_entities = _collect_existing_entities(G)

    # 3. 分批提取
    settings = get_settings()
    client = get_chat_client()
    batch_size = settings.entity_extraction_batch_size
    all_entities: list[dict] = []
    all_relationships: list[dict] = []

    for i in range(0, len(l2_chunks), batch_size):
        batch = l2_chunks[i : i + batch_size]
        chunks_text = "\n\n".join(c["content"] for c in batch)
        batch_chunk_uids = [c["chunk_uid"] for c in batch]

        result = _extract_batch(client, chunks_text, existing_entities)
        for ent in result["entities"]:
            ent["chunk_uids"] = batch_chunk_uids
            # 加到 existing_entities 供后续批次去重
            existing_entities.append((
                ent["name"],
                ent.get("description", ""),
                ent.get("description_embedding"),
            ))
        all_entities.extend(result["entities"])
        all_relationships.extend(result["relationships"])
        logger.info(
            "批次 %d/%d: 提取 %d 实体, %d 关系",
            i // batch_size + 1,
            (len(l2_chunks) + batch_size - 1) // batch_size,
            len(result["entities"]),
            len(result["relationships"]),
        )

    if not all_entities:
        logger.info("文档 %s 未提取到实体", document_name)
        return 0

    # 4. 跨批次去重（用 embedding 相似度）
    final_entities = _deduplicate_entities(all_entities, [])
    logger.info(
        "文档 %s: 去重后 %d 实体 (原始 %d), %d 关系",
        document_name, len(final_entities), len(all_entities), len(all_relationships),
    )

    # 5. 生成 entity_id + upsert 到 NetworkX 图
    # 构建 name → entity_id 映射
    name_to_id: dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        name_to_id[data.get("name", "")] = nid

    for ent in final_entities:
        if ent["name"] in name_to_id:
            ent["entity_id"] = name_to_id[ent["name"]]
        else:
            ent_id = str(uuid.uuid4())
            ent["entity_id"] = ent_id
            name_to_id[ent["name"]] = ent_id

    G = upsert_entities(G, final_entities, document_name, folder_id)

    # 解析关系中的 entity_name → entity_id
    for rel in all_relationships:
        src_id = name_to_id.get(rel["source"])
        tgt_id = name_to_id.get(rel["target"])
        if src_id:
            rel["source_id"] = src_id
        if tgt_id:
            rel["target_id"] = tgt_id

    G = upsert_relationships(G, all_relationships)
    save_graph(folder_id, G)

    # 6. 同步到 Neo4j（可选）
    if neo4j_driver:
        try:
            from neo4j_store import sync_to_neo4j
            sync_to_neo4j(neo4j_driver, final_entities, all_relationships, folder_id)
        except Exception as exc:
            logger.warning("Neo4j 同步失败（非致命）: %s", exc)

    return len(final_entities)


# ============================================================
# 异步包装
# ============================================================


_executor = ThreadPoolExecutor(max_workers=2)


async def run_entity_extraction(
    document_name: str,
    folder_id: str,
    conn_params: dict,
    neo4j_driver: Driver | None = None,
) -> int:
    """异步启动实体提取（在单独线程中运行，不阻塞事件循环）。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_extract,
        document_name,
        folder_id,
        conn_params,
        neo4j_driver,
    )


def _sync_extract(
    document_name: str,
    folder_id: str,
    conn_params: dict,
    neo4j_driver: Driver | None,
) -> int:
    with psycopg.connect(**conn_params) as conn:
        return extract_entities_from_document(
            document_name, folder_id, conn, neo4j_driver
        )
