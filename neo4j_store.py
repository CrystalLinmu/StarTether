"""Neo4j 连接管理 + 索引创建（仅用于可视化，检索不依赖）。"""

from __future__ import annotations

import logging

from neo4j import Driver, GraphDatabase

from config import get_settings

logger = logging.getLogger(__name__)


def get_neo4j_driver() -> Driver | None:
    """获取 Neo4j 驱动；未配置密码时返回 None。"""
    settings = get_settings()
    if not settings.neo4j_password:
        logger.info("Neo4j 未配置，跳过")
        return None
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        logger.info("Neo4j 连接成功")
        return driver
    except Exception as exc:
        logger.warning("Neo4j 连接失败: %s", exc)
        return None


def ensure_neo4j_constraints(driver: Driver) -> None:
    """创建必要的索引和约束。"""
    constraints = [
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX entity_folder_idx IF NOT EXISTS FOR (e:Entity) ON (e.folder_id)",
        "CREATE INDEX entity_document_idx IF NOT EXISTS FOR (e:Entity) ON (e.document_name)",
        "CREATE INDEX chunk_folder_idx IF NOT EXISTS FOR (c:Chunk) ON (c.folder_id)",
    ]
    with driver.session(database=get_settings().neo4j_database) as session:
        for stmt in constraints:
            try:
                session.run(stmt)
            except Exception as exc:
                logger.debug("Neo4j 约束/索引跳过: %s", exc)


def sync_to_neo4j(
    driver: Driver,
    entities: list[dict],
    relationships: list[dict],
    folder_id: str,
) -> int:
    """将 NetworkX 图的实体和关系批量写入 Neo4j。

    entities: [{entity_id, name, type, description, document_name, folder_id, confidence}]
    relationships: [{source_id, target_id, relation, description, confidence}]
    """
    database = get_settings().neo4j_database
    entity_count = 0
    rel_count = 0

    if entities:
        with driver.session(database=database) as session:
            result = session.run(
                """UNWIND $entities AS e
                   MERGE (n:Entity {entity_id: e.entity_id})
                   ON CREATE SET n.name = e.name, n.type = e.type,
                                 n.description = e.description, n.document_name = e.document_name,
                                 n.folder_id = e.folder_id, n.confidence = e.confidence,
                                 n.created_at = datetime()
                   ON MATCH SET n.description = CASE WHEN e.confidence > COALESCE(n.confidence, 0)
                                                     THEN e.description ELSE n.description END,
                               n.confidence = CASE WHEN e.confidence > COALESCE(n.confidence, 0)
                                                    THEN e.confidence ELSE n.confidence END
                   RETURN count(n)""",
                entities=entities,
            )
            entity_count = result.single()[0]

    if relationships:
        with driver.session(database=database) as session:
            result = session.run(
                """UNWIND $rels AS r
                   MATCH (src:Entity {entity_id: r.source_id})
                   MATCH (tgt:Entity {entity_id: r.target_id})
                   MERGE (src)-[rel:RELATES_TO {relation: r.relation}]->(tgt)
                   SET rel.description = r.description, rel.confidence = r.confidence
                   RETURN count(rel)""",
                rels=relationships,
            )
            rel_count = result.single()[0]

    logger.info("Neo4j 同步: %d 实体, %d 关系 (folder=%s)", entity_count, rel_count, folder_id)
    return entity_count + rel_count


def delete_folder_graph(driver: Driver, folder_id: str) -> None:
    """删除指定文件夹下的所有实体和关系。"""
    database = get_settings().neo4j_database
    with driver.session(database=database) as session:
        session.run(
            "MATCH (e:Entity {folder_id: $folder_id}) DETACH DELETE e",
            folder_id=folder_id,
        )
    logger.info("Neo4j 删除文件夹图谱: %s", folder_id)


def delete_document_graph(driver: Driver, document_name: str, folder_id: str) -> None:
    """删除文档的实体（只删除无其他文档引用的实体）。"""
    database = get_settings().neo4j_database
    with driver.session(database=database) as session:
        session.run(
            """MATCH (e:Entity {document_name: $doc, folder_id: $fid})
               WHERE NOT (e)<-[:RELATES_TO]-() OR size([(e)<-[:RELATES_TO]-() | 1]) = 0
               DETACH DELETE e""",
            doc=document_name,
            fid=folder_id,
        )
    logger.info("Neo4j 删除文档图谱: %s", document_name)


def query_subgraph(
    driver: Driver,
    folder_id: str,
    entity_names: list[str] | None = None,
    max_nodes: int = 80,
) -> dict:
    """查询文件夹下的子图，返回 {nodes: [...], edges: [...]} 供前端可视化。"""
    database = get_settings().neo4j_database
    nodes: list[dict] = []
    edges: list[dict] = []

    with driver.session(database=database) as session:
        if entity_names:
            result = session.run(
                """MATCH (e:Entity)
                   WHERE e.folder_id = $folder_id AND e.name IN $names
                   WITH collect(e) AS matched
                   UNWIND matched AS e
                   OPTIONAL MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
                   WHERE neighbor.folder_id = $folder_id
                   RETURN DISTINCT e, r, neighbor
                   LIMIT $limit""",
                folder_id=folder_id,
                names=entity_names,
                limit=max_nodes * 2,
            )
        else:
            result = session.run(
                """MATCH (e:Entity)-[r:RELATES_TO]->(e2:Entity)
                   WHERE e.folder_id = $folder_id
                   RETURN e, r, e2
                   LIMIT $limit""",
                folder_id=folder_id,
                limit=max_nodes * 2,
            )

        seen_nodes: set[str] = set()
        seen_edges: set[str] = set()

        for record in result:
            e = record.get("e")
            if e and e.get("entity_id") not in seen_nodes:
                seen_nodes.add(e.get("entity_id"))
                nodes.append({
                    "id": e.get("entity_id"),
                    "label": e.get("name", ""),
                    "type": e.get("type", "OTHER"),
                    "group": e.get("type", "OTHER"),
                    "description": e.get("description") or "",
                    "confidence": e.get("confidence") or 0.0,
                    "chunk_count": 0,
                })

            rel = record.get("r")
            e2 = record.get("e2")
            if rel and e2:
                src = e.get("entity_id") if e else None
                tgt = e2.get("entity_id")
                edge_id = f"{src}-{rel.get('relation', '')}-{tgt}"
                if edge_id not in seen_edges and src:
                    seen_edges.add(edge_id)
                    edges.append({
                        "id": edge_id,
                        "source": src,
                        "target": tgt,
                        "label": rel.get("relation") or "",
                        "description": rel.get("description") or "",
                    })

    return {"nodes": nodes, "edges": edges}
