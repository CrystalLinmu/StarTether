from __future__ import annotations

from dataclasses import dataclass

from pymilvus import DataType, Function, FunctionType, MilvusClient

from config import get_settings
from tokenizer import tokenize_for_search


@dataclass
class MilvusSearchHit:
    chunk_uid: str
    score: float


def get_milvus_client() -> MilvusClient:
    settings = get_settings()
    token = settings.milvus_token or None
    return MilvusClient(uri=settings.milvus_uri, token=token)


def ensure_milvus_collection() -> None:
    settings = get_settings()
    client = get_milvus_client()
    collection = settings.milvus_collection
    if client.has_collection(collection):
        client.load_collection(collection)
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_uid", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("parent_uid", DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field("chunk_level", DataType.INT64)
    schema.add_field("document_name", DataType.VARCHAR, max_length=512)
    schema.add_field("section_title", DataType.VARCHAR, max_length=1024)
    schema.add_field("position_hint", DataType.VARCHAR, max_length=256)
    schema.add_field("content", DataType.VARCHAR, max_length=settings.milvus_text_max_length)
    schema.add_field(
        "tokenized_content",
        DataType.VARCHAR,
        max_length=settings.milvus_text_max_length,
        enable_analyzer=True,
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="content_bm25",
            input_field_names=["tokenized_content"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection)


def upsert_leaf_chunks(records: list[dict]) -> None:
    if not records:
        return
    settings = get_settings()
    ensure_milvus_collection()
    client = get_milvus_client()
    client.upsert(collection_name=settings.milvus_collection, data=records)
    client.flush(collection_name=settings.milvus_collection)


def delete_document_vectors(document_name: str) -> None:
    settings = get_settings()
    client = get_milvus_client()
    if not client.has_collection(settings.milvus_collection):
        return
    client.delete(
        collection_name=settings.milvus_collection,
        filter=f'document_name == "{escape_filter_string(document_name)}"',
    )
    client.flush(collection_name=settings.milvus_collection)


def dense_search(query_embedding: list[float], limit: int) -> list[MilvusSearchHit]:
    settings = get_settings()
    ensure_milvus_collection()
    client = get_milvus_client()
    results = client.search(
        collection_name=settings.milvus_collection,
        data=[query_embedding],
        anns_field="dense",
        limit=limit,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=["chunk_uid"],
    )
    return parse_hits(results)


def sparse_search(query: str, limit: int) -> list[MilvusSearchHit]:
    settings = get_settings()
    ensure_milvus_collection()
    client = get_milvus_client()
    tokenized_query = tokenize_for_search(query)
    results = client.search(
        collection_name=settings.milvus_collection,
        data=[tokenized_query],
        anns_field="sparse",
        limit=limit,
        search_params={"metric_type": "BM25", "params": {}},
        output_fields=["chunk_uid"],
    )
    return parse_hits(results)


def parse_hits(results) -> list[MilvusSearchHit]:
    hits: list[MilvusSearchHit] = []
    if not results:
        return hits

    for item in results[0]:
        entity = item.get("entity", {}) if isinstance(item, dict) else {}
        chunk_uid = entity.get("chunk_uid") or item.get("id")
        score = item.get("distance", item.get("score", 0.0))
        if chunk_uid:
            hits.append(MilvusSearchHit(chunk_uid=str(chunk_uid), score=float(score)))
    return hits


def escape_filter_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
