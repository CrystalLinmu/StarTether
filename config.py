from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""

    embedding_model: str = "text-embedding-v4"
    chat_model: str = "deepseek-chat"

    # DeepSeek 对话模型（chat / grade / rewrite 共用）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    embedding_dim: int = 1024
    embedding_batch_size: int = 10
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_chunks_hybrid"
    milvus_text_max_length: int = 4096

    top_k: int = 3
    candidate_top_k: int = 12
    l1_chunk_max_chars: int = 2400
    l2_chunk_max_chars: int = 1600
    l3_chunk_max_chars: int = 500
    l3_chunk_overlap_chars: int = 60
    semantic_merge_similarity: float = 0.62
    auto_merge_min_children: int = 2
    auto_merge_child_ratio: float = 0.6
    auto_merge_max_parent_chars: int = 1800
    max_chunks_per_doc: int = 500

    query_rewrite_enabled: bool = True
    query_rewrite_min_contexts: int = 2
    query_rewrite_min_best_score: float = 0.015
    history_message_limit: int = 20
    summary_trigger_messages: int = 40
    summary_keep_recent_messages: int = 10

    dashscope_rerank_api_key: str = ""
    dashscope_rerank_model: str = "qwen3-rerank"
    dashscope_rerank_url: str = "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks"
    dashscope_rerank_instruct: str = "Given a web search query, retrieve relevant passages that answer the query."

    # GraphRAG
    graph_retrieval_enabled: bool = True
    graph_max_hops: int = 2
    graph_top_k_entities: int = 10
    graph_top_k_chunks: int = 5
    entity_extraction_batch_size: int = 8
    entity_extraction_confidence_threshold: float = 0.6
    entity_similarity_threshold: float = 0.82  # embedding 余弦 >0.82 → 同一实体
    graph_community_min_size: int = 3
    community_levels: int = 3  # Leiden 层级数（0=最粗, 2=最细）

    # Neo4j（可视化用，可选）
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-rag-dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---- 运行时覆盖（前端可修改，重启后恢复 .env 默认值） ----

_runtime_overrides: dict[str, object] = {}

EDITABLE_KEYS = {
    "top_k", "candidate_top_k",
    "auto_merge_min_children", "auto_merge_child_ratio", "auto_merge_max_parent_chars",
    "graph_retrieval_enabled", "graph_max_hops", "graph_top_k_chunks", "graph_top_k_entities",
    "entity_extraction_confidence_threshold", "entity_similarity_threshold",
}

EXPORTED_PARAMS = [
    # 分块
    ("l1_chunk_max_chars", "L1 大块", "按章节/段落切分的最大父块"),
    ("l2_chunk_max_chars", "L2 中块", "语义合并后的中间块，实体提取用"),
    ("l3_chunk_max_chars", "L3 小块", "最小检索粒度，送入 Milvus"),
    ("l3_chunk_overlap_chars", "重叠字符", "L3 块间重叠"),
    ("max_chunks_per_doc", "单文档上限", "超过截断"),
    # 检索
    ("embedding_dim", "向量维度", "text-embedding-v4 输出维度"),
    ("embedding_batch_size", "Embed 批量", "每批送 DashScope 的文本数"),
    ("top_k", "最终块数", "最终送给 LLM 的 chunk 数量"),
    ("candidate_top_k", "候选数", "Dense/BM25 各取候选上限"),
    # 合并
    ("semantic_merge_similarity", "合并相似度", "相邻 L3 余弦相似度阈值"),
    ("auto_merge_min_children", "最少子块", "触发父块替换的最少命中子块"),
    ("auto_merge_child_ratio", "命中比例", "命中子块/总子块触发阈值"),
    ("auto_merge_max_parent_chars", "父块上限", "超过此长度的父块不合并"),
    # GraphRAG 实体
    ("graph_retrieval_enabled", "总开关", "关闭后纯向量检索"),
    ("entity_extraction_batch_size", "提取批量", "每批 L2 chunk 送 LLM 数"),
    ("entity_extraction_confidence_threshold", "置信度阈值", "低于此值的实体丢弃"),
    ("entity_similarity_threshold", "去重相似度", "余弦 > 此值判定为同一实体"),
    # GraphRAG 检索
    ("graph_max_hops", "最大跳数", "图遍历邻居层级"),
    ("graph_top_k_chunks", "图谱块数", "图遍历后取 chunk 数"),
    ("graph_top_k_entities", "实体候选数", "匹配阶段最多取图中实体数"),
    # 社区
    ("community_levels", "层级数", "Leiden 检测层数"),
    ("graph_community_min_size", "最小社区", "少于 N 个实体的社区合并"),
]

_IMMEDIATE_KEYS = {
    "top_k", "candidate_top_k",
    "auto_merge_min_children", "auto_merge_child_ratio", "auto_merge_max_parent_chars",
    "graph_retrieval_enabled", "graph_max_hops", "graph_top_k_chunks", "graph_top_k_entities",
}


def get_runtime(key: str, default=None):
    """优先返回运行时覆盖值，否则从 Settings 取。"""
    settings = get_settings()
    if key in _runtime_overrides:
        # 类型转换
        default_val = getattr(settings, key, None)
        val = _runtime_overrides[key]
        if default_val is not None and not isinstance(val, type(default_val)):
            try:
                return type(default_val)(val)
            except (ValueError, TypeError):
                return default_val
        return val
    return getattr(settings, key, default)


def set_runtime(updates: dict) -> dict:
    """批量设置运行时参数，仅允许 EDITABLE_KEYS 中的键。返回生效的键值。"""
    applied = {}
    for key, val in updates.items():
        if key in EDITABLE_KEYS:
            _runtime_overrides[key] = val
            applied[key] = val
    return applied


def is_immediate(key: str) -> bool:
    return key in _IMMEDIATE_KEYS


def export_settings() -> list[dict]:
    """导出所有可展示参数列表。"""
    settings = get_settings()
    result = []
    for key, label, help_text in EXPORTED_PARAMS:
        val = get_runtime(key)
        if isinstance(val, bool):
            display = str(val).lower()
        elif isinstance(val, float):
            display = f"{val:.2f}"
        else:
            display = str(val)
        result.append({
            "key": key, "label": label, "value": display, "help": help_text,
            "editable": key in EDITABLE_KEYS,
            "immediate": is_immediate(key),
        })
    return result
