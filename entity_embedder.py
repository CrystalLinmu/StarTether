"""实体 / 问题 embedding 计算。

模型: paraphrase-multilingual-MiniLM-L12-v2（118MB）
中英文通用，384维。
"""

from __future__ import annotations

import logging

from langsmith import traceable
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


@traceable(name="embed_entities", run_type="embedding")
def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量计算文本 embedding，返回 384 维向量列表。"""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """单条文本 embedding。"""
    return embed_texts([text])[0]


def similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """归一化向量的余弦相似度（等价于点积）。"""
    return sum(a * b for a, b in zip(vec_a, vec_b))
