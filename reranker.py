import httpx
from langsmith import traceable

from config import get_settings
from schemas import RetrievedChunk


@traceable(name="rerank_chunks", run_type="retriever")
def rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    settings = get_settings()
    if not settings.dashscope_rerank_api_key or not chunks:
        return chunks

    payload = {
        "model": settings.dashscope_rerank_model,
        "query": question,
        "documents": [chunk.content for chunk in chunks],
        "top_n": min(settings.top_k, len(chunks)),
        "instruct": settings.dashscope_rerank_instruct,
    }
    headers = {
        "Authorization": f"Bearer {settings.dashscope_rerank_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            settings.dashscope_rerank_url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return chunks[: settings.top_k]

    results = response.json().get("output", {}).get("results", [])
    reranked_chunks: list[RetrievedChunk] = []
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or index >= len(chunks):
            continue

        chunk = chunks[index].model_copy()
        if score is not None:
            chunk.rerank_score = float(score)
        reranked_chunks.append(chunk)

    return reranked_chunks or chunks[: settings.top_k]
