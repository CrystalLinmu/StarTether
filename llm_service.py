from collections.abc import Iterator

from fastapi import HTTPException
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import APIStatusError, OpenAI

from config import get_settings
from schemas import RetrievedChunk


def get_client() -> OpenAI:
    """Embedding 客户端（DashScope）。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="请先在 .env 中配置 OPENAI_API_KEY")

    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return wrap_openai(OpenAI(**kwargs))


def get_chat_client() -> OpenAI:
    """对话 / Grade / Rewrite 客户端（DeepSeek）。"""
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=400, detail="请先在 .env 中配置 DEEPSEEK_API_KEY")

    return wrap_openai(OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    ))


@traceable(name="embed_texts", run_type="embedding")
def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = get_client()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]
        try:
            response = client.embeddings.create(model=settings.embedding_model, input=batch)
        except APIStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"向量模型调用失败：{exc.message}",
            ) from exc
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def format_contexts_for_prompt(contexts: list[RetrievedChunk]) -> str:
    formatted_contexts: list[str] = []
    for chunk in contexts:
        dense_score = f"{chunk.dense_score:.4f}" if chunk.dense_score is not None else "无"
        formatted_contexts.append(
            f"来源：{chunk.document_name}\n"
            f"章节：{chunk.section_title}\n"
            f"层级：L{chunk.chunk_level}\n"
            f"相似度：{dense_score}\n"
            f"内容：{chunk.content}"
        )
    return "\n\n---\n\n".join(formatted_contexts)


def build_answer_prompt(
    question: str,
    context_text: str,
    memory_summary: str = "",
    recent_messages: list[dict] | None = None,
) -> str:
    parts: list[str] = []
    parts.append("你是企业内部文档问答助手。只根据给定资料回答问题。")
    parts.append("如果资料中没有答案，直接说“资料中没有找到相关信息”。")

    if memory_summary:
        parts.append(f"\n对话摘要（之前的讨论要点）：\n{memory_summary}")

    if recent_messages:
        history = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in recent_messages
        )
        if history:
            parts.append(f"\n最近的对话：\n{history}")

    parts.append(f"\n资料：\n{context_text}")
    parts.append(f"\n用户当前问题：\n{question}")
    parts.append("\n请结合对话上下文理解用户意图后回答。")

    return "\n".join(parts)


def _build_chat_messages(
    question: str,
    contexts: list[RetrievedChunk],
    memory_summary: str = "",
    recent_messages: list[dict] | None = None,
) -> list[dict]:
    """仿照 SuperMew 构建多角色消息列表，LLM 能更好感知对话结构。"""
    messages: list[dict] = []

    # 系统提示
    system_parts = [
        "你是企业内部文档问答助手。只根据给定资料回答问题。",
        '如果资料中没有答案，直接说“资料中没有找到相关信息”。',
    ]
    if memory_summary:
        system_parts.append(f"\n对话摘要（之前的讨论要点）：\n{memory_summary}")
    messages.append({"role": "system", "content": "\n".join(system_parts)})

    # 历史对话
    if recent_messages:
        for m in recent_messages:
            messages.append({"role": m["role"], "content": m["content"]})

    # 当前问题（带检索资料）
    context_text = format_contexts_for_prompt(contexts)
    current_prompt = f"资料：\n{context_text}\n\n用户当前问题：\n{question}\n\n请结合对话上下文理解用户意图后回答。"
    messages.append({"role": "user", "content": current_prompt})

    return messages


@traceable(name="answer_question", run_type="llm")
def answer_question(
    question: str,
    contexts: list[RetrievedChunk],
    memory_summary: str = "",
    recent_messages: list[dict] | None = None,
) -> str:
    settings = get_settings()
    if not settings.chat_model:
        raise HTTPException(status_code=400, detail="请先在 .env 中配置 CHAT_MODEL")

    client = get_chat_client()
    messages = _build_chat_messages(question, contexts, memory_summary, recent_messages)

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def stream_answer_question(
    question: str,
    contexts: list[RetrievedChunk],
    memory_summary: str = "",
    recent_messages: list[dict] | None = None,
) -> Iterator[str]:
    settings = get_settings()
    if not settings.chat_model:
        raise HTTPException(status_code=400, detail="请先在 .env 中配置 CHAT_MODEL")

    client = get_chat_client()
    messages = _build_chat_messages(question, contexts, memory_summary, recent_messages)

    stream = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta
