from langsmith import traceable

from config import get_settings
from llm_service import get_chat_client
from schemas import ChatMessageRecord


def format_messages_for_summary(messages: list[ChatMessageRecord]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


@traceable(name="summarize_chat_memory", run_type="llm")
def summarize_chat_memory(
    old_summary: str,
    messages: list[ChatMessageRecord],
) -> str:
    settings = get_settings()
    client = get_chat_client()
    prompt = f"""你是会话记忆摘要器。请把旧摘要和新增对话合并成新的长期摘要。

要求：
1. 保留用户明确表达的目标、偏好、限制、重要事实。
2. 保留正在讨论的主题和已确认的上下文。
3. 删除寒暄、重复内容和无关细节。
4. 不要编造未出现的信息。
5. 用简洁中文输出。

旧摘要：
{old_summary or "无"}

新增对话：
{format_messages_for_summary(messages)}

新的长期摘要：
"""
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (response.choices[0].message.content or old_summary).strip()


@traceable(name="summarize_session_title", run_type="llm")
def summarize_session_title(messages: list[ChatMessageRecord]) -> str:
    settings = get_settings()
    client = get_chat_client()
    prompt = f"""请根据下面这段对话，生成一个中文会话标题。

要求：
1. 只输出标题，不要解释。
2. 不超过 16 个汉字。
3. 标题要表达用户真正想问的主题。

对话：
{format_messages_for_summary(messages)}

标题："""
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    title = (response.choices[0].message.content or "").strip()
    return title.strip("「」“”\"' \n")[:24]
