import json
import re

from langsmith import traceable

from config import get_settings
from llm_service import get_chat_client
from schemas import GradeResult, RetrievedChunk


def _contexts_for_grade(contexts: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"资料 {index + 1}：\n{chunk.content}"
        for index, chunk in enumerate(contexts[:5])
    )


def _parse_grade_response(content: str) -> GradeResult:
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        return GradeResult(can_answer=False, reason="评分模型没有返回有效 JSON。")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return GradeResult(can_answer=False, reason="评分模型返回的 JSON 无法解析。")

    return GradeResult(
        can_answer=bool(data.get("can_answer")),
        reason=str(data.get("reason") or ""),
        missing_information=str(data.get("missing_information") or ""),
    )


@traceable(name="grade_contexts", run_type="llm")
def grade_contexts(question: str, contexts: list[RetrievedChunk]) -> GradeResult:
    if not contexts:
        return GradeResult(can_answer=False, reason="没有检索到可用资料。")

    settings = get_settings()
    client = get_chat_client()
    prompt = f"""你是 RAG 检索结果质检员。请判断给定资料是否足以回答用户问题。

判断标准：
1. 资料中包含了回答问题所需的关键事实或数字，即判定为 can_answer=true。
2. 只要资料中有任何一条包含答案的线索，就应判定为可回答。
3. 不要因为资料没有以用户期望的方式呈现就判定为不足。
4. 只返回 JSON，不要输出多余文字。

JSON 格式：
{{
  "can_answer": true,
  "reason": "资料为什么足够或不足（20字以内）",
  "missing_information": "如果不能回答，缺少什么信息"
}}

用户问题：
{question}

资料：
{_contexts_for_grade(contexts)}
"""
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _parse_grade_response(response.choices[0].message.content or "")
