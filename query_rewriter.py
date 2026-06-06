from langsmith import traceable

from config import get_settings
from llm_service import get_chat_client
from schemas import ChatMessageRecord, MemoryContext, RetrievedChunk
from tokenizer import tokenize_for_search


CONTEXT_SIGNALS = (
    "他",
    "她",
    "这个",
    "那个",
    "它",
    "这项",
    "那项",
    "这种",
    "上面",
    "前面",
    "刚才",
    "第一条",
    "第二条",
    "继续",
    "展开",
    "详细点",
    "怎么办",
    "怎么处理",
)

STEP_BACK_SIGNALS = (
    "这个",
    "那个",
    "它",
    "这块",
    "那块",
    "这种",
    "上面",
    "刚才",
    "怎么办",
    "怎么处理",
)


def best_retrieval_score(chunks: list[RetrievedChunk]) -> float:
    best = 0.0
    for chunk in chunks:
        score = chunk.rerank_score
        if score is None:
            score = chunk.rrf_score
        if score is None:
            score = chunk.dense_score
        if score is not None:
            best = max(best, score)
    return best


def should_rewrite(chunks: list[RetrievedChunk]) -> bool:
    settings = get_settings()
    if not settings.query_rewrite_enabled:
        return False
    if len(chunks) < settings.query_rewrite_min_contexts:
        return True
    return best_retrieval_score(chunks) < settings.query_rewrite_min_best_score


def explain_should_rewrite(chunks: list[RetrievedChunk]) -> str:
    settings = get_settings()
    if not settings.query_rewrite_enabled:
        return "查询重写已关闭"
    if len(chunks) < settings.query_rewrite_min_contexts:
        return f"召回块数 {len(chunks)} 小于阈值 {settings.query_rewrite_min_contexts}"
    best_score = best_retrieval_score(chunks)
    if best_score < settings.query_rewrite_min_best_score:
        return f"最佳检索分数 {best_score:.4f} 小于阈值 {settings.query_rewrite_min_best_score:.4f}"
    return "召回数量和分数达到阈值，不需要重写"


def has_clear_topic(question: str) -> bool:
    tokenized = tokenize_for_search(question)
    return any(len(token) >= 3 for token in tokenized.split())


def needs_contextual_rewrite(question: str, memory: MemoryContext) -> bool:
    if not memory.summary and not memory.recent_messages:
        return False

    compact_question = question.strip()
    if any(signal in compact_question for signal in CONTEXT_SIGNALS):
        return True
    return len(compact_question) <= 8 and not has_clear_topic(compact_question)


def explain_contextual_rewrite(question: str, memory: MemoryContext) -> str:
    if not memory.summary and not memory.recent_messages:
        return "没有历史摘要和最近消息，不需要上下文改写"

    compact_question = question.strip()
    matched = [signal for signal in CONTEXT_SIGNALS if signal in compact_question]
    if matched:
        return f"问题包含上下文指代词：{', '.join(matched)}"
    if len(compact_question) <= 8 and not has_clear_topic(compact_question):
        return "问题较短且主题不明确，需要结合历史对话补全"
    return "问题本身较完整，不需要上下文改写"


def format_history(history: list[ChatMessageRecord]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in history)


@traceable(name="contextualize_question", run_type="llm")
def contextualize_question(question: str, memory: MemoryContext) -> str:
    if not needs_contextual_rewrite(question, memory):
        return question

    settings = get_settings()
    client = get_chat_client()
    prompt = f"""请根据长期摘要和最近对话，把当前问题改写成一个脱离上下文也能理解的独立检索问题。
要求：
1. 只输出改写后的问题。
2. 不要回答问题。
3. 如果当前问题已经完整，就原样输出。

长期摘要：
{memory.summary or "无"}

最近对话：
{format_history(memory.recent_messages) or "无"}

当前问题：
{question}
"""
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return (response.choices[0].message.content or question).strip()


def choose_rewrite_order(question: str) -> list[str]:
    compact_question = question.strip()
    tokenized = tokenize_for_search(compact_question)
    token_count = len(tokenized.split())

    if any(signal in compact_question for signal in STEP_BACK_SIGNALS):
        return ["step_back", "hyde"]
    if len(compact_question) <= 8 or token_count <= 2:
        return ["hyde", "step_back"]
    return ["step_back", "hyde"]


def explain_rewrite_order(question: str) -> str:
    compact_question = question.strip()
    tokenized = tokenize_for_search(compact_question)
    token_count = len(tokenized.split())
    matched = [signal for signal in STEP_BACK_SIGNALS if signal in compact_question]

    if matched:
        return f"问题包含指代或处理类表达：{', '.join(matched)}，优先 Step-Back"
    if len(compact_question) <= 8 or token_count <= 2:
        return "问题较短或关键词较少，优先 HyDE 扩展语义"
    return "问题较完整，优先 Step-Back 扩大检索范围"


@traceable(name="rewrite_step_back_query", run_type="llm")
def rewrite_step_back_query(question: str) -> str:
    settings = get_settings()
    client = get_chat_client()
    prompt = f"""请把下面的用户问题改写成一个更通用、更完整、更适合知识库检索的问题。
要求：
1. 只输出改写后的问题。
2. 不要回答问题。
3. 不要引入具体事实，只补全检索意图。

用户问题：
{question}
"""
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (response.choices[0].message.content or question).strip()


@traceable(name="generate_hyde_query", run_type="llm")
def generate_hyde_query(question: str) -> str:
    settings = get_settings()
    client = get_chat_client()
    prompt = f"""请为下面的用户问题生成一段“假设性的文档片段”，用于知识库检索。
要求：
1. 只输出一段检索用文本。
2. 不要说“根据资料”。
3. 不要编造具体公司数字、金额、日期。
4. 用可能出现在企业制度文档中的表达方式描述相关主题、政策、流程和关键词。

用户问题：
{question}
"""
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (response.choices[0].message.content or question).strip()


def rewrite_query(question: str, strategy: str) -> str:
    if strategy == "hyde":
        return generate_hyde_query(question)
    if strategy == "step_back":
        return rewrite_step_back_query(question)
    return question
