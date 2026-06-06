from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    debug: bool = False
    folder_id: str | None = None
    enable_graph: bool = True


class RetrievedChunk(BaseModel):
    id: int
    chunk_uid: str
    parent_uid: str | None = None
    chunk_level: int = 3
    document_name: str
    section_title: str
    position_hint: str = ""
    content: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    merge_ratio: float | None = None
    merged_from_children: list[str] = Field(default_factory=list)
    merge_info: dict | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    contexts: list[RetrievedChunk]
    retrieval_query: str
    contextualized_question: str | None = None
    used_contextual_rewrite: bool = False
    rewrite_strategy: str = "none"
    rewritten_query: str | None = None
    rewrite_attempts: list[str] = Field(default_factory=list)
    can_answer: bool = True
    grade_reason: str = ""
    missing_information: str = ""
    memory_summary: str = ""
    debug_info: dict | None = None


class SessionListItem(BaseModel):
    session_id: str
    title: str
    message_count: int
    updated_at: str


class SessionDetail(BaseModel):
    session_id: str
    summary: str = ""
    messages: list[ChatMessageRecord] = Field(default_factory=list)


class DocumentListItem(BaseModel):
    document_name: str
    chunk_count: int
    leaf_chunk_count: int
    created_at: str


class DocumentChunkItem(BaseModel):
    id: int
    chunk_uid: str
    parent_uid: str | None = None
    chunk_level: int
    document_name: str
    section_title: str
    position_hint: str = ""
    chunk_index: int
    content: str
    has_embedding: bool
    has_search_vector: bool


class DeleteDocumentResponse(BaseModel):
    document_name: str
    deleted_chunks: int


class RetrievalResult(BaseModel):
    contexts: list[RetrievedChunk]
    retrieval_query: str
    contextualized_question: str | None = None
    used_contextual_rewrite: bool = False
    rewrite_strategy: str = "none"
    rewritten_query: str | None = None
    rewrite_attempts: list[str] = Field(default_factory=list)
    can_answer: bool = True
    grade_reason: str = ""
    missing_information: str = ""
    debug_info: dict | None = None


class GradeResult(BaseModel):
    can_answer: bool
    reason: str = ""
    missing_information: str = ""


class ChatMessageRecord(BaseModel):
    id: int | None = None
    role: str
    content: str
    rag_trace: dict | None = None


class MemoryContext(BaseModel):
    summary: str = ""
    recent_messages: list[ChatMessageRecord] = Field(default_factory=list)


class IngestResponse(BaseModel):
    filename: str
    chunks: int
    folder_id: str = ""
    entities_extracted: int = 0


# ---- Folder models ----

class FolderCreate(BaseModel):
    folder_name: str = Field(min_length=1, max_length=100)
    parent_folder_id: str | None = None


class FolderUpdate(BaseModel):
    folder_name: str = Field(min_length=1, max_length=100)


class FolderItem(BaseModel):
    folder_id: str
    folder_name: str
    parent_folder_id: str | None = None
    document_count: int = 0
    entity_count: int = 0
    created_at: str
    children: list["FolderItem"] = Field(default_factory=list)


class FolderDetail(FolderItem):
    documents: list[DocumentListItem] = Field(default_factory=list)


class DocumentMove(BaseModel):
    target_folder_id: str


# ---- Graph / Neo4j models ----

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    group: str = ""
    description: str = ""
    confidence: float = 0.0
    chunk_count: int = 0


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    description: str = ""


class SubgraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
