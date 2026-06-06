export type SessionItem = {
  session_id: string;
  title: string;
  message_count: number;
  updated_at: string;
};

export type ChatMessage = {
  id?: number;
  role: "user" | "assistant";
  content: string;
  rag_trace?: RagTrace | null;
};

export type SessionDetail = {
  session_id: string;
  summary: string;
  messages: ChatMessage[];
};

export type DocumentItem = {
  document_name: string;
  chunk_count: number;
  leaf_chunk_count: number;
  created_at: string;
};

export type RetrievedChunk = {
  id: number;
  chunk_uid: string;
  parent_uid?: string | null;
  chunk_level: number;
  document_name: string;
  section_title: string;
  position_hint: string;
  content: string;
  dense_score?: number | null;
  sparse_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  merge_ratio?: number | null;
  merged_from_children?: string[];
  merge_info?: Record<string, unknown> | null;
};

export type RagRun = {
  strategy: string;
  query: string;
  dense_results?: any[];
  keyword_results?: any[];
  rrf_results?: any[];
  rerank_results?: any[];
  auto_merged_results?: any[];
  final_results?: any[];
  timings?: Record<string, number>;
  summary?: Record<string, number>;
  grade?: {
    can_answer: boolean;
    reason: string;
    missing_information?: string;
  };
  rewrite_trigger_reason?: string;
};

export type RagTrace = {
  original_question?: string;
  contextualized_question?: string;
  used_contextual_rewrite?: boolean;
  contextual_rewrite_reason?: string;
  selected_strategy?: string;
  selected_query?: string;
  rewrite_order_reason?: string;
  rewrite_attempts?: string[];
  runs?: RagRun[];
};

export type StreamStep = {
  code: string;
  label: string;
  count?: number;
  elapsed_ms?: number;
  merged_count?: number;
  can_answer?: boolean;
  reason?: string;
  strategy?: string;
  query?: string;
};

export type ChatDone = {
  type: "done";
  session_id: string;
  answer: string;
  contexts: RetrievedChunk[];
  retrieval_query: string;
  contextualized_question?: string | null;
  used_contextual_rewrite: boolean;
  rewrite_strategy: string;
  rewritten_query?: string | null;
  can_answer: boolean;
  grade_reason: string;
  debug_info?: RagTrace | null;
  graph_data?: GraphData | null;
};

// ---- Folder types ----

export type FolderItem = {
  folder_id: string;
  folder_name: string;
  parent_folder_id: string | null;
  document_count: number;
  entity_count: number;
  created_at: string;
  children: FolderItem[];
};

// ---- Graph types ----

export type GraphNode = {
  id: string;
  label: string;
  type: string;
  group: string;
  description: string;
  confidence: number;
  chunk_count: number;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  description: string;
};

export type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type SubgraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

const API_BASE = "";

async function readJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  const data = await res.json();
  return normalizeText(data) as T;
}

export function listSessions() {
  return readJson<SessionItem[]>("/sessions");
}

export function getSession(sessionId: string) {
  return readJson<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function deleteSession(sessionId: string) {
  return readJson<{ session_id: string; deleted: boolean }>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
  );
}

export function renameSession(sessionId: string, title: string) {
  return readJson<{ session_id: string; title: string }>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    },
  );
}

export function listDocuments() {
  return readJson<DocumentItem[]>("/documents");
}

export function deleteDocument(documentName: string) {
  return readJson<{ document_name: string; deleted_chunks: number }>(
    `/documents/${encodeURIComponent(documentName)}`,
    { method: "DELETE" },
  );
}

export type UploadDone = {
  filename: string;
  chunks: number;
  folder_id?: string;
  stats?: Record<string, unknown>;
};

export async function uploadDocument(
  file: File,
  onStep: (step: StreamStep) => void,
) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/ingest/stream", { method: "POST", body: form });
  if (!res.ok || !res.body) {
    throw new Error(await res.text());
  }
  let donePayload: UploadDone | null = null;
  await readSse(res, (event) => {
    if (event.type === "step") onStep(event as StreamStep);
    if (event.type === "done") donePayload = event as UploadDone;
    if (event.type === "error") throw new Error(event.message || "上传失败");
  });
  return donePayload;
}

export async function streamChat(
  question: string,
  sessionId: string | null,
  onEvent: (event: Record<string, any>) => void,
  folderId?: string,
  enableGraph?: boolean,
) {
  const res = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      debug: true,
      folder_id: folderId || null,
      enable_graph: enableGraph ?? true,
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(await res.text());
  }
  await readSse(res, onEvent);
}

// ---- Folder APIs ----

export function listFolders() {
  return readJson<FolderItem[]>("/folders");
}

export function listFolderTree() {
  return readJson<FolderItem[]>("/folders/tree");
}

export function createFolder(name: string, parentId?: string) {
  return readJson<FolderItem>(
    "/folders",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_name: name, parent_folder_id: parentId || null }),
    },
  );
}

export function updateFolder(folderId: string, name: string) {
  return readJson<FolderItem>(
    `/folders/${encodeURIComponent(folderId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_name: name }),
    },
  );
}

export function deleteFolder(folderId: string) {
  return readJson<{ deleted: boolean; folder_id: string }>(
    `/folders/${encodeURIComponent(folderId)}`,
    { method: "DELETE" },
  );
}

export function listDocumentsInFolder(folderId: string) {
  return readJson<DocumentItem[]>(`/folders/${encodeURIComponent(folderId)}/documents`);
}

export async function uploadDocumentToFolder(
  folderId: string,
  file: File,
  onStep: (step: StreamStep) => void,
) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/folders/${encodeURIComponent(folderId)}/ingest/stream`, {
    method: "POST",
    body: form,
  });
  if (!res.ok || !res.body) throw new Error(await res.text());
  let donePayload: UploadDone | null = null;
  await readSse(res, (event) => {
    if (event.type === "step") onStep(event as StreamStep);
    if (event.type === "done") donePayload = event as UploadDone;
    if (event.type === "error") throw new Error(event.message || "上传失败");
  });
  return donePayload;
}

// ---- Graph API ----

export function getFolderGraph(folderId: string, entityNames?: string[]) {
  const params = entityNames?.length ? `?entity_names=${entityNames.join(",")}` : "";
  return readJson<SubgraphData>(`/folders/${encodeURIComponent(folderId)}/graph${params}`);
}

async function readSse(res: Response, onEvent: (event: Record<string, any>) => void) {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      onEvent(normalizeText(JSON.parse(line.slice(6))) as Record<string, any>);
    }
  }
}

function normalizeText(value: unknown): unknown {
  if (typeof value === "string") return fixMojibake(value);
  if (Array.isArray(value)) return value.map(normalizeText);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, normalizeText(item)]),
    );
  }
  return value;
}

function fixMojibake(text: string): string {
  let hasChinese = false, hasHighByte = false;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0x4E00 && c <= 0x9FFF) { hasChinese = true; break; }
    if (c >= 0x80 && c <= 0xFF) hasHighByte = true;
  }
  if (hasChinese || !hasHighByte) return text;
  try {
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i) & 0xFF;
    const decoded = new TextDecoder("utf-8").decode(bytes);
    for (let i = 0; i < decoded.length; i++) {
      if (decoded.charCodeAt(i) >= 0x4E00 && decoded.charCodeAt(i) <= 0x9FFF) return decoded;
    }
  } catch { /* ignore */ }
  return text;
}

function readableScore(text: string): number {
  let chinese = 0, bad = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0x4E00 && c <= 0x9FFF) chinese++;
    else if (c >= 0x80 && c <= 0xFF) bad++;
  }
  return chinese * 2 - bad * 3;
}
