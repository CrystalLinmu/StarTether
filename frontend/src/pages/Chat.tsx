import {
  CheckCircle2, Circle, Combine, Database, FileSearch, GitBranch, Globe,
  Layers, Loader2, ScanSearch, Search, Send, Sparkles, Zap, ChevronDown, Network,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { getSession, streamChat, type ChatMessage, type GraphData, type GraphNode, type RetrievedChunk, type StreamStep } from "../api";
import FolderPicker from "../components/FolderPicker";
import KnowledgeGraph from "../components/KnowledgeGraph";
import { listFolderTree, type FolderItem } from "../api";

type ChatProps = {
  sessionId: string | null;
  folderId: string | null;
  onSessionChange: (s: string) => void;
  onDataChanged: () => void;
};
type LocalMessage = ChatMessage & {
  steps?: StreamStep[];
  contexts?: RetrievedChunk[];
  streaming?: boolean;
  error?: string;
};

const features = [
  { icon: FileSearch, title: "混合检索", desc: "稠密向量 + BM25 双路召回，RRF 融合排序" },
  { icon: Layers, title: "智能分块", desc: "三级层级分块，语义合并，父块自动替换" },
  { icon: Sparkles, title: "查询改写", desc: "上下文补全 · Step-Back · HyDE 多策略重写" },
];

function stepIcon(code: string) {
  if (code.startsWith("dense")) return Database;
  if (code.startsWith("keyword")) return Layers;
  if (code.startsWith("rrf")) return Combine;
  if (code.startsWith("rerank")) return Sparkles;
  if (code.startsWith("auto_merge")) return Layers;
  if (code.startsWith("grade")) return CheckCircle2;
  if (code.startsWith("context")) return Search;
  if (code.startsWith("graph_entity")) return Search;
  if (code.startsWith("graph_match")) return GitBranch;
  if (code.startsWith("graph_traverse")) return Network;
  if (code.startsWith("graph")) return GitBranch;
  if (code.startsWith("global")) return Globe;
  if (code.startsWith("answer")) return Zap;
  return Circle;
}

function stepLabel(code: string): string {
  if (code === "dense_done") return "稠密检索";
  if (code === "keyword_done") return "BM25 检索";
  if (code === "rrf_done") return "RRF 融合";
  if (code === "rerank_done") return "Rerank 重排序";
  if (code === "auto_merge_done") return "父块合并";
  if (code === "grade_done") return "充分性判断";
  if (code === "context_done") return "上下文改写";
  if (code === "graph_start") return "升级图谱检索…";
  if (code === "graph_done") return "图谱检索";
  if (code === "graph_fusion_done") return "结果融合";
  if (code === "global_start") return "全局摘要检索…";
  if (code === "global_done") return "全局摘要";
  if (code === "answer_done") return "答案生成";
  if (code === "rewrite_done") return "查询重写";
  return code;
}

function levelName(strategy: string | undefined): string {
  if (!strategy || strategy === "original") return "L0 · 向量检索";
  if (strategy === "graph_local") return "L2 · GraphRAG 局部搜索";
  if (strategy === "global") return "L3 · GraphRAG 全局搜索";
  if (strategy === "step_back" || strategy === "hyde") return "L1 · 查询重写";
  return `L0 · ${strategy || "向量检索"}`;
}

/** Group steps by retrieval level */
function groupStepsByLevel(steps: StreamStep[]): { level: string; border: string; steps: StreamStep[] }[] {
  const groups: { level: string; border: string; steps: StreamStep[] }[] = [];
  const colors = [
    "border-l-indigo-400",   // L0
    "border-l-amber-400",    // L1
    "border-l-emerald-400",  // L2
    "border-l-rose-400",     // L3
  ];

  let currentLevel = "L0 · 向量检索";
  let currentIdx = 0;
  let groupSteps: StreamStep[] = [];

  for (const s of steps) {
    if (!s.code.endsWith("_done") && s.code !== "graph_start") continue;
    // Detect level transitions
    let newLevel: string | null = null;
    if (s.code === "rewrite_done") newLevel = "L1 · 查询重写";
    if (s.code === "graph_entity_start" || s.code === "graph_entity_done" || s.code === "graph_start" || s.code === "graph_match_start" || s.code === "graph_match_done" || s.code === "graph_traverse_start" || s.code === "graph_traverse_done" || s.code === "graph_done" || s.code === "graph_fusion_done") newLevel = "L2 · GraphRAG 局部搜索";
    if (s.code === "global_start" || s.code === "global_done") newLevel = "L3 · GraphRAG 全局搜索";
    if (s.code === "answer_done") newLevel = "答案生成";

    if (newLevel && newLevel !== currentLevel) {
      if (groupSteps.length > 0) {
        groups.push({ level: currentLevel, border: colors[currentIdx % colors.length], steps: [...groupSteps] });
      }
      currentLevel = newLevel;
      currentIdx++;
      groupSteps = [];
    }

    const label = s.label || stepLabel(s.code);
    groupSteps.push({ ...s, label });
  }

  if (groupSteps.length > 0) {
    groups.push({ level: currentLevel, border: colors[currentIdx % colors.length], steps: [...groupSteps] });
  }

  return groups;
}

/* ── 检索过程（层级分组 + 可折叠） ── */
function RagStepsCollapse({ steps }: { steps: StreamStep[] }) {
  const [open, setOpen] = useState(false);
  const groups = useMemo(() => groupStepsByLevel(steps), [steps]);
  const totalDone = groups.reduce((acc, g) => acc + g.steps.length, 0);
  if (!totalDone) return null;

  return (
    <div className="mt-2 rounded-xl border border-border-subtle bg-white overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left hover:bg-bg-surface transition-colors"
      >
        <span className="font-mono text-[10px] font-semibold text-text-muted uppercase tracking-wider">
          检索过程 · {totalDone} 步 · {groups.length} 层
        </span>
        <ChevronDown size={13} className={`text-text-muted transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="border-t border-border-subtle px-3 py-2 space-y-2 animate-fade-up">
          {groups.map((group, gi) => (
            <div key={gi} className={`border-l-2 ${group.border} pl-3 py-1`}>
              <span className="font-mono text-[10px] font-bold text-text-muted uppercase tracking-wider mb-1 block">
                {group.level}
              </span>
              <div className="space-y-0.5">
                {group.steps.map((s, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-[11px]">
                    {s.code.endsWith("_done") ? (
                      s.code === "grade_done" && (s as any).can_answer === false ? (
                        <Circle size={10} className="text-error shrink-0" />
                      ) : (
                        <CheckCircle2 size={10} className="text-success shrink-0" />
                      )
                    ) : (
                      <span className="w-2.5 flex-shrink-0" />
                    )}
                    <span className="font-body text-text-secondary truncate">{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── 流式步骤 ── */
function RagLiveSteps({ steps, streaming }: { steps: StreamStep[]; streaming: boolean }) {
  const doneSteps = useMemo(() => steps.filter((s) => s.code.endsWith("_done")), [steps]);
  const runningBaseCodes = steps
    .filter((s) => s.code.endsWith("_start"))
    .map((s) => s.code.replace("_start", ""))
    .filter((base) => !steps.some((d) => d.code === base + "_done"));
  const hasRunning = runningBaseCodes.length > 0;
  const lastRunning = [...steps].reverse().find((s) => s.code.endsWith("_start"));
  const hasGraph = steps.some((s) => s.code.startsWith("graph"));

  if (!doneSteps.length && !hasRunning && !streaming) return null;

  return (
    <div className="mt-3 space-y-1 border-t border-border-subtle pt-3">
      {doneSteps.map((s, i) => {
        // Insert separator before graph steps
        const isGraph = s.code.startsWith("graph");
        const prevIsGraph = i > 0 && (doneSteps[i - 1]?.code.startsWith("graph"));
        const showSep = isGraph && !prevIsGraph;
        return (
          <div key={i}>
            {showSep && (
              <div className="flex items-center gap-2 my-1.5">
                <div className="flex-1 border-t border-border-subtle" />
                <span className="font-mono text-[9px] text-text-muted px-1">↑ 升级</span>
                <div className="flex-1 border-t border-border-subtle" />
              </div>
            )}
            <div className="flex items-center gap-2 text-[12px] animate-fade-up">
              <CheckCircle2 size={13} className="text-success shrink-0" />
              <span className="font-body text-text-secondary flex-1">{s.label || stepLabel(s.code)}</span>
              {(s as any).elapsed_ms != null && <span className="font-mono text-[10px] text-text-muted">{(s as any).elapsed_ms}ms</span>}
            </div>
          </div>
        );
      })}
      {hasRunning && lastRunning && (
        <>
          {!hasGraph && lastRunning.code.startsWith("graph") && doneSteps.length > 0 && (
            <div className="flex items-center gap-2 my-1.5">
              <div className="flex-1 border-t border-border-subtle" />
              <span className="font-mono text-[9px] text-text-muted px-1">↑ 升级</span>
              <div className="flex-1 border-t border-border-subtle" />
            </div>
          )}
          <div className="flex items-center gap-2 text-[12px] animate-fade-up">
            <Loader2 size={13} className="animate-spin text-accent shrink-0" />
            <span className="font-body text-accent font-medium">{lastRunning.label}</span>
          </div>
        </>
      )}
      {streaming && !hasRunning && doneSteps.length > 0 && (
        <div className="flex items-center gap-2 text-[12px]">
          <Loader2 size={13} className="animate-spin text-accent shrink-0" />
          <span className="font-body text-text-muted">生成回答中…</span>
        </div>
      )}
    </div>
  );
}

function scoreOf(c: RetrievedChunk) {
  const s = c.rerank_score ?? c.dense_score ?? c.rrf_score ?? c.sparse_score;
  return typeof s === "number" ? s.toFixed(3) : "—";
}

const TYPE_COLORS_BY_TYPE: Record<string, string> = {
  PERSON: "#6366F1", ORGANIZATION: "#10B981", LOCATION: "#F59E0B",
  CONCEPT: "#8B5CF6", TIME: "#06B6D4", EVENT: "#EF4444",
  LAW_REGULATION: "#F97316", TECHNOLOGY: "#3B82F6", PRODUCT: "#A855F7",
  OTHER: "#6B7280",
};

/* ── Unified Right Panel ── */
function RightPanelBody({
  selectedMsg, selectedTrace, selectedRun, finalContexts, graphData, messages, selectedTraceIdx,
}: {
  selectedMsg: LocalMessage | null;
  selectedTrace: any;
  selectedRun: any;
  finalContexts: RetrievedChunk[];
  graphData: GraphData | null;
  messages: LocalMessage[];
  selectedTraceIdx: number | null;
}) {
  const [card, setCard] = useState("fusion");
  const [showGraph, setShowGraph] = useState(true);  // 默认展开
  const [showNeighbors, setShowNeighbors] = useState(true);
  const summary = selectedRun?.summary || {};

  if (!selectedMsg) {
    return (
      <div className="flex flex-col items-center justify-center flex-1 gap-2 text-text-muted py-16 px-4">
        <ScanSearch size={32} className="opacity-20" />
        <p className="font-body text-[13px]">点击一条回答查看检索追踪</p>
      </div>
    );
  }

  const round = selectedTraceIdx != null
    ? [...messages].filter((_, i) => i <= selectedTraceIdx && messages[i].role === "user").length
    : "?";

  type CardDef = { key: string; label: string; value: string | number; color: string; icon: React.ElementType };
  const cards: CardDef[] = [
    { key: "dense", label: "稠密", value: summary.dense_count ?? "—", color: "text-indigo-500", icon: Database },
    { key: "keyword", label: "BM25", value: summary.keyword_count ?? "—", color: "text-amber-500", icon: Layers },
    { key: "rrf", label: "RRF", value: summary.rrf_count ?? "—", color: "text-violet-500", icon: Combine },
    { key: "rerank", label: "精排", value: summary.rerank_count ?? "—", color: "text-rose-500", icon: Sparkles },
    { key: "merge", label: "合并", value: summary.merged_parent_count ?? "—", color: "text-emerald-500", icon: Layers },
  ];

  // GraphRAG 数据
  const graphChunksTotal = (graphData as any)?.graph_chunks_found ?? 0;
  const graphOverlap = (graphData as any)?.overlap_with_vector ?? 0;
  const graphOnly = graphChunksTotal - graphOverlap;
  const matchedEntities = (graphData as any)?.entities_matched || [];
  const subNodes = graphData?.nodes || [];
  const matchedNames = new Set(matchedEntities.map((e: any) => e.name || e.label));
  const neighborEntities = subNodes.filter((n: any) => !matchedNames.has(n.label || n.id));

  // 仅在 GraphRAG 模式下显示图谱卡片
  if (graphChunksTotal > 0) {
    cards.push({ key: "graph", label: "图谱", value: `${graphOverlap}+${graphOnly}`, color: "text-cyan-500", icon: GitBranch });
  }
  cards.push({ key: "fusion", label: "融合", value: finalContexts.length, color: "text-accent", icon: Zap });

  // Detail content per card
  function renderDetail() {
    // Which result list to show
    let chunks: RetrievedChunk[] | null = null;
    switch (card) {
      case "dense": chunks = selectedRun?.dense_results || null; break;
      case "keyword": chunks = selectedRun?.keyword_results || null; break;
      case "rrf": chunks = selectedRun?.rrf_results || null; break;
      case "rerank": chunks = selectedRun?.rerank_results || null; break;
      case "merge": chunks = selectedRun?.auto_merged_results || null; break;
      case "fusion": chunks = selectedRun?.final_results || finalContexts; break;
      case "graph": chunks = null; break;
      case "entity": chunks = null; break;
      default: chunks = selectedRun?.final_results || finalContexts;
    }

    // Graph card: show graph retrieval stats
    if (card === "graph") {
      return (
        <div className="space-y-2">
          <div className="grid grid-cols-3 gap-1.5 text-center">
            <div className="rounded-lg bg-indigo-50 p-2"><div className="font-display text-[15px] font-bold text-indigo-500">{graphChunksTotal}</div><div className="font-mono text-[9px] text-text-muted">图谱召回</div></div>
            <div className="rounded-lg bg-emerald-50 p-2"><div className="font-display text-[15px] font-bold text-emerald-500">{graphOverlap}</div><div className="font-mono text-[9px] text-text-muted">向量重叠</div></div>
            <div className="rounded-lg bg-cyan-50 p-2"><div className="font-display text-[15px] font-bold text-cyan-500">{graphOnly}</div><div className="font-mono text-[9px] text-text-muted">图谱独有</div></div>
          </div>
          {matchedEntities.length > 0 && (
            <div>
              <p className="font-mono text-[10px] text-text-muted mb-1">匹配实体 ({matchedEntities.length})</p>
              <div className="flex flex-wrap gap-1">
                {matchedEntities.map((e: any, i: number) => (
                  <span key={i} className="inline-flex items-center gap-1 rounded-md bg-indigo-50 px-2 py-0.5 font-mono text-[11px] text-indigo-600">{e.name || e.label}</span>
                ))}
              </div>
            </div>
          )}
          {neighborEntities.length > 0 && (
            <div>
              <p className="font-mono text-[10px] text-text-muted mb-1">邻居实体 ({neighborEntities.length})</p>
              <div className="flex flex-wrap gap-1">
                {neighborEntities.slice(0, 15).map((n: any, i: number) => (
                  <span key={n.id || i} className="inline-flex items-center gap-1 rounded-md border border-border-subtle bg-white px-2 py-0.5 font-mono text-[10px]">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: TYPE_COLORS_BY_TYPE[n.type] || "#6B7280" }} />
                    {n.label}
                  </span>
                ))}
                {neighborEntities.length > 15 && <span className="font-mono text-[10px] text-text-muted">+{neighborEntities.length - 15}</span>}
              </div>
            </div>
          )}
        </div>
      );
    }

    if (!chunks?.length) return <p className="py-8 text-center font-mono text-[11px] text-text-muted/50">该阶段暂无详细数据</p>;

    const top5 = chunks.slice(0, 5);
    return (
      <div className="space-y-2">
        {top5.map((chunk: RetrievedChunk, i: number) => (
          <div key={chunk.chunk_uid || i} className="rounded-lg border border-border-subtle bg-white p-2.5">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-[10px] font-semibold text-text-muted truncate max-w-[70%]">
                {chunk.document_name}
                {chunk.merge_info && ((chunk.merge_info as any)?.source === "graph_only")
                  ? <span className="ml-1 text-emerald-500">·图谱</span>
                  : chunk.merge_info && ((chunk.merge_info as any)?.source === "vector_graph_overlap")
                  ? <span className="ml-1 text-indigo-400">·双路</span> : null}
              </span>
              <span className="font-mono text-[11px] font-bold text-accent">{scoreOf(chunk)}</span>
            </div>
            <p className="line-clamp-3 font-body text-[12px] leading-relaxed text-text-secondary">{chunk.content}</p>
          </div>
        ))}
        {chunks.length > 5 && (
          <p className="text-center font-mono text-[10px] text-text-muted">显示前5条，共{chunks.length}条</p>
        )}
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar">
      {/* Header */}
      <div className="px-4 pt-4 pb-2">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-[11px] text-text-muted">第{round}轮</span>
          <span className="font-mono text-[11px] font-bold text-accent">{levelName(selectedTrace?.selected_strategy)}</span>
        </div>
        {/* Grade */}
        <div className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-semibold ${
          selectedRun?.grade?.can_answer === false ? "bg-error-bg text-error" : "bg-success-bg text-success"}`}>
          <CheckCircle2 size={13} />
          {selectedRun?.grade?.can_answer === false ? "资料不足" : "PASS"}
          {selectedRun?.grade?.reason && <span className="font-normal text-[11px] opacity-70">· {selectedRun.grade.reason}</span>}
        </div>
      </div>

      {/* Metric cards */}
      <div className="px-3 pb-2">
        <div className="flex flex-wrap gap-1.5">
          {cards.map((c) => (
            <button
              key={c.key}
              onClick={() => setCard(c.key)}
              className={`flex flex-col items-center rounded-lg px-2.5 py-1.5 min-w-[3rem] transition-colors ${
                card === c.key ? "bg-accent-subtle ring-1 ring-accent/20" : "bg-white border border-border-subtle hover:bg-bg-surface"
              }`}
            >
              <c.icon size={13} className={c.color} />
              <span className={`font-mono text-[11px] font-bold ${c.color}`}>{c.value}</span>
              <span className="font-mono text-[9px] text-text-muted">{c.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Detail area */}
      <div className="px-4 py-2 border-t border-border-subtle">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-[11px] font-bold text-text-muted uppercase tracking-wider">
            {cards.find(c => c.key === card)?.label} 结果
          </span>
        </div>
        {renderDetail()}
      </div>

      {/* Knowledge Graph (default expanded) */}
      {graphData && graphData.nodes?.length > 0 && (
        <div className="px-4 py-2 border-t border-border-subtle">
          <button onClick={() => setShowGraph(!showGraph)}
            className="flex w-full items-center justify-between font-mono text-[11px] font-bold text-text-muted uppercase tracking-wider">
            <span>知识图谱 · {graphData.nodes.length} 节点 · {graphData.edges?.length || 0} 边 · 匹配{matchedEntities.length} · 邻居{neighborEntities.length}</span>
            <ChevronDown size={12} className={`transition-transform ${showGraph ? "rotate-180" : ""}`} />
          </button>
          {showGraph && (
            <div className="mt-2">
              <KnowledgeGraph data={graphData} height={280} />
            </div>
          )}
        </div>
      )}

      {/* Rewrite attempts */}
      {selectedTrace?.rewrite_attempts?.length > 0 && (
        <div className="px-4 py-2 border-t border-border-subtle">
          <span className="font-mono text-[11px] font-bold text-text-muted uppercase tracking-wider">重写记录</span>
          <div className="space-y-1 mt-1.5">
            {selectedTrace.rewrite_attempts.map((att: string, i: number) => (
              <div key={i} className="rounded-lg border border-border-subtle bg-white p-2 font-mono text-[11px] text-text-secondary truncate">{att}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════ */

export default function Chat({ sessionId, folderId, onSessionChange, onDataChanged }: ChatProps) {
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [enableGraph, setEnableGraph] = useState(true);
  const [folders, setFolders] = useState<FolderItem[]>([]);
  const [selectedTraceIdx, setSelectedTraceIdx] = useState<number | null>(null);
  const [detailCard, setDetailCard] = useState<string>("fusion");

  useEffect(() => { listFolderTree().then(setFolders).catch(() => {}); }, []);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const streamingRef = useRef(false);

  useEffect(() => {
    if (!sessionId) { setMessages([]); setSelectedTraceIdx(null); return; }
    if (streamingRef.current) return;
    getSession(sessionId).then((s) => {
      setMessages(s.messages);
      // Auto-select last assistant
      const lastAi = [...s.messages].reverse().findIndex((m) => m.role === "assistant");
      if (lastAi >= 0) setSelectedTraceIdx(s.messages.length - 1 - lastAi);
    }).catch(() => setMessages([]));
  }, [sessionId]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Selected trace data
  const selectedMsg = selectedTraceIdx != null ? messages[selectedTraceIdx] : null;
  const selectedTrace = selectedMsg?.rag_trace;
  const selectedRun = selectedTrace?.runs?.at(-1) || selectedTrace?.runs?.[0];
  const finalContexts = selectedRun?.final_results || selectedMsg?.contexts || [];
  const rawGraph = (selectedTrace as any)?.graph_data;
  const graphData: GraphData | null = rawGraph?.subgraph || (rawGraph?.nodes ? rawGraph : null);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;
    setInput(""); setLoading(true); streamingRef.current = true;
    const ai = messages.length + 1;
    setMessages((prev) => [...prev, { role: "user", content: question }, { role: "assistant", content: "", steps: [], streaming: true, contexts: [] }]);
    // Auto-select this new message for trace panel
    setSelectedTraceIdx(messages.length + 1);
    try {
      await streamChat(question, sessionId, (event) => {
        if (event.type === "meta" && event.session_id) onSessionChange(event.session_id);
        if (event.type === "step") setMessages((prev) => prev.map((m, i) => i === ai ? { ...m, steps: [...(m.steps || []), event as StreamStep] } : m));
        if (event.type === "answer_delta") setMessages((prev) => prev.map((m, i) => i === ai ? { ...m, content: m.content + (event.content || "") } : m));
        if (event.type === "done") { setLoading(false); streamingRef.current = false; setMessages((prev) => prev.map((m, i) => i === ai ? { ...m, content: event.answer || m.content, contexts: event.contexts || [], rag_trace: event.debug_info, streaming: false } : m)); onDataChanged(); }
        if (event.type === "error") throw new Error(event.message || "生成失败");
      }, folderId || undefined, enableGraph);
    } catch (error) { setMessages((prev) => prev.map((m, i) => i === ai ? { ...m, streaming: false, error: error instanceof Error ? error.message : String(error) } : m)); }
    finally { setLoading(false); streamingRef.current = false; }
  }

  return (
    <div className="flex h-full overflow-hidden">
      <div className="relative flex h-full flex-1 flex-col bg-bg-page">
        <header className="z-20 flex h-14 shrink-0 items-center justify-between border-b border-border-subtle bg-white/80 backdrop-blur px-8">
          <div className="flex items-center gap-4">
            <span className="font-display text-[17px] text-text-primary tracking-tight">知识检索</span>
            <span className="font-mono text-[10px] text-text-muted">{sessionId ? sessionId.slice(0, 12) + "…" : "新会话"}</span>
          </div>
          {loading && (
            <span className="flex items-center gap-2 font-mono text-[11px] text-accent">
              <Loader2 size={14} className="animate-spin" /> 检索中
            </span>
          )}
        </header>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <div className="mx-auto max-w-2xl px-6 py-8">
            {messages.length === 0 && (
              <div className="flex flex-col items-center py-16 animate-scale-in">
                <h2 className="font-display text-[28px] text-text-primary tracking-tight mb-3">知识引擎就绪</h2>
                <p className="font-body text-[15px] text-text-secondary mb-12 max-w-md text-center leading-relaxed">
                  上传企业文档后，通过自然语言进行精准检索问答。
                </p>
                <div className="grid grid-cols-3 gap-4 w-full max-w-xl mb-12">
                  {features.map((f) => (
                    <div key={f.title} className="flex flex-col items-center gap-3 rounded-xl border border-border-subtle bg-white p-5 text-center">
                      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-subtle">
                        <f.icon size={17} className="text-accent" />
                      </div>
                      <p className="font-body text-[13px] font-semibold text-text-primary">{f.title}</p>
                      <p className="font-body text-[11px] text-text-muted leading-relaxed">{f.desc}</p>
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 rounded-full border border-border-subtle bg-white px-4 py-2">
                  <Zap size={11} className="text-accent" />
                  <span className="font-mono text-[11px] text-text-muted">Ctrl + Enter 发送 · 先上传文档再提问</span>
                </div>
              </div>
            )}

            <div className="space-y-8">
              {messages.map((message, index) => {
                const isSelected = selectedTraceIdx === index && message.role === "assistant";
                return message.role === "user" ? (
                  <div key={index} className="flex flex-col items-end animate-fade-up">
                    <div className="max-w-[80%] rounded-2xl rounded-br-md bg-accent-subtle px-5 py-3.5 font-body text-[14px] leading-relaxed text-text-primary">
                      {message.content}
                    </div>
                  </div>
                ) : (
                  <div key={index} className="flex flex-col items-start gap-2 animate-fade-up">
                    <div
                      className={`max-w-[95%] rounded-2xl rounded-bl-md border bg-white px-6 py-5 shadow-sm cursor-pointer transition-colors ${
                        isSelected ? "border-accent ring-1 ring-accent/20" : "border-border-subtle hover:border-border-default"
                      }`}
                      onClick={() => setSelectedTraceIdx(index)}
                    >
                      {/* 流式步骤 */}
                      {message.streaming && !message.content && (
                        <RagLiveSteps steps={message.steps || []} streaming={!!message.streaming} />
                      )}

                      {/* 答案正文 */}
                      {message.content ? (
                        <div className="font-body text-[14px] leading-[1.8] text-text-primary whitespace-pre-wrap">
                          {message.content}
                        </div>
                      ) : !message.streaming && message.error ? null : !message.streaming && !message.steps?.length ? (
                        <div className="font-body text-[13px] text-text-muted">暂无回答</div>
                      ) : message.streaming && !message.steps?.length ? (
                        <div className="flex items-center gap-2 text-text-muted">
                          <Loader2 size={14} className="animate-spin text-accent" />
                          <span className="font-body text-[13px]">检索知识库中…</span>
                        </div>
                      ) : null}

                      {message.error && (
                        <div className="mt-3 rounded-lg border border-error/15 bg-error-bg px-4 py-2.5 font-mono text-[12px] text-error">
                          ✕ {message.error}
                        </div>
                      )}

                      {/* 选中指示 */}
                      {isSelected && !message.streaming && message.content && (
                        <div className="mt-2 flex items-center gap-1.5 text-[10px] text-accent">
                          <Network size={10} />
                          <span>右侧面板显示此检索</span>
                        </div>
                      )}
                    </div>

                    {/* 检索过程折叠面板 */}
                    {!message.streaming && message.steps && message.steps.length > 0 && message.content && (
                      <RagStepsCollapse steps={message.steps} />
                    )}
                  </div>
                );
              })}
              <div ref={bottomRef} />
            </div>
          </div>
        </div>

        <div className="border-t border-border-subtle bg-white/70 backdrop-blur px-8 py-5">
          <div className="relative mx-auto max-w-3xl">
            <div className="flex items-center gap-3 mb-2">
              <FolderPicker
                folders={folders}
                selectedFolderId={folderId}
                onChange={() => {}}
              />
              <label className="flex items-center gap-1.5 text-[11px] text-stone-500 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={enableGraph}
                  onChange={(e) => setEnableGraph(e.target.checked)}
                  className="accent-indigo-500 h-3.5 w-3.5"
                />
                GraphRAG
              </label>
            </div>
            <textarea
              className="w-full resize-none rounded-2xl border border-border-default bg-bg-input px-5 py-4 pr-14 font-body text-[14px] text-text-primary placeholder:text-text-muted outline-none transition-all duration-200 focus:border-accent/30 focus:ring-2 focus:ring-accent/10"
              placeholder="输入问题，Ctrl + Enter 发送…"
              rows={2} value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.ctrlKey && e.key === "Enter") sendMessage(); }}
            />
            <button onClick={sendMessage} disabled={loading}
              className="absolute bottom-3 right-3 rounded-xl bg-accent p-2.5 text-white transition-all hover:bg-accent-hover disabled:opacity-40 group">
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} className="transition-transform group-hover:translate-x-0.5" />}
            </button>
          </div>
        </div>
      </div>

      {/* ── Right Panel (unified) ── */}
      <aside className="flex w-[24rem] shrink-0 flex-col border-l border-border-subtle bg-bg-panel">
        <RightPanelBody
          selectedMsg={selectedMsg}
          selectedTrace={selectedTrace}
          selectedRun={selectedRun}
          finalContexts={finalContexts}
          graphData={graphData}
          messages={messages}
          selectedTraceIdx={selectedTraceIdx}
        />
        <div className="border-t border-border-subtle px-3 py-2 shrink-0">
          <Link to="/trace" className="font-mono text-[11px] text-accent hover:underline flex items-center gap-1">
            <ScanSearch size={12} /> 完整追踪 →
          </Link>
        </div>
      </aside>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border-subtle bg-white p-2.5">
      <span className="block font-mono text-[9px] text-text-muted uppercase tracking-wide mb-0.5">{label}</span>
      <span className="font-mono text-[12px] font-semibold text-text-primary">{value}</span>
    </div>
  );
}
