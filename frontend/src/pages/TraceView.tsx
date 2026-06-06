import { ArrowLeft, BarChart3, Combine, Download, FileText, Info, Layers, Network, SortDesc, Type, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getSession, type ChatMessage, type RagRun, type RetrievedChunk } from "../api";

export default function TraceView({ sessionId }: { sessionId: string | null }) {
  const [message, setMessage] = useState<ChatMessage | null>(null);
  useEffect(() => {
    if (!sessionId) { setMessage(null); return; }
    getSession(sessionId).then((s) => { const a = [...s.messages].reverse().find((m) => m.role === "assistant" && m.rag_trace); setMessage(a || null); });
  }, [sessionId]);

  const trace = message?.rag_trace;
  const run = useMemo(() => trace?.runs?.at(-1) || trace?.runs?.[0], [trace]);
  const finalResults = run?.final_results || [];

  return (
    <div className="flex h-full flex-col overflow-hidden bg-bg-page">
      <header className="z-20 flex h-14 shrink-0 items-center justify-between border-b border-border-subtle bg-white/80 backdrop-blur px-8">
        <div className="flex items-center gap-5">
          <Link to="/" className="flex items-center gap-2 font-mono text-[10px] font-semibold text-text-muted uppercase tracking-wider hover:text-accent transition-colors">
            <ArrowLeft size={14} /> 返回
          </Link>
          <div className="h-5 w-px bg-border-subtle" />
          <span className="font-mono text-[10px] text-text-muted">SESSION <span className="text-accent font-bold">{sessionId?.slice(0, 14) || "—"}</span></span>
        </div>
        <button className="flex items-center gap-2 rounded-lg border border-border-default px-4 py-2 font-mono text-[11px] text-text-secondary hover:border-accent/20 hover:text-accent transition-all">
          <Download size={13} /> 导出
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-page custom-scrollbar">
        <div className="mx-auto max-w-7xl">
          <div className="mb-12">
            <h1 className="font-display text-[32px] text-text-primary tracking-tight mb-3">检索详情</h1>
            <p className="font-body text-[15px] text-text-secondary max-w-md leading-relaxed">
              问题改写 → 混合召回 → RRF → Rerank → 父块合并 → 最终上下文。
            </p>
          </div>

          {!run ? (
            <div className="flex flex-col items-center gap-4 py-28 text-center animate-scale-in">
              <BarChart3 size={32} className="text-text-muted/20" />
              <p className="font-display text-[20px] text-text-muted">暂无检索记录</p>
              <p className="font-body text-[14px] text-text-muted/50">发起问答后可查看完整追踪</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 stagger">
              <div className="lg:col-span-4 space-y-5">
                <div className="rounded-xl border border-border-subtle bg-white p-6">
                  <h3 className="flex items-center gap-2 font-display text-[16px] text-text-primary mb-5 pb-3 border-b border-border-subtle">
                    <Info size={16} className="text-accent" /> 查询信息
                  </h3>
                  <Field label="原始问题" value={trace?.original_question || "—"} />
                  <Field label="改写后" value={trace?.contextualized_question || trace?.selected_query || "—"} />
                  <Field label="检索 Query" value={trace?.selected_query || run.query || "—"} accent />
                  <div className="grid grid-cols-2 gap-3">
                    <Field label="策略" value={trace?.selected_strategy || run.strategy || "—"} compact />
                    <Field label="判定" value={run.grade?.can_answer ? "PASS" : "不足"} compact highlight={run.grade?.can_answer} />
                  </div>
                  <Field label="原因" value={run.grade?.reason || "—"} />
                </div>

                <div className="rounded-xl border border-border-subtle bg-white p-6">
                  <h3 className="flex items-center gap-2 font-display text-[16px] text-text-primary mb-5 pb-3 border-b border-border-subtle">
                    <BarChart3 size={16} className="text-accent" /> 性能统计
                  </h3>
                  <Stat icon={Layers} label="向量" count={`${run.summary?.dense_count ?? 0}`} ms={run.timings?.dense_ms} />
                  <Stat icon={Type} label="关键词" count={`${run.summary?.keyword_count ?? 0}`} ms={run.timings?.keyword_ms} />
                  <Stat icon={Combine} label="RRF" count={`${run.summary?.rrf_count ?? 0}`} ms={run.timings?.rrf_ms} />
                  <Stat icon={SortDesc} label="Rerank" count={`${run.summary?.rerank_count ?? 0}`} ms={run.timings?.rerank_ms} />
                  <Stat icon={Network} label="合并" count={`${run.summary?.merged_parent_count ?? 0}`} ms={run.timings?.auto_merge_ms} />
                  <div className="mt-4 flex items-center justify-between rounded-lg bg-accent-subtle px-4 py-3">
                    <span className="font-body text-[13px] font-semibold">最终送入</span>
                    <span className="flex items-center gap-1 font-mono text-[13px] font-bold text-accent">
                      <Zap size={12} /> {run.summary?.final_count ?? finalResults.length} chunks
                    </span>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-8 space-y-5">
                <Tabs run={run} />
                <div className="space-y-4">
                  <h2 className="font-display text-[17px] text-text-primary">最终上下文</h2>
                  {finalResults.map((chunk) => (
                    <div key={chunk.chunk_uid} className="rounded-xl border border-border-subtle bg-white overflow-hidden">
                      <div className="flex items-start justify-between p-5 pb-0">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="rounded-md bg-accent-subtle px-2 py-0.5 font-mono text-[10px] font-bold text-accent">L{chunk.chunk_level}</span>
                            <h4 className="font-body text-[13px] font-semibold text-text-primary">{chunk.document_name}</h4>
                          </div>
                          <p className="font-mono text-[10px] text-text-muted">{chunk.position_hint || "—"} · {chunk.section_title || "正文"}</p>
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-border-subtle bg-bg-surface p-2.5 shrink-0 ml-4">
                          <Score l="RERANK" v={chunk.rerank_score} p />
                          <Score l="RRF" v={chunk.rrf_score} />
                          <Score l="DENSE" v={chunk.dense_score} />
                          <Score l="SPARSE" v={chunk.sparse_score} />
                        </div>
                      </div>
                      <div className="p-5 pt-4">
                        <p className="font-body text-[13px] leading-[1.85] text-text-secondary">{chunk.content}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Tabs({ run }: { run: RagRun }) {
  const items = [`最终 · ${run.summary?.final_count ?? 0}`, `合并 · ${run.summary?.merged_parent_count ?? 0}`, `Rerank · ${run.summary?.rerank_count ?? 0}`, `RRF · ${run.summary?.rrf_count ?? 0}`, `向量 · ${run.summary?.dense_count ?? 0}`, `关键词 · ${run.summary?.keyword_count ?? 0}`];
  return (
    <div className="flex gap-1 rounded-lg border border-border-subtle bg-white p-1">
      {items.map((t, i) => (
        <button key={t} className={`flex-1 truncate rounded-md px-3 py-2 font-mono text-[10px] font-semibold transition-all ${
          i === 0 ? "bg-accent-subtle text-accent" : "text-text-muted hover:bg-bg-surface"}`}>{t}</button>
      ))}
    </div>
  );
}

function Field({ label, value, compact, highlight, accent }: { label: string; value: string; compact?: boolean; highlight?: boolean; accent?: boolean }) {
  return (
    <div className={compact ? "" : "mb-4"}>
      <span className="block mb-1 font-mono text-[9px] font-bold text-text-muted uppercase tracking-[0.12em]">{label}</span>
      <p className={`font-body text-[13px] leading-relaxed px-3 py-2 rounded-lg border border-border-subtle bg-bg-surface ${accent ? "text-accent font-semibold" : highlight === true ? "text-success font-semibold" : highlight === false ? "text-error font-semibold" : "text-text-primary"}`}>{value}</p>
    </div>
  );
}

function Stat({ icon: Icon, label, count, ms }: { icon: typeof Layers; label: string; count: string; ms?: number }) {
  return (
    <div className="flex items-center justify-between py-2.5 px-1 rounded-lg hover:bg-bg-surface transition-colors">
      <div className="flex items-center gap-3"><Icon size={14} className="text-text-muted" /><span className="font-body text-[12px] text-text-secondary">{label}</span></div>
      <div className="flex items-center gap-4"><span className="font-mono text-[11px] font-semibold">{count}</span><span className="w-10 text-right font-mono text-[10px] text-text-muted">{ms ?? 0}ms</span></div>
    </div>
  );
}

function Score({ l, v, p }: { l: string; v?: number | null; p?: boolean }) {
  return <div className="flex justify-between gap-3"><span className="font-mono text-[9px] text-text-muted">{l}</span><span className={`font-mono text-[10px] font-bold ${p ? "text-accent" : "text-text-secondary"}`}>{typeof v === "number" ? v.toFixed(3) : "—"}</span></div>;
}
