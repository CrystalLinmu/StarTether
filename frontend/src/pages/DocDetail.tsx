import { ArrowLeft, FileText, Layers, GitBranch, Network, Hash, Cpu } from "lucide-react";
import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import KnowledgeGraph from "../components/KnowledgeGraph";
import type { GraphData } from "../api";

interface DocStats {
  l1: number; l2: number; l3: number; total: number; chars: number;
}

export default function DocDetail() {
  const { name } = useParams<{ name: string }>();
  const [stats, setStats] = useState<DocStats | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!name) return;
    fetch(`/documents/${encodeURIComponent(name)}/graph`)
      .then(r => r.json())
      .then(data => {
        setStats(data.stats);
        setGraph(data.graph);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [name]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-bg-page text-text-muted">
        <Loader size={24} className="animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-bg-page overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-4 shrink-0 border-b border-border-subtle bg-white px-6 py-3">
        <Link to="/documents" className="flex items-center gap-1.5 text-text-muted hover:text-accent transition-colors">
          <ArrowLeft size={16} /> <span className="font-body text-[13px]">返回</span>
        </Link>
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <FileText size={16} className="text-text-muted shrink-0" />
          <h1 className="font-display text-[16px] text-text-primary truncate">{decodeURIComponent(name || "")}</h1>
        </div>
      </header>

      {/* Body: Graph (left 60%) + Stats (right 40%) */}
      <div className="flex flex-1 overflow-hidden">
        {/* Graph area */}
        <div className="flex-[3] border-r border-border-subtle bg-white p-4 flex flex-col">
          <h2 className="font-mono text-[11px] font-bold text-text-muted uppercase tracking-wider mb-3 flex items-center gap-2">
            <Network size={14} className="text-accent" /> 知识图谱
            {graph?.nodes?.length ? <span className="font-normal text-text-muted">· {graph.nodes.length} 实体 · {graph.edges?.length || 0} 关系</span> : null}
          </h2>
          <div className="flex-1 min-h-0">
            {graph && graph.nodes?.length > 0 ? (
              <KnowledgeGraph data={graph} height={window.innerHeight - 140} />
            ) : (
              <div className="flex flex-col items-center justify-center h-full gap-2 text-text-muted">
                <Network size={36} className="opacity-20" />
                <p className="font-body text-[13px]">该文档暂无实体数据</p>
                <p className="font-mono text-[10px] text-text-muted/50">上传文档后 GraphRAG 会自动提取</p>
              </div>
            )}
          </div>
        </div>

        {/* Stats area */}
        <div className="flex-[2] overflow-y-auto p-5 space-y-5">
          {stats && (
            <>
              <h2 className="font-mono text-[11px] font-bold text-text-muted uppercase tracking-wider mb-1">文档统计</h2>

              <div className="grid grid-cols-2 gap-3">
                <StatCard icon={Layers} label="L1 大块" value={stats.l1} color="text-indigo-500" bg="bg-indigo-50" />
                <StatCard icon={Layers} label="L2 中块" value={stats.l2} color="text-violet-500" bg="bg-violet-50" />
                <StatCard icon={Layers} label="L3 小块" value={stats.l3} color="text-emerald-500" bg="bg-emerald-50" />
                <StatCard icon={Hash} label="总块数" value={stats.total} color="text-amber-500" bg="bg-amber-50" />
                <StatCard icon={FileText} label="总字数" value={stats.chars.toLocaleString()} color="text-rose-500" bg="bg-rose-50" />
                <StatCard icon={Network} label="实体" value={graph?.nodes?.length ?? 0} color="text-indigo-500" bg="bg-indigo-50" />
                <StatCard icon={GitBranch} label="关系" value={graph?.edges?.length ?? 0} color="text-pink-500" bg="bg-pink-50" />
                <StatCard icon={Cpu} label="社区层级" value="3" color="text-cyan-500" bg="bg-cyan-50" />
              </div>

              {/* Entity list */}
              {graph?.nodes?.length ? (
                <div>
                  <h3 className="font-mono text-[11px] font-bold text-text-muted uppercase tracking-wider mb-2">
                    实体列表 · {graph.nodes.length}
                  </h3>
                  <div className="flex flex-wrap gap-1.5">
                    {graph.nodes.map((n) => (
                      <span key={n.id}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-border-subtle bg-white px-2.5 py-1 font-mono text-[11px]"
                        title={n.description}>
                        <span className="w-2 h-2 rounded-full" style={{
                          backgroundColor: {
                            PERSON: "#6366F1", ORGANIZATION: "#10B981", LOCATION: "#F59E0B",
                            CONCEPT: "#8B5CF6", TIME: "#06B6D4", EVENT: "#EF4444",
                            LAW_REGULATION: "#F97316", TECHNOLOGY: "#3B82F6", PRODUCT: "#A855F7",
                          }[n.type] || "#6B7280"
                        }} />
                        {n.label}
                        <span className="text-[9px] text-text-muted">{n.type}</span>
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* Pipeline flow */}
              <div className="rounded-xl border border-border-subtle bg-white p-4">
                <h3 className="font-mono text-[10px] font-bold text-text-muted uppercase tracking-wider mb-2">处理管道</h3>
                <div className="flex items-center justify-center gap-1.5 text-[11px] font-mono text-text-muted flex-wrap">
                  <span className="rounded-md bg-bg-surface px-2 py-1">解析</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="rounded-md bg-bg-surface px-2 py-1">L1/L2/L3 切分</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="rounded-md bg-bg-surface px-2 py-1">Milvus 向量</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="rounded-md bg-bg-surface px-2 py-1">PG 存储</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="rounded-md bg-accent-subtle px-2 py-1 text-accent">GraphRAG 提取</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="rounded-md bg-accent-subtle px-2 py-1 text-accent">社区检测</span>
                </div>
              </div>
            </>
          )}

          {!stats && (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-text-muted py-16">
              <p className="font-body text-[13px]">无法加载文档统计</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Loader({ size = 24, className = "" }: { size?: number; className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round" opacity="0.25" />
    </svg>
  );
}

function StatCard({ icon: Icon, label, value, color, bg }: {
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string; value: string | number; color: string; bg: string;
}) {
  return (
    <div className={`rounded-xl ${bg} p-3.5 text-center`}>
      <Icon size={17} className={`mx-auto mb-1.5 ${color}`} />
      <p className="font-mono text-[11px] text-text-muted mb-0.5">{label}</p>
      <p className={`font-display text-[20px] font-bold ${color}`}>{value}</p>
    </div>
  );
}
