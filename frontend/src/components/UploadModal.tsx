import React, { useState, useRef } from "react";
import {
  X, Upload, FileText, CheckCircle2, Loader2, Circle,
  Database, Layers, Zap, Hash, FileSearch, Cpu, Search,
  GitBranch,
} from "lucide-react";
import type { StreamStep, FolderItem } from "../api";

export type UploadStats = {
  l1_chunks?: number;
  l2_chunks?: number;
  l3_chunks?: number;
  total_chunks?: number;
  chars?: number;
  document_name?: string;
  folder_id?: string;
  filename?: string;
  entities_extracted?: number;
};

interface Props {
  folders: FolderItem[];
  selectedFolderId: string | null;
  open: boolean;
  steps: StreamStep[];
  status: string;
  stats: UploadStats | null;
  graphStats: { entities?: number; relationships?: number } | null;
  uploading: boolean;
  onClose: () => void;
  onUpload: (file: File, folderId: string | null) => void;
}

const STEP_LABELS: Record<string, string> = {
  file_saved: "文件已保存",
  parse_start: "文档解析中…",
  parse_done: "文档解析完成",
  split_start: "三级分块中…",
  split_done: "分块完成",
  insert_start: "写入数据库中…",
  insert_done: "数据库写入完成",
  milvus_start: "向量化中…",
  milvus_done: "向量入库完成",
  commit_done: "入库完成",
  graph_extract_start: "GraphRAG 实体提取中…",
  graph_extract_done: "GraphRAG 提取完成",
  empty: "无可用内容",
};

export default function UploadModal({
  folders, selectedFolderId, open, steps, status, stats, graphStats, uploading,
  onClose, onUpload,
}: Props) {
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  if (!open) return null;

  const doneSteps = steps.filter((s) => s.code.endsWith("_done") || s.code === "empty");
  const runningCodes = steps
    .filter((s) => s.code.endsWith("_start"))
    .map((s) => s.code.replace("_start", ""))
    .filter((base) => !steps.some((d) => d.code === base + "_done"));
  const hasRunning = runningCodes.length > 0;
  const lastRunning = [...steps].reverse().find((s) => s.code.endsWith("_start"));
  const hasError = steps.some((s) => s.code === "error");
  const isDone = !!stats && !uploading;

  function handleFile(f: File) {
    onUpload(f, selectedFolderId);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm p-8">
      <div className="w-full max-w-lg rounded-2xl bg-white shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border-subtle">
          <h2 className="font-display text-[17px] text-text-primary tracking-tight">
            {isDone ? "入库完成" : uploading ? "文档入库中…" : "上传文档"}
          </h2>
          <button onClick={onClose} className="rounded-lg p-1.5 text-text-muted hover:bg-bg-surface transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="px-6 py-5 space-y-4 max-h-[65vh] overflow-y-auto custom-scrollbar">
          {/* ── File Picker (before upload) ── */}
          {!uploading && !isDone && (
            <div
              className={`relative flex flex-col items-center gap-4 rounded-xl border-2 border-dashed p-10 transition-colors cursor-pointer ${
                dragOver ? "border-accent bg-accent-subtle" : "border-border-default hover:border-accent/30 bg-bg-surface"
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault(); setDragOver(false);
                const f = e.dataTransfer.files[0];
                if (f) handleFile(f);
              }}
              onClick={() => fileRef.current?.click()}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-subtle">
                <Upload size={22} className="text-accent" />
              </div>
              <div className="text-center">
                <p className="font-body text-[14px] font-semibold text-text-primary">点击或拖拽文件到此处</p>
                <p className="font-mono text-[11px] text-text-muted mt-1">支持 .txt / .pdf / .docx</p>
              </div>
              <input
                ref={fileRef}
                type="file"
                className="hidden"
                accept=".txt,.pdf,.docx"
                onChange={(e) => {
                  const f = e.target.files?.[0]; e.target.value = "";
                  if (f) handleFile(f);
                }}
              />
            </div>
          )}

          {/* ── Streaming Steps ── */}
          {(uploading || isDone) && (
            <div className="space-y-1.5">
              {doneSteps.map((s, i) => (
                <div key={i} className="flex items-center gap-2.5 text-[13px] animate-fade-up">
                  {s.code === "error" ? (
                    <Circle size={14} className="text-error shrink-0" />
                  ) : (
                    <CheckCircle2 size={14} className="text-success shrink-0" />
                  )}
                  <span className="font-body text-text-secondary flex-1">
                    {s.label || STEP_LABELS[s.code] || s.code}
                  </span>
                  {s.elapsed_ms != null && (
                    <span className="font-mono text-[10px] text-text-muted">{s.elapsed_ms}ms</span>
                  )}
                </div>
              ))}
              {hasRunning && lastRunning && (
                <div className="flex items-center gap-2.5 text-[13px] animate-fade-up">
                  <Loader2 size={14} className="animate-spin text-accent shrink-0" />
                  <span className="font-body text-accent font-medium flex-1">
                    {lastRunning.label || STEP_LABELS[lastRunning.code] || lastRunning.code}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ── Done: Stats Panel ── */}
          {isDone && stats && (
            <div className="space-y-3 animate-fade-up border-t border-border-subtle pt-4">
              <h3 className="font-mono text-[10px] font-bold text-text-muted uppercase tracking-[0.15em]">入库统计</h3>

              {/* Chunk stats */}
              <div className="grid grid-cols-3 gap-2">
                <StatCard icon={Layers} label="L1 大块" value={stats.l1_chunks ?? "—"} color="text-indigo-500" bg="bg-indigo-50" />
                <StatCard icon={Layers} label="L2 中块" value={stats.l2_chunks ?? "—"} color="text-violet-500" bg="bg-violet-50" />
                <StatCard icon={Layers} label="L3 小块" value={stats.l3_chunks ?? "—"} color="text-emerald-500" bg="bg-emerald-50" />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <StatCard icon={FileText} label="总块数" value={stats.total_chunks ?? "—"} color="text-amber-500" bg="bg-amber-50" />
                <StatCard icon={Hash} label="总字数" value={stats.chars != null ? stats.chars.toLocaleString() : "—"} color="text-rose-500" bg="bg-rose-50" />
              </div>

              {/* GraphRAG stats */}
              {graphStats && (graphStats.entities != null) && (
                <div className="space-y-2">
                  <h4 className="font-mono text-[10px] font-bold text-text-muted uppercase tracking-[0.15em]">GraphRAG</h4>
                  <div className="grid grid-cols-2 gap-2">
                    <StatCard icon={GitBranch} label="实体" value={graphStats.entities ?? "—"} color="text-indigo-500" bg="bg-indigo-50" />
                    <StatCard icon={GitBranch} label="关系" value={graphStats.relationships ?? "—"} color="text-pink-500" bg="bg-pink-50" />
                  </div>
                </div>
              )}

              {/* Pipeline flow */}
              <div className="rounded-xl border border-border-subtle bg-bg-surface p-3">
                <div className="flex items-center justify-center gap-2 text-[11px] font-mono text-text-muted">
                  <span>文档</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="flex items-center gap-1"><FileSearch size={11} /> 解析</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="flex items-center gap-1"><Layers size={11} /> 切分</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="flex items-center gap-1"><Database size={11} /> PG</span>
                  <span className="text-text-muted/30">→</span>
                  <span className="flex items-center gap-1"><Cpu size={11} /> Milvus</span>
                </div>
              </div>

              <p className="text-center font-mono text-[10px] text-text-muted">
                {stats.document_name || stats.filename}{graphStats?.entities != null ? ` · ${graphStats.entities} 实体已索引` : ""}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-border-subtle bg-bg-surface">
          {isDone ? (
            <button
              onClick={onClose}
              className="rounded-lg bg-accent px-6 py-2 font-body text-[13px] font-semibold text-white transition-all hover:bg-accent-hover"
            >
              关闭
            </button>
          ) : uploading ? (
            <button
              disabled
              className="rounded-lg bg-bg-surface px-6 py-2 font-body text-[13px] text-text-muted cursor-not-allowed"
            >
              请等待完成…
            </button>
          ) : (
            <button
              onClick={onClose}
              className="rounded-lg border border-border-default px-6 py-2 font-body text-[13px] text-text-secondary hover:bg-bg-surface transition-colors"
            >
              取消
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color, bg }: {
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string; value: string | number; color: string; bg: string;
}) {
  return (
    <div className={`rounded-xl ${bg} p-3 text-center`}>
      <Icon size={15} className={`mx-auto mb-1 ${color}`} />
      <p className="font-mono text-[10px] text-text-muted mb-0.5">{label}</p>
      <p className={`font-display text-[17px] font-bold ${color}`}>{value}</p>
    </div>
  );
}
