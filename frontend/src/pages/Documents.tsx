import { FileText, Layers, Hash, Calendar } from "lucide-react";
import type { DocumentItem } from "../api";

export default function Documents({ documents }: { documents: DocumentItem[] }) {
  return (
    <div className="flex-1 overflow-y-auto bg-bg-page p-page custom-scrollbar">
      <div className="max-w-4xl">
        {/* Header */}
        <div className="mb-12">
          <h1 className="font-display text-[32px] text-text-primary tracking-tight mb-3">文档中心</h1>
          <p className="font-body text-[15px] text-text-secondary max-w-md leading-relaxed">
            已索引的知识库文档。在左侧面板上传新的 PDF、DOCX 或 TXT 文件。
          </p>
        </div>

        {/* Stats */}
        {documents.length > 0 && (
          <div className="flex items-center gap-8 mb-8 pb-8 border-b border-border-subtle animate-fade-up">
            <StatFigure value={documents.length} label="文档总数" />
            <StatFigure value={documents.reduce((s, d) => s + d.leaf_chunk_count, 0)} label="L3 叶子块" />
            <StatFigure value={documents.reduce((s, d) => s + d.chunk_count, 0)} label="全部块" />
          </div>
        )}

        {/* Document list */}
        {documents.length === 0 ? (
          <div className="flex flex-col items-center gap-4 py-24 text-center animate-scale-in">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-bg-surface border border-border-subtle">
              <FileText size={28} className="text-text-muted/30" />
            </div>
            <p className="font-display text-[20px] text-text-muted">暂无文档</p>
            <p className="font-body text-[14px] text-text-muted/60">在左侧面板点击「上传文档」开始</p>
          </div>
        ) : (
          <div className="space-y-3 stagger">
            {documents.map((doc) => (
              <div key={doc.document_name}
                className="group flex items-center gap-6 rounded-xl border border-border-subtle bg-white p-5 transition-all hover:border-border-default hover:shadow-sm">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-bg-surface border border-border-subtle">
                  <FileText size={22} className="text-text-muted/50" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="font-body text-[15px] font-semibold text-text-primary truncate mb-1.5">{doc.document_name}</h3>
                  <div className="flex items-center gap-5">
                    <span className="flex items-center gap-1.5 font-mono text-[11px] text-text-muted">
                      <Layers size={11} /> <b className="text-text-secondary">{doc.leaf_chunk_count}</b> L3 叶子
                    </span>
                    <span className="flex items-center gap-1.5 font-mono text-[11px] text-text-muted">
                      <Hash size={11} /> <b className="text-text-secondary">{doc.chunk_count}</b> 全部
                    </span>
                    <span className="flex items-center gap-1.5 font-mono text-[11px] text-text-muted">
                      <Calendar size={11} /> {new Date(doc.created_at).toLocaleDateString("zh-CN")}
                    </span>
                  </div>
                </div>
                <span className="shrink-0 rounded-lg bg-success-bg px-3 py-1.5 font-mono text-[10px] font-semibold text-success">已索引</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatFigure({ value, label }: { value: number; label: string }) {
  return (
    <div>
      <span className="font-display text-[36px] text-text-primary leading-none">{value}</span>
      <span className="block mt-1 font-mono text-[11px] text-text-muted uppercase tracking-wide">{label}</span>
    </div>
  );
}
