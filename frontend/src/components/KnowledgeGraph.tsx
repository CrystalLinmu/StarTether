import { useEffect, useRef, useState } from "react";
import { Network } from "vis-network/standalone";
import { DataSet } from "vis-data/standalone";
import type { GraphData } from "../api";
import { Search, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

const TYPE_COLORS: Record<string, { bg: string; border: string }> = {
  PERSON: { bg: "#6366F1", border: "#4F46E5" },
  ORGANIZATION: { bg: "#10B981", border: "#059669" },
  LOCATION: { bg: "#F59E0B", border: "#D97706" },
  CONCEPT: { bg: "#8B5CF6", border: "#7C3AED" },
  TIME: { bg: "#06B6D4", border: "#0891B2" },
  EVENT: { bg: "#EF4444", border: "#DC2626" },
  LAW_REGULATION: { bg: "#F97316", border: "#EA580C" },
  TECHNOLOGY: { bg: "#3B82F6", border: "#2563EB" },
  PRODUCT: { bg: "#A855F7", border: "#9333EA" },
  OTHER: { bg: "#6B7280", border: "#4B5563" },
};

interface Props {
  data: GraphData | null;
  height?: number;
  onNodeClick?: (nodeId: string) => void;
}

export default function KnowledgeGraph({ data, height = 360, onNodeClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const [selectedNode, setSelectedNode] = useState<{ label: string; type: string; description: string } | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (!containerRef.current || !data?.nodes?.length) return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const nodes = new (DataSet as any)(
      data.nodes.map((n: any) => ({
        id: n.id,
        label: n.label,
        group: n.type || n.group || "OTHER",
        title: `<b>${n.label}</b><br/>${n.type}<br/>${n.description || ""}`,
        value: Math.max(8, Math.min(30, (n.chunk_count || 0) * 4 + 10)),
        color: TYPE_COLORS[n.type || n.group] || TYPE_COLORS.OTHER,
        font: { size: 12, face: "system-ui, sans-serif", color: "#1e293b" },
        borderWidth: 2,
        shape: "dot",
      })),
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const edges = new (DataSet as any)(
      (data.edges || []).map((e: any, i: number) => ({
        id: e.id || `e${i}`,
        from: e.source,
        to: e.target,
        label: e.label,
        title: e.description || e.label,
        arrows: { to: { enabled: true, scaleFactor: 0.6 } },
        font: { size: 9, align: "middle", color: "#94a3b8" },
        color: { color: "#cbd5e1", hover: "#6366F1" },
        width: 1.2,
        smooth: { enabled: true, type: "continuous" as const, roundness: 0.5 },
      })),
    );

    const options = {
      nodes: {
        scaling: { min: 8, max: 30 },
      },
      edges: {
        smooth: { enabled: true, type: "continuous" as const, roundness: 0.5 },
      },
      physics: {
        enabled: true,
        solver: "barnesHut" as const,
        barnesHut: { gravitationalConstant: -2000, centralGravity: 0.3, springLength: 180, springConstant: 0.04 },
        stabilization: { iterations: 80 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 150,
        zoomView: true,
        dragView: true,
        navigationButtons: false,
      },
      groups: Object.fromEntries(
        Object.entries(TYPE_COLORS).map(([type, colors]) => [
          type,
          {
            color: { background: colors.bg, border: colors.border, highlight: { background: colors.bg, border: colors.border } },
          },
        ]),
      ),
    };

    const network = new Network(containerRef.current, { nodes, edges }, options as any);
    networkRef.current = network;

    network.on("click", (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0] as string;
        const nodeData = (data.nodes || []).find((n) => n.id === nodeId);
        if (nodeData) {
          setSelectedNode({
            label: nodeData.label,
            type: nodeData.type,
            description: nodeData.description,
          });
        }
        onNodeClick?.(nodeId);
      } else {
        setSelectedNode(null);
      }
    });

    return () => {
      network.destroy();
    };
  }, [data, onNodeClick]);

  // Search filter
  const filtered = data?.nodes?.filter((n) =>
    !search || n.label.toLowerCase().includes(search.toLowerCase()),
  );

  const handleFit = () => networkRef.current?.fit({ animation: true });
  const handleZoomIn = () => {
    const s = networkRef.current?.getScale() || 1;
    networkRef.current?.moveTo({ scale: s * 1.3 });
  };
  const handleZoomOut = () => {
    const s = networkRef.current?.getScale() || 1;
    networkRef.current?.moveTo({ scale: s * 0.7 });
  };

  if (!data?.nodes?.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2 text-text-muted py-12">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="opacity-20">
          <circle cx="5" cy="6" r="2" /><circle cx="19" cy="6" r="2" /><circle cx="12" cy="18" r="2" />
          <line x1="7" y1="6.5" x2="11" y2="16" /><line x1="17" y1="6.5" x2="13" y2="16" /><line x1="5" y1="8" x2="12" y2="16" />
        </svg>
        <p className="font-body text-[12px]">暂无知谱数据</p>
        <p className="font-mono text-[10px] text-text-muted/50">启用图谱增强后在此展示</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5">
        <div className="relative flex-1">
          <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            className="w-full rounded-md border border-border-subtle bg-white pl-6 pr-2 py-1 font-body text-[11px] text-text-primary placeholder:text-text-muted/50 outline-none focus:ring-1 focus:ring-accent/30"
            placeholder="搜索实体…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <button onClick={handleZoomIn} className="p-1 rounded hover:bg-bg-surface text-text-muted" title="放大"><ZoomIn size={13} /></button>
        <button onClick={handleZoomOut} className="p-1 rounded hover:bg-bg-surface text-text-muted" title="缩小"><ZoomOut size={13} /></button>
        <button onClick={handleFit} className="p-1 rounded hover:bg-bg-surface text-text-muted" title="适应"><Maximize2 size={13} /></button>
      </div>

      {/* Graph canvas */}
      <div ref={containerRef} className="rounded-xl border border-border-subtle bg-white overflow-hidden" style={{ height }} />

      {/* Legend */}
      <div className="flex flex-wrap gap-1">
        {Object.entries(TYPE_COLORS).slice(0, 6).map(([type, colors]) => (
          <span key={type} className="inline-flex items-center gap-1 rounded-full bg-bg-surface px-2 py-0.5 font-mono text-[9px] text-text-muted">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: colors.bg }} />
            {type}
          </span>
        ))}
      </div>

      {/* Selected node detail */}
      {selectedNode && (
        <div className="rounded-xl border border-border-subtle bg-white p-3 animate-fade-up">
          <div className="flex items-center justify-between mb-1">
            <span className="font-body text-[12px] font-semibold text-text-primary">{selectedNode.label}</span>
            <span className="font-mono text-[9px] text-text-muted">{selectedNode.type}</span>
          </div>
          {selectedNode.description && (
            <p className="font-body text-[11px] text-text-secondary leading-relaxed">{selectedNode.description}</p>
          )}
        </div>
      )}

      {/* Stats */}
      <div className="flex items-center gap-3 text-[10px] text-text-muted font-mono">
        <span>{data.nodes.length} 实体</span>
        <span>{data.edges?.length || 0} 关系</span>
        {search && filtered && <span>匹配 {filtered.length}</span>}
      </div>
    </div>
  );
}
