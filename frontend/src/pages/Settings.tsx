import { AlertTriangle, Brain, Database, GitBranch, Network, Save, Scissors, Search, Settings2, Zap } from "lucide-react";
import { useEffect, useState } from "react";

interface SettingItem {
  key: string; label: string; value: string; help: string;
  editable: boolean; immediate: boolean;
}

interface GroupDef { title: string; icon: typeof Database; items: SettingItem[]; }

const GROUP_KEYS: Record<string, string[]> = {
  "文档分块": ["l1_chunk_max_chars", "l2_chunk_max_chars", "l3_chunk_max_chars", "l3_chunk_overlap_chars", "max_chunks_per_doc"],
  "向量检索": ["embedding_dim", "embedding_batch_size", "top_k", "candidate_top_k"],
  "父块合并": ["semantic_merge_similarity", "auto_merge_min_children", "auto_merge_child_ratio", "auto_merge_max_parent_chars"],
  "GraphRAG · 实体提取": ["graph_retrieval_enabled", "entity_extraction_batch_size", "entity_extraction_confidence_threshold", "entity_similarity_threshold"],
  "GraphRAG · 图谱检索": ["graph_max_hops", "graph_top_k_chunks", "graph_top_k_entities"],
  "GraphRAG · 社区检测": ["community_levels", "graph_community_min_size"],
};

const GROUP_ICONS: Record<string, typeof Database> = {
  "文档分块": Scissors, "向量检索": Search, "父块合并": Network,
  "GraphRAG · 实体提取": Brain, "GraphRAG · 图谱检索": GitBranch, "GraphRAG · 社区检测": Zap,
};

export default function Settings() {
  const [allItems, setAllItems] = useState<SettingItem[]>([]);
  const [editing, setEditing] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");

  useEffect(() => {
    fetch("/api/settings").then(r => r.json()).then((items: SettingItem[]) => {
      setAllItems(items);
      const init: Record<string, string> = {};
      items.forEach(i => { if (i.editable) init[i.key] = i.value; });
      setEditing(init);
    }).catch(() => setStatus("加载失败"));
  }, []);

  async function handleSave() {
    setSaving(true); setStatus("");
    try {
      const changes: Record<string, any> = {};
      for (const [key, val] of Object.entries(editing)) {
        const orig = allItems.find(i => i.key === key);
        if (!orig || val === orig.value) continue;
        const origVal = String(orig.value);
        const strVal = String(val);
        // Type coerce
        if (origVal === "true" || origVal === "false") changes[key] = strVal === "true";
        else if (origVal.includes(".")) changes[key] = parseFloat(strVal);
        else changes[key] = parseInt(strVal) || strVal;
      }
      if (Object.keys(changes).length === 0) { setStatus("无变更"); setSaving(false); return; }

      const res = await fetch("/api/settings", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
      const data = await res.json();
      setStatus(`已应用 ${data.applied} 项变更：${data.keys.join(", ")}`);
      // Refresh
      const refreshed = await fetch("/api/settings").then(r => r.json());
      setAllItems(refreshed);
    } catch (e) {
      setStatus(`保存失败：${e instanceof Error ? e.message : String(e)}`);
    } finally { setSaving(false); }
  }

  function buildGroups(): GroupDef[] {
    const itemMap: Record<string, SettingItem> = {};
    allItems.forEach(i => itemMap[i.key] = i);

    return Object.entries(GROUP_KEYS).map(([title, keys]) => ({
      title,
      icon: GROUP_ICONS[title] || Settings2,
      items: keys.map(k => itemMap[k]).filter(Boolean),
    }));
  }

  if (!allItems.length) return <div className="flex-1 p-8 text-text-muted">加载中…</div>;
  const groups = buildGroups();

  return (
    <div className="flex-1 overflow-y-auto bg-bg-page p-page custom-scrollbar">
      <div className="max-w-5xl">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="font-display text-[28px] text-text-primary tracking-tight mb-1">系统参数</h1>
            <p className="font-body text-[14px] text-text-secondary">
              <span className="text-success font-semibold">绿色</span>=即时生效 · <span className="text-error font-semibold">红色</span>=重新上传生效 · 不可编辑项只读
            </p>
          </div>
          <button
            onClick={handleSave} disabled={saving}
            className="flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 font-body text-[13px] font-semibold text-white transition-all hover:bg-accent-hover disabled:opacity-50"
          >
            <Save size={15} /> {saving ? "保存中…" : "保存变更"}
          </button>
        </div>

        {status && (
          <div className={`mb-6 rounded-xl px-5 py-3 font-body text-[13px] ${status.startsWith("已") ? "bg-success-bg text-success" : "bg-error-bg text-error"}`}>
            {status}
          </div>
        )}

        <div className="space-y-10">
          {groups.map((group) => (
            <section key={group.title}>
              <h2 className="flex items-center gap-2.5 font-display text-[16px] text-text-primary tracking-tight mb-3">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-subtle">
                  <group.icon size={14} className="text-accent" />
                </div>
                {group.title}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {group.items.map((item) => (
                  <div key={item.key} className="flex items-start gap-3 rounded-xl border border-border-subtle bg-white p-3.5">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2 mb-0.5">
                        <code className="font-mono text-[11px] font-semibold text-text-primary">{item.key}</code>
                        <span className="font-body text-[12px] text-text-muted">{item.label}</span>
                      </div>
                      {item.editable ? (
                        <input
                          className="w-full rounded-lg border border-border-default bg-bg-input px-2 py-1 font-display text-[17px] font-bold text-accent outline-none focus:ring-2 focus:ring-accent/20"
                          value={editing[item.key] ?? item.value}
                          onChange={(e) => setEditing(prev => ({ ...prev, [item.key]: e.target.value }))}
                        />
                      ) : (
                        <p className="font-display text-[17px] font-bold text-accent mb-1">{item.value}</p>
                      )}
                      <p className="font-body text-[11px] text-text-muted leading-relaxed">{item.help}</p>
                    </div>
                    <span className={`shrink-0 rounded-md px-2 py-0.5 font-mono text-[9px] font-bold ${
                      item.immediate ? "bg-success-bg text-success" : "bg-error-bg text-error"
                    }`}>
                      {item.immediate ? "即时" : "需重传"}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
