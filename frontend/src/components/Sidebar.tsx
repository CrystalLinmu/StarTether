import { FileText, MessageSquare, MoreHorizontal, Pin, Plus, RefreshCw, Settings, Trash2, Upload, Pencil, PinOff } from "lucide-react";
import React, { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import type { DocumentItem, FolderItem, SessionItem } from "../api";
import FolderTree from "./FolderTree";

type SidebarProps = {
  sessions: SessionItem[];
  documents: DocumentItem[];
  folders: FolderItem[];
  activeSessionId: string | null;
  selectedFolderId: string | null;
  pinnedSessions: string[];
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => void;
  onSelectFolder: (folderId: string) => void;
  onOpenUpload: () => void;
  onDeleteDocument: (documentName: string) => void;
  onRefresh: () => void;
  onCreateFolder: (name: string, parentId?: string) => Promise<void>;
  onDeleteFolder: (folderId: string) => Promise<void>;
  onRenameFolder: (folderId: string, name: string) => Promise<void>;
  onDeleteSession: (sessionId: string) => Promise<void>;
  onRenameSession: (sessionId: string, title: string) => Promise<void>;
  onTogglePinSession: (sessionId: string) => void;
};

export default function Sidebar({
  sessions, documents, folders, activeSessionId, selectedFolderId, pinnedSessions,
  onNewSession, onSelectSession, onSelectFolder, onOpenUpload,
  onDeleteDocument, onRefresh,
  onCreateFolder, onDeleteFolder, onRenameFolder,
  onDeleteSession, onRenameSession, onTogglePinSession,
}: SidebarProps) {
  return (
    <nav className="fixed left-0 top-0 z-40 flex h-full w-60 flex-col border-r border-border-subtle bg-white">
      {/* Brand */}
      <div className="px-5 pt-6 pb-4 shrink-0">
        <h1 className="font-display text-xl text-text-primary leading-none tracking-tight">
          RAG Nexus
        </h1>
        <p className="mt-1 font-mono text-[10px] text-text-muted uppercase tracking-[0.2em]">
          知识引擎
        </p>
      </div>

      {/* New Session */}
      <div className="px-3 pb-2 shrink-0">
        <button
          onClick={onNewSession}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2 font-body text-[13px] font-semibold text-white transition-all duration-200 hover:bg-accent-hover active:scale-[0.98]"
        >
          <Plus size={15} strokeWidth={2.5} />
          新会话
        </button>
      </div>

      {/* ── Scrollable: Sessions only ── */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-3 pt-1 pb-1 min-h-0">
        <div className="mb-1.5 flex items-center justify-between px-1.5">
          <span className="font-mono text-[10px] font-semibold text-text-muted uppercase tracking-[0.15em]">
            对话历史
          </span>
          <button onClick={onRefresh} className="rounded p-1 text-text-muted transition-colors hover:text-accent">
            <RefreshCw size={12} />
          </button>
        </div>

        {sessions.length === 0 ? (
          <p className="py-8 text-center font-mono text-[11px] text-text-muted">暂无对话</p>
        ) : (
          <div className="space-y-0.5">
            {/* Pinned first */}
            {sessions
              .filter((s) => pinnedSessions.includes(s.session_id))
              .map((s) => (
                <SessionRow
                  key={s.session_id}
                  session={s}
                  isActive={activeSessionId === s.session_id}
                  isPinned={true}
                  onSelect={() => onSelectSession(s.session_id)}
                  onDelete={() => onDeleteSession(s.session_id)}
                  onRename={(title) => onRenameSession(s.session_id, title)}
                  onTogglePin={() => onTogglePinSession(s.session_id)}
                />
              ))}
            {/* Then unpinned */}
            {sessions
              .filter((s) => !pinnedSessions.includes(s.session_id))
              .map((s) => (
                <SessionRow
                  key={s.session_id}
                  session={s}
                  isActive={activeSessionId === s.session_id}
                  isPinned={false}
                  onSelect={() => onSelectSession(s.session_id)}
                  onDelete={() => onDeleteSession(s.session_id)}
                  onRename={(title) => onRenameSession(s.session_id, title)}
                  onTogglePin={() => onTogglePinSession(s.session_id)}
                />
              ))}
          </div>
        )}
      </div>

      {/* ── Fixed: Document Center ── */}
      <div className="shrink-0 border-t border-border-subtle px-3 pt-3 pb-2">
        <FolderTree
          folders={folders}
          selectedFolderId={selectedFolderId}
          onSelect={onSelectFolder}
          onCreateFolder={onCreateFolder}
          onDeleteFolder={onDeleteFolder}
          onRenameFolder={onRenameFolder}
        />

        {/* Upload */}
        <div className="mt-2">
          <button
            className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border-default bg-bg-surface px-3 py-2 font-body text-[12px] text-text-secondary transition-all duration-200 hover:border-accent/30 hover:text-accent"
            onClick={onOpenUpload}
          >
            <Upload size={13} />
            {selectedFolderId ? "上传至当前文件夹" : "上传文档"}
          </button>
        </div>

        {/* Documents */}
        <div className="mt-2.5 max-h-[30vh] overflow-y-auto custom-scrollbar">
          <span className="mb-1.5 block px-1.5 font-mono text-[10px] font-semibold text-text-muted uppercase tracking-[0.15em]">
            {selectedFolderId ? "文件夹文档" : "全部文档"}
          </span>
          {documents.length === 0 ? (
            <p className="py-4 text-center font-mono text-[11px] text-text-muted/50">暂无文档</p>
          ) : (
            <div className="space-y-0.5">
              {documents.map((doc) => (
                <div key={doc.document_name}
                  className="group flex items-center gap-2 rounded-md px-1.5 py-1.5 transition-all hover:bg-bg-surface">
                  <FileText size={14} className="shrink-0 text-text-muted" />
                  <div className="min-w-0 flex-1">
                    <Link to={`/doc/${encodeURIComponent(doc.document_name)}`} className="truncate font-body text-[12px] font-medium text-text-primary hover:text-accent transition-colors">{doc.document_name}</Link>
                    <p className="font-mono text-[10px] text-text-muted">L3 {doc.leaf_chunk_count}/{doc.chunk_count}</p>
                  </div>
                  <button onClick={() => onDeleteDocument(doc.document_name)}
                    className="rounded p-1 text-text-muted opacity-0 transition-all hover:bg-error-bg hover:text-error group-hover:opacity-100">
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Bottom */}
      <div className="shrink-0 border-t border-border-subtle px-3 py-2.5">
        <NavLink to="/settings"
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-lg px-3 py-2 font-body text-[13px] transition-all duration-150 ${
              isActive ? "bg-accent-subtle text-accent font-semibold" : "text-text-secondary hover:bg-bg-surface"
            }`}>
          <Settings size={15} /> 系统设置
        </NavLink>
      </div>
    </nav>
  );
}

/* ── Session Row ── */
function SessionRow({
  session, isActive, isPinned, onSelect, onDelete, onRename, onTogglePin,
}: {
  session: SessionItem;
  isActive: boolean;
  isPinned: boolean;
  onSelect: () => void;
  onDelete: () => Promise<void>;
  onRename: (title: string) => Promise<void>;
  onTogglePin: () => void;
  key?: React.Key;
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div
      className={`group relative flex w-full items-start gap-2 rounded-lg px-2.5 py-2 text-left transition-all duration-150 cursor-pointer ${
        isActive
          ? "bg-accent-subtle text-accent font-semibold"
          : "text-text-secondary hover:bg-bg-surface hover:text-text-primary"
      }`}
      onClick={onSelect}
    >
      {isPinned && <Pin size={10} className="mt-0.5 shrink-0 text-amber-400" />}
      {!isPinned && <MessageSquare size={14} className="mt-0.5 shrink-0 opacity-40" />}
      <div className="min-w-0 flex-1">
        <p className="truncate font-body text-[13px] leading-snug">{session.title || "新会话"}</p>
        <p className="font-mono text-[10px] text-text-muted">{session.message_count} 条</p>
      </div>

      {/* Three-dot menu */}
      <button
        className="shrink-0 rounded p-0.5 text-text-muted opacity-0 group-hover:opacity-100 hover:bg-stone-200 transition-opacity"
        onClick={(e) => { e.stopPropagation(); setMenuOpen(!menuOpen); }}
      >
        <MoreHorizontal size={13} />
      </button>

      {menuOpen && (
        <>
          <div className="fixed inset-0 z-30" onClick={(e) => { e.stopPropagation(); setMenuOpen(false); }} />
          <div className="absolute right-0 top-7 z-40 w-32 rounded-lg border border-border-default bg-white shadow-lg py-1 text-[12px]">
            <button
              className="flex w-full items-center gap-2 px-3 py-1.5 hover:bg-bg-surface text-text-secondary"
              onClick={async (e) => {
                e.stopPropagation(); setMenuOpen(false);
                const name = prompt("重命名为：", session.title || "");
                if (!name?.trim()) return;
                try { await onRename(name.trim()); } catch (err) { alert(`重命名失败：${err}`); }
              }}
            >
              <Pencil size={12} /> 重命名
            </button>
            <button
              className="flex w-full items-center gap-2 px-3 py-1.5 hover:bg-bg-surface text-text-secondary"
              onClick={(e) => { e.stopPropagation(); setMenuOpen(false); onTogglePin(); }}
            >
              {isPinned ? <><PinOff size={12} /> 取消置顶</> : <><Pin size={12} /> 置顶</>}
            </button>
            <button
              className="flex w-full items-center gap-2 px-3 py-1.5 hover:bg-error-bg text-error"
              onClick={async (e) => {
                e.stopPropagation(); setMenuOpen(false);
                if (!confirm(`确定删除会话"${session.title || '新会话'}"？此操作不可恢复。`)) return;
                try { await onDelete(); } catch (err) { alert(`删除失败：${err}`); }
              }}
            >
              <Trash2 size={12} /> 删除
            </button>
          </div>
        </>
      )}
    </div>
  );
}
