import React, { useState } from "react";
import { Folder, FolderOpen, FolderPlus, ChevronRight, ChevronDown, MoreVertical, Trash2, Pencil } from "lucide-react";
import type { FolderItem } from "../api";

interface FolderTreeProps {
  folders: FolderItem[];
  selectedFolderId: string | null;
  onSelect: (folderId: string) => void;
  onCreateFolder: (name: string, parentId?: string) => Promise<void>;
  onDeleteFolder: (folderId: string) => Promise<void>;
  onRenameFolder: (folderId: string, name: string) => Promise<void>;
}

export default function FolderTree({
  folders,
  selectedFolderId,
  onSelect,
  onCreateFolder,
  onDeleteFolder,
  onRenameFolder,
}: FolderTreeProps) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between px-2 py-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
          文档中心
        </span>
        <button
          className="p-0.5 rounded hover:bg-stone-200 dark:hover:bg-stone-700 text-stone-500"
          title="新建文件夹"
          onClick={async (e) => {
            e.stopPropagation();
            const name = prompt("文件夹名称：");
            if (!name?.trim()) return;
            try {
              await onCreateFolder(name.trim());
            } catch (err) {
              alert(`创建失败：${err instanceof Error ? err.message : String(err)}`);
            }
          }}
        >
          <FolderPlus size={14} />
        </button>
      </div>
      {folders.length === 0 ? (
        <p className="px-2 text-xs text-stone-400 italic">暂无文件夹</p>
      ) : (
        folders.map((f) => (
          <FolderTreeNode
            key={f.folder_id}
            folder={f}
            depth={0}
            selectedFolderId={selectedFolderId}
            onSelect={onSelect}
            onCreateFolder={onCreateFolder}
            onDeleteFolder={onDeleteFolder}
            onRenameFolder={onRenameFolder}
          />
        ))
      )}
    </div>
  );
}

function FolderTreeNode({
  folder,
  depth,
  selectedFolderId,
  onSelect,
  onCreateFolder,
  onDeleteFolder,
  onRenameFolder,
}: {
  folder: FolderItem;
  depth: number;
  selectedFolderId: string | null;
  onSelect: (id: string) => void;
  onCreateFolder: (name: string, parentId?: string) => Promise<void>;
  onDeleteFolder: (id: string) => Promise<void>;
  onRenameFolder: (id: string, name: string) => Promise<void>;
  key?: React.Key;
}) {
  const [expanded, setExpanded] = useState(depth < 1);
  const [menuOpen, setMenuOpen] = useState(false);
  const hasChildren = folder.children && folder.children.length > 0;
  const isSelected = selectedFolderId === folder.folder_id;

  return (
    <div>
      <div
        className={`flex items-center gap-1 px-2 py-1 cursor-pointer rounded text-sm group ${
          isSelected
            ? "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300"
            : "hover:bg-stone-100 dark:hover:bg-stone-800 text-stone-700 dark:text-stone-300"
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onSelect(folder.folder_id)}
      >
        <button
          className="p-0.5 rounded hover:bg-stone-200 dark:hover:bg-stone-700 flex-shrink-0"
          onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
        >
          {hasChildren ? (
            expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />
          ) : (
            <span className="w-3" />
          )}
        </button>
        {expanded ? <FolderOpen size={14} className="text-amber-500 flex-shrink-0" />
                   : <Folder size={14} className="text-amber-500 flex-shrink-0" />}
        <span className="flex-1 truncate text-xs">{folder.folder_name}</span>
        <span className="text-[10px] text-stone-400 flex-shrink-0">
          {folder.document_count > 0 && `${folder.document_count}d`}
          {folder.entity_count > 0 && ` · ${folder.entity_count}E`}
        </span>
        <div className="relative">
          <button
            className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-stone-200 dark:hover:bg-stone-700"
            onClick={(e) => { e.stopPropagation(); setMenuOpen(!menuOpen); }}
          >
            <MoreVertical size={12} />
          </button>
          {menuOpen && (
            <div className="absolute right-0 top-5 bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-md shadow-lg z-20 py-1 min-w-[100px]">
              <button
                className="w-full text-left px-3 py-1.5 text-xs hover:bg-stone-100 dark:hover:bg-stone-700 flex items-center gap-1.5"
                onClick={async (e) => {
                  e.stopPropagation(); setMenuOpen(false);
                  const name = prompt("重命名为：", folder.folder_name);
                  if (!name?.trim()) return;
                  try { await onRenameFolder(folder.folder_id, name.trim()); }
                  catch (err) { alert(`重命名失败：${err instanceof Error ? err.message : String(err)}`); }
                }}
              >
                <Pencil size={11} /> 重命名
              </button>
              <button
                className="w-full text-left px-3 py-1.5 text-xs hover:bg-stone-100 dark:hover:bg-stone-700 flex items-center gap-1.5"
                onClick={async (e) => {
                  e.stopPropagation(); setMenuOpen(false);
                  const name = prompt("子文件夹名称：");
                  if (!name?.trim()) return;
                  try { await onCreateFolder(name.trim(), folder.folder_id); }
                  catch (err) { alert(`创建失败：${err instanceof Error ? err.message : String(err)}`); }
                }}
              >
                <FolderPlus size={11} /> 新建子文件夹
              </button>
              <button
                className="w-full text-left px-3 py-1.5 text-xs hover:bg-red-50 dark:hover:bg-red-900/30 text-red-600 flex items-center gap-1.5"
                onClick={async (e) => {
                  e.stopPropagation(); setMenuOpen(false);
                  if (!confirm(`确定删除文件夹"${folder.folder_name}"？内部文档不会被删除。`)) return;
                  try { await onDeleteFolder(folder.folder_id); }
                  catch (err) { alert(`删除失败：${err instanceof Error ? err.message : String(err)}`); }
                }}
              >
                <Trash2 size={11} /> 删除
              </button>
            </div>
          )}
        </div>
      </div>
      {expanded && hasChildren && (
        <div>
          {folder.children.map((child) => (
            <FolderTreeNode
              key={child.folder_id}
              folder={child}
              depth={depth + 1}
              selectedFolderId={selectedFolderId}
              onSelect={onSelect}
              onCreateFolder={onCreateFolder}
              onDeleteFolder={onDeleteFolder}
              onRenameFolder={onRenameFolder}
            />
          ))}
        </div>
      )}
    </div>
  );
}
