import { Folder } from "lucide-react";
import type { FolderItem } from "../api";

interface FolderPickerProps {
  folders: FolderItem[];
  selectedFolderId: string | null;
  onChange: (folderId: string | null) => void;
  placeholder?: string;
}

export default function FolderPicker({
  folders,
  selectedFolderId,
  onChange,
  placeholder = "搜索全部文件夹",
}: FolderPickerProps) {
  const selected = folders.find((f) => f.folder_id === selectedFolderId);

  return (
    <div className="flex items-center gap-1.5">
      <Folder size={14} className="text-stone-400 flex-shrink-0" />
      <select
        className="text-xs bg-stone-100 dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded px-2 py-1 text-stone-700 dark:text-stone-300 focus:outline-none focus:ring-1 focus:ring-indigo-400 cursor-pointer max-w-[160px] truncate"
        value={selectedFolderId || ""}
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">{placeholder}</option>
        {folders.map((f) => (
          <option key={f.folder_id} value={f.folder_id}>
            {f.folder_name}
            {f.document_count > 0 ? ` (${f.document_count})` : ""}
          </option>
        ))}
      </select>
      {selected && (
        <span className="text-[10px] text-stone-400">
          {selected.document_count} 文档{selected.entity_count > 0 ? ` · ${selected.entity_count} 实体` : ""}
        </span>
      )}
    </div>
  );
}
