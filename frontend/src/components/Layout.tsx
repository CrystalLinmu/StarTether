import { ReactNode } from "react";
import type { DocumentItem, FolderItem, SessionItem } from "../api";
import Sidebar from "./Sidebar";

type LayoutProps = {
  children: ReactNode;
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

export default function Layout(props: LayoutProps) {
  const { children, ...sidebarProps } = props;
  return (
    <div className="flex h-screen overflow-hidden bg-bg-page">
      <Sidebar {...sidebarProps} />
      <main className="relative ml-60 flex h-full flex-1 flex-col overflow-hidden">
        {children}
      </main>
    </div>
  );
}
