import { useCallback, useEffect, useRef, useState } from "react";
import { BrowserRouter as Router, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import {
  createFolder,
  deleteDocument,
  deleteFolder,
  deleteSession,
  listDocuments,
  listDocumentsInFolder,
  listFolderTree,
  listSessions,
  renameSession,
  updateFolder,
  uploadDocument,
  uploadDocumentToFolder,
  type DocumentItem,
  type FolderItem,
  type SessionItem,
  type StreamStep,
  type UploadDone,
} from "./api";
import Layout from "./components/Layout";
import UploadModal, { type UploadStats } from "./components/UploadModal";
import Chat from "./pages/Chat";
import DocDetail from "./pages/DocDetail";
import Documents from "./pages/Documents";
import Settings from "./pages/Settings";
import TraceView from "./pages/TraceView";

function AppShell() {
  const navigate = useNavigate();
  const location = useLocation();
  const isDocPage = location.pathname.startsWith("/doc/");
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [folders, setFolders] = useState<FolderItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [uploadSteps, setUploadSteps] = useState<StreamStep[]>([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadUploading, setUploadUploading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const [uploadStats, setUploadStats] = useState<UploadStats | null>(null);
  const [uploadGraphStats, setUploadGraphStats] = useState<{ entities?: number; relationships?: number } | null>(null);
  const [pinnedSessions, setPinnedSessions] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem("pinnedSessions") || "[]"); } catch { return []; }
  });

  const refresh = useCallback(async () => {
    const [sessionRows, documentRows, folderRows] = await Promise.all([
      listSessions(),
      selectedFolderId ? listDocumentsInFolder(selectedFolderId) : listDocuments(),
      listFolderTree(),
    ]);
    setSessions(sessionRows);
    setDocuments(documentRows);
    setFolders(folderRows);
    return sessionRows;
  }, [selectedFolderId]);

  // 启动时加载数据，不自动选中会话——初始界面为新会话
  useEffect(() => {
    refresh().catch((error) => setUploadStatus(`加载失败：${error.message}`));
  }, [refresh]);

  async function handleUpload(file: File, folderId: string | null) {
    setUploadOpen(true); setUploadUploading(true); setUploadSteps([]); setUploadStats(null); setUploadGraphStats(null);
    setUploadStatus(`准备上传：${file.name}`);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const result: UploadDone | null = folderId
        ? await uploadDocumentToFolder(folderId, file, (step) => {
            setUploadSteps((prev) => [...prev, step]);
          }, controller.signal)
        : await uploadDocument(file, (step) => {
            setUploadSteps((prev) => [...prev, step]);
          }, controller.signal);
      setUploadStats({
        ...(result?.stats || {}),
        filename: result?.filename || file.name,
        folder_id: result?.folder_id || folderId || "",
      } as UploadStats);
      setUploadGraphStats((result as any)?.graph_stats || null);
      setUploadStatus("入库完成");
      await refresh();
    } catch (error) {
      setUploadSteps((prev) => [...prev, { code: "error", label: String(error) } as StreamStep]);
      setUploadStatus(`上传失败：${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setUploadUploading(false);
    }
  }

  function handleUploadToFolder(folderId: string, file: File) {
    setSelectedFolderId(folderId);
    handleUpload(file, folderId);
  }

  async function handleDeleteDocument(documentName: string) {
    await deleteDocument(documentName);
    await refresh();
  }

  function handleNewSession() {
    setActiveSessionId(null);
    navigate("/");
  }

  function handleSelectSession(sessionId: string) {
    setActiveSessionId(sessionId);
    navigate("/");
  }

  function handleSelectFolder(folderId: string) {
    setSelectedFolderId(folderId);
  }

  async function handleCreateFolder(name: string, parentId?: string) {
    try {
      await createFolder(name, parentId);
      await refresh();
    } catch (err) {
      alert(`创建文件夹失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleDeleteFolder(folderId: string) {
    try {
      await deleteFolder(folderId);
    } catch (err) {
      alert(`删除文件夹失败：${err instanceof Error ? err.message : String(err)}`);
    }
    if (selectedFolderId === folderId) setSelectedFolderId(null);
    await refresh();
  }

  async function handleRenameFolder(folderId: string, name: string) {
    try {
      await updateFolder(folderId, name);
      await refresh();
    } catch (err) {
      alert(`重命名失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleDeleteSession(sessionId: string) {
    try {
      await deleteSession(sessionId);
      if (activeSessionId === sessionId) setActiveSessionId(null);
      await refresh();
    } catch (err) {
      alert(`删除会话失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function handleRenameSession(sessionId: string, title: string) {
    try {
      await renameSession(sessionId, title);
      await refresh();
    } catch (err) {
      alert(`重命名失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function handleTogglePinSession(sessionId: string) {
    setPinnedSessions((prev) => {
      const next = prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [...prev, sessionId];
      localStorage.setItem("pinnedSessions", JSON.stringify(next));
      return next;
    });
  }

  // 文档详情页：全屏，不走 Layout
  if (isDocPage) {
    return (
      <Routes>
        <Route path="/doc/:name" element={<DocDetail />} />
      </Routes>
    );
  }

  return (
    <>
      <UploadModal
        folders={folders}
        selectedFolderId={selectedFolderId}
        open={uploadOpen}
        steps={uploadSteps}
        status={uploadStatus}
        stats={uploadStats}
        graphStats={uploadGraphStats}
        uploading={uploadUploading}
        onClose={() => { abortRef.current?.abort(); setUploadOpen(false); setUploadUploading(false); setUploadSteps([]); setUploadStats(null); }}
        onUpload={handleUpload}
      />
      <Layout
      sessions={sessions}
      documents={documents}
      folders={folders}
      activeSessionId={activeSessionId}
      selectedFolderId={selectedFolderId}
      pinnedSessions={pinnedSessions}
      onNewSession={handleNewSession}
      onSelectSession={handleSelectSession}
      onSelectFolder={handleSelectFolder}
      onOpenUpload={() => setUploadOpen(true)}
      onDeleteDocument={handleDeleteDocument}
      onRefresh={refresh}
      onCreateFolder={handleCreateFolder}
      onDeleteFolder={handleDeleteFolder}
      onRenameFolder={handleRenameFolder}
      onDeleteSession={handleDeleteSession}
      onRenameSession={handleRenameSession}
      onTogglePinSession={handleTogglePinSession}
    >
      <Routes>
        <Route
          path="/"
          element={
            <Chat
              sessionId={activeSessionId}
              folderId={selectedFolderId}
              onSessionChange={setActiveSessionId}
              onDataChanged={refresh}
            />
          }
        />
        <Route path="/settings" element={<Settings />} />
        <Route path="/trace" element={<TraceView sessionId={activeSessionId} />} />
        <Route path="/documents" element={<Documents documents={documents} />} />
      </Routes>
    </Layout>
    </>
  );
}

export default function App() {
  return (
    <Router>
      <AppShell />
    </Router>
  );
}
