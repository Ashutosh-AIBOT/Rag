"use client";

import { useState, useEffect } from "react";
import {
  Upload,
  Tag,
  Calendar,
  FileText,
  Layers,
  Trash2,
  AlertCircle,
  CheckCircle2,
  FileDown,
  Info,
  Clock,
  ExternalLink
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000/api";

interface Document {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  total_pages: number;
  upload_date: string;
  tags: string;
  chunk_count: number;
  status: string;
}

interface ChunkPreview {
  content: string;
  metadata: {
    doc_id: string;
    page: number;
    strategy: string;
    section?: string;
    [key: string]: any;
  };
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [tagsInput, setTagsInput] = useState("");
  
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");

  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [activeDocChunks, setActiveDocChunks] = useState<ChunkPreview[]>([]);
  const [activeStrategyFilter, setActiveStrategyFilter] = useState("recursive");
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.error("Failed to load documents", e);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      setUploadError("");
      setUploadMessage("");
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadMessage("Uploading and starting ingestion pipeline...");
    setUploadError("");

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("tags", tagsInput.trim());

    try {
      const res = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Ingestion failed with status: ${res.status}`);
      }

      const data = await res.json();
      setUploadMessage(`Successfully ingested! Chunks: ${data.chunk_count || 0}`);
      setSelectedFile(null);
      setTagsInput("");
      
      // Clear file input element
      const fileInput = document.getElementById("file-upload") as HTMLInputElement;
      if (fileInput) fileInput.value = "";

      // Refresh list
      fetchDocuments();
    } catch (e: any) {
      console.error(e);
      setUploadError(e.message || "An error occurred during ingestion.");
      setUploadMessage("");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if (!confirm("Are you sure you want to delete this document? This will remove all chunks and vector embeddings.")) {
      return;
    }

    try {
      const res = await fetch(`${API_URL}/documents/${docId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (activeDocId === docId) {
          setActiveDocId(null);
          setActiveDocChunks([]);
        }
        fetchDocuments();
      }
    } catch (e) {
      console.error("Failed to delete document", e);
    }
  };

  const handleDocClick = async (docId: string) => {
    setActiveDocId(docId);
    setIsLoadingChunks(true);
    try {
      // In the backend, we can query Chroma chunks for a document!
      // Wait, let's call the debug chunk search with source filename as filter
      // to retrieve all chunks for this document!
      const doc = documents.find(d => d.id === docId);
      if (!doc) return;

      const res = await fetch(
        `${API_URL}/chunks/search?q=*&k=100&source=${encodeURIComponent(doc.filename)}`
      );
      if (res.ok) {
        const data = await res.json();
        setActiveDocChunks(data.retrieved_chunks || []);
      }
    } catch (e) {
      console.error("Failed to load document chunks", e);
    } finally {
      setIsLoadingChunks(false);
    }
  };

  const getFormatSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const filteredChunks = activeDocChunks.filter(
    c => c.metadata.strategy === activeStrategyFilter
  );

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 flex flex-col gap-8">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-100">Document Management</h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload and manage corpus documents. Each file runs through 4 concurrent chunking strategies.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Side: Upload & List */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Upload Form Card */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-sm shadow-xl">
            <h3 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Upload className="h-5 w-5 text-blue-500" />
              <span>Ingest New Document</span>
            </h3>

            <form onSubmit={handleUpload} className="space-y-4">
              <div className="border-2 border-dashed border-slate-800 rounded-xl p-6 text-center hover:border-slate-700 transition-all bg-slate-950/20">
                <input
                  type="file"
                  id="file-upload"
                  accept=".pdf,.txt,.docx,.md"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center justify-center gap-2"
                >
                  <FileText className="h-8 w-8 text-slate-500" />
                  <span className="text-sm font-medium text-slate-300">
                    {selectedFile ? selectedFile.name : "Select PDF, TXT, DOCX, or Markdown"}
                  </span>
                  <span className="text-xs text-slate-500">
                    {selectedFile ? getFormatSize(selectedFile.size) : "Max file size: 20MB"}
                  </span>
                </label>
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
                  Comma-Separated Tags
                </label>
                <div className="relative">
                  <Tag className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                  <input
                    type="text"
                    placeholder="e.g. attention, transformer, deep-learning"
                    value={tagsInput}
                    onChange={(e) => setTagsInput(e.target.value)}
                    className="w-full rounded-lg border border-slate-800 bg-slate-950 py-2 pl-10 pr-4 text-sm text-slate-200 placeholder-slate-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={isUploading || !selectedFile}
                className="w-full flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-blue-600/20 transition-all hover:bg-blue-500 hover:shadow-blue-500/30 disabled:bg-slate-800 disabled:text-slate-500 disabled:shadow-none"
              >
                {isUploading ? (
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-white" />
                ) : (
                  <span>Upload & Ingest</span>
                )}
              </button>
            </form>

            {uploadMessage && (
              <div className="mt-4 flex items-center gap-2.5 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3.5 text-emerald-400 text-sm">
                <CheckCircle2 className="h-4.5 w-4.5 flex-shrink-0" />
                <p>{uploadMessage}</p>
              </div>
            )}

            {uploadError && (
              <div className="mt-4 flex items-center gap-2.5 rounded-lg border border-rose-500/20 bg-rose-500/5 p-3.5 text-rose-400 text-sm">
                <AlertCircle className="h-4.5 w-4.5 flex-shrink-0" />
                <p>{uploadError}</p>
              </div>
            )}
          </div>

          {/* List of Documents Card */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
            <h3 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Layers className="h-5 w-5 text-indigo-400" />
              <span>Ingested Corpus</span>
            </h3>

            <div className="space-y-3">
              {documents.length > 0 ? (
                documents.map((doc) => (
                  <div
                    key={doc.id}
                    onClick={() => handleDocClick(doc.id)}
                    className={`group rounded-xl border p-4 cursor-pointer transition-all flex justify-between items-center ${
                      activeDocId === doc.id
                        ? "border-blue-500 bg-blue-500/[0.01]"
                        : "border-slate-800 bg-slate-950/20 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <FileText className="h-4 w-4 text-blue-400" />
                        <span className="text-sm font-semibold text-slate-200 truncate">{doc.filename}</span>
                      </div>
                      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                        <span className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          <span>{new Date(doc.upload_date).toLocaleDateString()}</span>
                        </span>
                        <span>{getFormatSize(doc.file_size)}</span>
                        <span>{doc.chunk_count} Chunks</span>
                      </div>
                      {doc.tags && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {doc.tags.split(",").map((t: string, i: number) => (
                            <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/60 text-slate-400">
                              {t.trim()}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(doc.id);
                      }}
                      className="ml-4 p-2 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-all opacity-0 group-hover:opacity-100 focus:opacity-100"
                      title="Delete document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))
              ) : (
                <div className="text-center py-12 border border-dashed border-slate-800 rounded-xl text-slate-500 text-sm">
                  No documents in corpus yet. Ingest a document above!
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Side: Chunking Strategy & Previews */}
        <div className="lg:col-span-5">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md min-h-[480px]">
            <h3 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
              <Layers className="h-5 w-5 text-indigo-400" />
              <span>Chunk Preview Inspector</span>
            </h3>

            {activeDocId ? (
              <div className="space-y-4">
                {/* Strategy selector for previews */}
                <div className="flex flex-wrap gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-850">
                  {["recursive", "parent_child", "section", "semantic"].map((strat) => (
                    <button
                      key={strat}
                      onClick={() => setActiveStrategyFilter(strat)}
                      className={`flex-1 text-center py-1.5 px-2.5 rounded-md text-xs font-semibold uppercase tracking-wide transition-all ${
                        activeStrategyFilter === strat
                          ? "bg-blue-600 text-white shadow"
                          : "text-slate-500 hover:text-slate-300"
                      }`}
                    >
                      {strat.replace("_", "-")}
                    </button>
                  ))}
                </div>

                <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
                  {isLoadingChunks ? (
                    <div className="flex flex-col items-center justify-center py-16 gap-3">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-400 border-t-white" />
                      <span className="text-xs text-slate-500">Querying collection chunks...</span>
                    </div>
                  ) : filteredChunks.length > 0 ? (
                    filteredChunks.map((c, i) => (
                      <div key={i} className="rounded-lg border border-slate-800 bg-slate-950/40 p-4 space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] font-bold text-slate-500">CHUNK #{i + 1}</span>
                          <span className="text-[10px] text-slate-500">Page {c.metadata.page + 1}</span>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed font-sans">{c.content}</p>
                        {c.metadata.section && (
                          <div className="text-[10px] text-slate-500 border-t border-slate-900 pt-1.5">
                            Section: <span className="italic">{c.metadata.section}</span>
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-12 text-slate-500 text-xs italic">
                      No chunks created for strategy '{activeStrategyFilter}' in this document.
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-24 text-slate-500 text-sm">
                <Info className="h-8 w-8 text-slate-600 mb-2" />
                <span>Select an ingested document to inspect its generated chunks.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
