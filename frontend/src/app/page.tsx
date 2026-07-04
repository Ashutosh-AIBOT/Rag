"use client";

import { useState, useEffect, useRef } from "react";

const API_URL = "http://127.0.0.1:8000/api";

interface Document {
  id: string;
  filename: string;
  file_type: string;
  chunk_count: number;
  upload_date: string;
  status: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<"chat" | "documents">("chat");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  };

  const toggleDocSelection = (id: string) => {
    setSelectedDocs((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  const selectAllDocs = () => {
    setSelectedDocs(documents.map((d) => d.id));
  };

  const clearSelection = () => {
    setSelectedDocs([]);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      fetchDocuments();
    } catch (err) {
      console.error("Upload failed");
    }
    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMessage,
          document_ids: selectedDocs.length > 0 ? selectedDocs : undefined,
        }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error: Failed to get answer" }]);
    }
    setLoading(false);
  };

  const handleDelete = async (id: string) => {
    try {
      await fetch(`${API_URL}/documents/${id}`, { method: "DELETE" });
      setSelectedDocs((prev) => prev.filter((d) => d !== id));
      fetchDocuments();
    } catch (err) {
      console.error("Delete failed");
    }
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 bg-[#171717] flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-lg font-semibold">RAG Platform</h1>
        </div>
        <nav className="flex-1 p-2">
          <button
            onClick={() => setActiveTab("chat")}
            className={`w-full text-left px-3 py-2 rounded-lg mb-1 ${activeTab === "chat" ? "bg-gray-700" : "hover:bg-gray-800"}`}
          >
            💬 Chat
          </button>
          <button
            onClick={() => setActiveTab("documents")}
            className={`w-full text-left px-3 py-2 rounded-lg ${activeTab === "documents" ? "bg-gray-700" : "hover:bg-gray-800"}`}
          >
            📄 Documents ({documents.length})
          </button>
        </nav>
        <div className="p-4 border-t border-gray-700">
          <div className="text-xs text-gray-500">
            Status: <span className="text-green-400">Connected</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {activeTab === "chat" ? (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto scrollbar-hide p-4">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <h2 className="text-2xl font-semibold mb-2">RAG Platform</h2>
                    <p className="text-gray-400">Ask questions about your documents</p>
                  </div>
                </div>
              ) : (
                <div className="max-w-3xl mx-auto space-y-4">
                  {messages.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === "user" ? "bg-blue-600" : "bg-[#2f2f2f]"}`}>
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    </div>
                  ))}
                  {loading && (
                    <div className="flex justify-start">
                      <div className="bg-[#2f2f2f] rounded-2xl px-4 py-3">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-gray-700">
              <div className="max-w-3xl mx-auto">
                <div className="flex items-center bg-[#2f2f2f] rounded-2xl px-4 py-2">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                    placeholder="Ask a question..."
                    className="flex-1 bg-transparent outline-none text-white placeholder-gray-500"
                    disabled={loading}
                  />
                  <button
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                    className="ml-2 p-2 rounded-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </>
        ) : (
          /* Documents Tab */
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-3xl mx-auto">
              {/* Header */}
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Documents</h2>
                <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-sm font-medium">
                  {uploading ? "Uploading..." : "Upload File"}
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleUpload}
                    className="hidden"
                    accept=".pdf,.txt,.docx,.md"
                  />
                </label>
              </div>

              {/* Selection Controls */}
              {documents.length > 0 && (
                <div className="flex items-center gap-4 mb-4 p-3 bg-[#2f2f2f] rounded-lg">
                  <button
                    onClick={selectAllDocs}
                    className="text-sm text-blue-400 hover:text-blue-300"
                  >
                    Select All
                  </button>
                  <button
                    onClick={clearSelection}
                    className="text-sm text-gray-400 hover:text-gray-300"
                  >
                    Clear
                  </button>
                  <span className="text-sm text-gray-500">
                    {selectedDocs.length} selected
                  </span>
                </div>
              )}

              {/* Documents List */}
              {documents.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <p>No documents uploaded yet</p>
                  <p className="text-sm mt-2">Upload PDF, TXT, DOCX, or MD files</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {documents.map((doc) => (
                    <div
                      key={doc.id}
                      className={`flex items-center justify-between bg-[#2f2f2f] rounded-xl p-4 hover:bg-[#3a3a3a] transition cursor-pointer ${
                        selectedDocs.includes(doc.id) ? "ring-2 ring-blue-500" : ""
                      }`}
                      onClick={() => toggleDocSelection(doc.id)}
                    >
                      <div className="flex items-center space-x-3">
                        <div
                          className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                            selectedDocs.includes(doc.id)
                              ? "bg-blue-600 border-blue-600"
                              : "border-gray-500"
                          }`}
                        >
                          {selectedDocs.includes(doc.id) && (
                            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                        <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center text-sm font-bold">
                          {doc.file_type.includes("pdf")
                            ? "PDF"
                            : doc.file_type.includes("text")
                            ? "TXT"
                            : "DOC"}
                        </div>
                        <div>
                          <p className="font-medium">{doc.filename}</p>
                          <p className="text-sm text-gray-400">
                            {doc.chunk_count} chunks • {doc.file_type}
                          </p>
                          <p className="text-xs text-gray-500">
                            Status: {doc.status === "completed" ? "✅ Ready" : doc.status === "processing" ? "⏳ Processing..." : doc.status === "failed" ? "❌ Failed" : "📋 Pending"}
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(doc.id);
                        }}
                        className="text-gray-400 hover:text-red-400 p-2"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
