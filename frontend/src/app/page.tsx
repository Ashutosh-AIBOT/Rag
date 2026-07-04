"use client";

import { useState, useRef, useEffect } from "react";

const API_URL = "http://127.0.0.1:8000/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [strategy, setStrategy] = useState("vector");
  const [showDocuments, setShowDocuments] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      alert(data.message);
      fetchDocuments();
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Upload failed");
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          k: 5,
          strategy: strategy,
        }),
      });
      const data = await res.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Query failed:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, an error occurred." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await fetch(`${API_URL}/documents/${docId}`, { method: "DELETE" });
      fetchDocuments();
    } catch (error) {
      console.error("Delete failed:", error);
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <div
        className={`${
          showDocuments ? "w-80" : "w-0"
        } transition-all duration-300 bg-gray-800 overflow-hidden`}
      >
        <div className="p-4">
          <h2 className="text-lg font-bold mb-4">Documents</h2>
          <label className="block mb-4">
            <span className="sr-only">Choose file</span>
            <input
              type="file"
              onChange={handleUpload}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
            />
          </label>
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="p-3 bg-gray-700 rounded-lg flex justify-between items-center"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{doc.filename}</p>
                  <p className="text-xs text-gray-400">
                    {doc.chunk_count} chunks •{" "}
                    <span
                      className={
                        doc.status === "completed"
                          ? "text-green-400"
                          : doc.status === "processing"
                          ? "text-yellow-400"
                          : "text-red-400"
                      }
                    >
                      {doc.status}
                    </span>
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(doc.id)}
                  className="ml-2 text-red-400 hover:text-red-300"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex items-center gap-4">
          <button
            onClick={() => setShowDocuments(!showDocuments)}
            className="p-2 hover:bg-gray-700 rounded-lg"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
          <h1 className="text-lg font-bold">RAG Platform</h1>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="ml-auto bg-gray-700 rounded-lg px-3 py-1 text-sm"
          >
            <option value="vector">Vector Search</option>
            <option value="bm25">BM25 Search</option>
            <option value="hybrid">Hybrid Search</option>
            <option value="rrf">RRF Search</option>
          </select>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-3xl rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 text-white"
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-gray-600 text-xs text-gray-400">
                    Sources: {msg.sources.join(", ")}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-700 rounded-2xl px-4 py-3">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-700">
          <div className="max-w-3xl mx-auto flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask a question..."
              className="flex-1 bg-gray-700 rounded-full px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-full px-6 py-3 font-medium transition-colors"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
