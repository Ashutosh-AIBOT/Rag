"use client";

import { useState, useEffect, useRef } from "react";
import { Search, Sparkles } from "lucide-react";
import QueryPanel from "@/components/QueryPanel";
import AnswerDisplay from "@/components/AnswerDisplay";
import ChunkInspector from "@/components/ChunkInspector";
import PipelineVisualizer from "@/components/PipelineVisualizer";
import MetadataFilters from "@/components/MetadataFilters";

const API = "http://127.0.0.1:8000/api";

interface Chunk { content: string; metadata: Record<string, any>; }
interface Trace {
  original_query: string; strategy: string;
  transformed_queries?: string[]; retrieved_chunks?: Chunk[];
  reranked_chunks?: Chunk[]; answer?: string; latency_ms?: number;
}
interface Doc { id: string; filename: string; }

export default function QueryPage() {
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState("hybrid-rerank");
  const [isLoading, setIsLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [showFilters, setShowFilters] = useState(false);

  // Metadata filters
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [filterPageStart, setFilterPageStart] = useState<number | "">("");
  const [filterPageEnd, setFilterPageEnd] = useState<number | "">("");
  const [filterSection, setFilterSection] = useState("");
  const [filterTags, setFilterTags] = useState("");
  const [filterStrategy, setFilterStrategy] = useState("");
  const [k, setK] = useState(5);
  const [rerank, setRerank] = useState(false);

  const [documents, setDocuments] = useState<Doc[]>([]);
  
  // Abort controller reference for task cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetch(`${API}/documents`).then(r => r.json()).then(d => setDocuments(d.documents || [])).catch(() => {});
    
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const buildFilters = () => {
    const f: Record<string, any> = {};
    if (selectedSources.length > 0) f.source = selectedSources;
    if (filterPageStart !== "") f.page_start = filterPageStart;
    if (filterPageEnd !== "") f.page_end = filterPageEnd;
    if (filterSection) f.section = filterSection;
    if (filterTags) f.tags = filterTags;
    if (filterStrategy) f.strategy = filterStrategy;
    return Object.keys(f).length > 0 ? f : undefined;
  };

  const handleAsk = async () => {
    if (!question.trim()) return;

    // Abort any ongoing query
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setAnswer("");
    setTrace(null);
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, strategy, k, rerank, filters: buildFilters() }),
        signal: controller.signal,
      });
      const data = await res.json();
      setAnswer(data.answer || "");
      setTrace(data.trace || null);
    } catch (e: any) {
      if (e.name === "AbortError") {
        console.log("Ask request aborted.");
      } else {
        setAnswer("Error: Failed to query backend.");
      }
    } finally {
      if (abortControllerRef.current === controller) {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
    }
  };

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-lg shadow-blue-600/20">
          <Search className="h-5 w-5 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-slate-100">Advanced Query Interface</h1>
          <p className="text-xs text-slate-500">Hybrid search · Cross-encoder re-ranking · LCEL pipeline</p>
        </div>
      </div>

      {/* Query Panel */}
      <QueryPanel
        question={question} setQuestion={setQuestion}
        strategy={strategy} setStrategy={setStrategy}
        isLoading={isLoading} onAsk={handleAsk}
        showFilters={showFilters} setShowFilters={setShowFilters}
      />

      {/* Metadata Filters (collapsible) */}
      {showFilters && (
        <MetadataFilters
          documents={documents}
          selectedSources={selectedSources} setSelectedSources={setSelectedSources}
          filterPageStart={filterPageStart} setFilterPageStart={setFilterPageStart}
          filterPageEnd={filterPageEnd} setFilterPageEnd={setFilterPageEnd}
          filterSection={filterSection} setFilterSection={setFilterSection}
          filterTags={filterTags} setFilterTags={setFilterTags}
          filterStrategy={filterStrategy} setFilterStrategy={setFilterStrategy}
          k={k} setK={setK}
          rerank={rerank} setRerank={setRerank}
        />
      )}

      {/* Answer + Chunks side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7">
          <AnswerDisplay answer={answer} isLoading={isLoading} chunks={trace?.retrieved_chunks} />
        </div>
        <div className="lg:col-span-5">
          <ChunkInspector
            retrievedChunks={trace?.retrieved_chunks || []}
            rerankedChunks={trace?.reranked_chunks || []}
          />
        </div>
      </div>

      {/* Pipeline Visualizer */}
      <PipelineVisualizer trace={trace} />
    </div>
  );
}
