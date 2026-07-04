"use client";

import { useState, useEffect } from "react";
import {
  Search,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  Cpu,
  Clock,
  Layers,
  Sparkles,
  FileText,
  Bookmark,
  CheckCircle,
  AlertCircle,
  HelpCircle,
  Database,
  ArrowRight
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000/api";

interface Chunk {
  content: string;
  metadata: {
    doc_id?: string;
    page?: number;
    strategy?: string;
    source?: string;
    section?: string;
    tags?: string;
    original_rank?: number;
    reranked_position?: number;
    relevance_score?: number;
    bm25_score?: number;
    rrf_score?: number;
    [key: string]: any;
  };
}

interface PipelineTrace {
  original_query: string;
  strategy: string;
  transformed_queries?: string[];
  retrieved_chunks?: Chunk[];
  reranked_chunks?: Chunk[];
  answer?: string;
  latency_ms?: number;
}

export default function QueryPage() {
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState("hybrid_rerank");
  const [k, setK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [rerankTopK, setRerankTopK] = useState(3);
  
  // Filters
  const [showFilters, setShowFilters] = useState(false);
  const [filterSource, setFilterSource] = useState("");
  const [filterPageStart, setFilterPageStart] = useState<number | "">("");
  const [filterPageEnd, setFilterPageEnd] = useState<number | "">("");
  const [filterSection, setFilterSection] = useState("");
  const [filterTags, setFilterTags] = useState("");
  const [filterStrategy, setFilterStrategy] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [queryId, setQueryId] = useState<string | null>(null);
  const [pipelineTrace, setPipelineTrace] = useState<PipelineTrace | null>(null);
  const [error, setError] = useState("");
  
  // Expanded visualizer step
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  // Ingested documents list for filter dropdown
  const [documents, setDocuments] = useState<any[]>([]);

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
      console.error("Failed to load documents for filters", e);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return;

    setIsLoading(true);
    setError("");
    setStreamedAnswer("");
    setPipelineTrace(null);
    setQueryId(null);

    // Build filters dict
    const filters: any = {};
    if (filterSource) filters["source"] = filterSource;
    if (filterPageStart !== "") filters["page_start"] = Number(filterPageStart);
    if (filterPageEnd !== "") filters["page_end"] = Number(filterPageEnd);
    if (filterSection) filters["section"] = filterSection;
    if (filterTags) filters["tags"] = filterTags;
    if (filterStrategy) filters["strategy"] = filterStrategy;

    try {
      // Step 1: Perform the streaming request
      const response = await fetch(`${API_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          strategy,
          k,
          rerank,
          rerank_top_k: rerankTopK,
          filters: Object.keys(filters).length > 0 ? filters : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let answerText = "";

      if (reader) {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunkText = decoder.decode(value);
          const lines = chunkText.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data.startsWith("Error: ")) {
                setError(data.slice(7));
              } else {
                answerText += data;
                setStreamedAnswer(answerText);
              }
            }
          }
        }
      }

      // Step 2: Query is finished. Now fetch the latest query trace from history stats
      // Let's call GET /api/stats to get total queries and fetch details
      const statsRes = await fetch(`${API_URL}/stats`);
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        // Since we don't have a direct "get latest query id" from stream,
        // we can fetch the list of query history or pipeline details.
        // Wait, the backend trace router has GET /api/query/{query_id}/pipeline.
        // Let's perform a generic query GET to obtain the pipeline trace or query list!
        // To find the query_id, let's fetch list of eval results or fetch query details by searching list.
        // Better yet: we can fetch the last query ID via stats or fetch direct from SQLite database history.
        // Let's retrieve evaluation results or call search endpoint to find the trace.
        // Wait, does query.py have an endpoint to list query history? Let's check how we can fetch it.
        // Yes, the compare page lists queries, or we can fetch the last trace using a debug search.
        // Let's call /api/stats to find the total_queries.
        // Actually, we can retrieve the latest trace from database!
      }
      
      // Let's do a fallback: search for the latest query ID in database or trigger a fetch.
      // Wait, let's fetch the list of recent traces from database to populate pipeline tracer!
      // Let's call a query search or run a query to get recent query history.
      const historyRes = await fetch(`${API_URL}/stats`); // Let's check if we can query last history.
      // Wait, let's just make a simple endpoint on backend if needed, or query direct.
      // Ah! In `backend/app/routers/query.py`, there is `POST /api/query` which returns:
      // { "query_id", "answer", "sources", "trace", "latency_ms" }
      // Oh! If the user wants the full pipeline trace, we can execute the non-streaming POST `/api/query`
      // in parallel or fetch it to get the exact query_id and pipeline trace!
      // This is a brilliant strategy: we run streaming for real-time output,
      // and right after stream finishes, we make a quick POST `/api/query` (or call it directly)
      // to retrieve the precise `query_id` and complete `trace`!
      // Wait! If we call POST `/api/query`, it will run RAG again, which is redundant.
      // Is there a way to retrieve the last query from the history?
      // Yes! Let's write an endpoint `GET /api/query/recent` that returns the most recent queries.
      // Wait, does `database.py` have a list of query history?
      // Yes, `query_history` table stores all queries.
      // Let's check if we have a list query history endpoint.
      // No, we didn't add one. Let's add a `GET /api/query/recent` endpoint in `query.py` so the UI can list recent queries and show their traces!
      // This is extremely helpful and solves the issue cleanly!
    } catch (e: any) {
      console.error(e);
      setError(e.message || "An error occurred during query generation.");
    } finally {
      setIsLoading(false);
      // Let's fetch the latest query history item after 1 second so SQLite has time to commit.
      setTimeout(() => {
        fetchLatestTrace();
      }, 1000);
    }
  };

  const fetchLatestTrace = async () => {
    try {
      const res = await fetch(`${API_URL}/query/recent`);
      if (res.ok) {
        const data = await res.json();
        if (data.queries && data.queries.length > 0) {
          const latest = data.queries[0];
          setQueryId(latest.id);
          
          // Fetch the pipeline trace
          const traceRes = await fetch(`${API_URL}/query/${latest.id}/pipeline`);
          if (traceRes.ok) {
            const traceData = await traceRes.json();
            setPipelineTrace(traceData.pipeline_trace);
          }
        }
      }
    } catch (e) {
      console.error("Failed to fetch latest trace", e);
    }
  };

  // On page load, fetch recent traces if any
  useEffect(() => {
    fetchLatestTrace();
  }, []);

  const getScoreColor = (score?: number) => {
    if (score === undefined) return "text-slate-400 border-slate-800 bg-slate-900";
    if (score >= 0.7 || score > 5.0) return "text-emerald-400 border-emerald-500/20 bg-emerald-500/5";
    if (score >= 0.4 || score > 2.0) return "text-amber-400 border-amber-500/20 bg-amber-500/5";
    return "text-rose-400 border-rose-500/20 bg-rose-500/5";
  };

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 flex flex-col gap-8">
      {/* Search Header Card */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-sm shadow-xl">
        <div className="flex flex-col md:flex-row gap-4 items-center">
          <div className="relative flex-1 w-full">
            <Search className="absolute left-4 top-3.5 h-5 w-5 text-slate-400" />
            <input
              type="text"
              placeholder="Ask the LangChain RAG engine a question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 py-3.5 pl-12 pr-4 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-base"
              disabled={isLoading}
            />
          </div>
          <div className="flex gap-2 w-full md:w-auto">
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3.5 text-sm font-medium text-slate-300 focus:border-blue-500 focus:outline-none"
            >
              <option value="vector">Basic Vector</option>
              <option value="hybrid">Hybrid (BM25 + Vector)</option>
              <option value="hybrid_rerank">Hybrid + Rerank (Default)</option>
              <option value="parent_child">Parent-Child Retriever</option>
              <option value="multi_query">Multi-Query Expansion</option>
              <option value="hyde">HyDE (Hypothetical Doc)</option>
              <option value="step_back">Step-Back Prompting</option>
              <option value="decomposition">Query Decomposition</option>
            </select>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-2 rounded-xl border px-4 py-3.5 text-sm font-medium transition-all ${
                showFilters
                  ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                  : "border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200"
              }`}
            >
              <SlidersHorizontal className="h-4 w-4" />
              <span>Filters</span>
              {showFilters ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            <button
              onClick={handleAsk}
              disabled={isLoading || !question.trim()}
              className="flex-1 md:flex-none flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-blue-600/20 transition-all hover:bg-blue-500 hover:shadow-blue-500/30 disabled:bg-slate-800 disabled:text-slate-500 disabled:shadow-none"
            >
              {isLoading ? (
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-white" />
              ) : (
                <span>Ask</span>
              )}
            </button>
          </div>
        </div>

        {/* Collapsible Metadata Filters Panel */}
        {showFilters && (
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 pt-4 border-t border-slate-800/80 animate-fadeIn">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Source Document</label>
              <select
                value={filterSource}
                onChange={(e) => setFilterSource(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
              >
                <option value="">All Documents</option>
                {documents.map((doc) => (
                  <option key={doc.id} value={doc.filename}>
                    {doc.filename}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Page Start</label>
                <input
                  type="number"
                  placeholder="Min"
                  value={filterPageStart}
                  onChange={(e) => setFilterPageStart(e.target.value !== "" ? Number(e.target.value) : "")}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Page End</label>
                <input
                  type="number"
                  placeholder="Max"
                  value={filterPageEnd}
                  onChange={(e) => setFilterPageEnd(e.target.value !== "" ? Number(e.target.value) : "")}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Section Name</label>
              <input
                type="text"
                placeholder="e.g. Abstract, Introduction"
                value={filterSection}
                onChange={(e) => setFilterSection(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">User Tags</label>
              <input
                type="text"
                placeholder="e.g. key-terms, equations"
                value={filterTags}
                onChange={(e) => setFilterTags(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Chunking Strategy Filter</label>
              <select
                value={filterStrategy}
                onChange={(e) => setFilterStrategy(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
              >
                <option value="">All Strategies</option>
                <option value="recursive">Recursive Character</option>
                <option value="parent_child">Parent-Child</option>
                <option value="section">Section-Based</option>
                <option value="semantic">Semantic Splitter</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Retrieve K</label>
                <input
                  type="number"
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div className="flex items-center gap-2 pt-6">
                <input
                  type="checkbox"
                  id="rerank-check"
                  checked={rerank}
                  onChange={(e) => setRerank(e.target.checked)}
                  className="rounded border-slate-800 bg-slate-950 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="rerank-check" className="text-xs text-slate-400 select-none cursor-pointer">Cross Rerank</label>
              </div>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 text-rose-400">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Side: Answer Display */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md min-h-[300px] flex flex-col">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="h-5 w-5 text-blue-400" />
              <h2 className="text-lg font-semibold text-slate-200">Answer</h2>
              {isLoading && (
                <span className="ml-2 px-2 py-0.5 text-[10px] font-semibold text-blue-400 bg-blue-500/10 border border-blue-500/20 rounded-full animate-pulse">
                  Streaming...
                </span>
              )}
            </div>
            <div className="flex-1 text-slate-300 text-base leading-relaxed whitespace-pre-wrap font-sans">
              {streamedAnswer || (isLoading ? "Thinking..." : "Your generated answer will appear here. Ask a query above to begin.")}
            </div>
            
            {/* Citations List */}
            {pipelineTrace?.retrieved_chunks && (
              <div className="mt-6 pt-6 border-t border-slate-800/80">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Sources Cited</h3>
                <div className="flex flex-wrap gap-2">
                  {Array.from(new Set(pipelineTrace.retrieved_chunks.map(c => c.metadata.source).filter(Boolean))).map((src, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-slate-400"
                    >
                      <FileText className="h-3 w-3 text-blue-500" />
                      <span>{src}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Retrieved Chunks Inspector */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-indigo-400" />
                <h2 className="text-lg font-semibold text-slate-200 font-sans">Retrieved Chunks</h2>
              </div>
              <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-semibold text-slate-400">
                {pipelineTrace?.retrieved_chunks?.length || 0} Retrieved
              </span>
            </div>

            <div className="space-y-4 max-h-[480px] overflow-y-auto pr-1">
              {pipelineTrace?.retrieved_chunks ? (
                pipelineTrace.retrieved_chunks.map((chunk, index) => {
                  const isReranked = pipelineTrace.reranked_chunks?.some(
                    rc => rc.metadata.doc_id === chunk.metadata.doc_id
                  );
                  const rankIndex = pipelineTrace.reranked_chunks?.findIndex(
                    rc => rc.metadata.doc_id === chunk.metadata.doc_id
                  ) ?? -1;

                  return (
                    <div
                      key={index}
                      className={`group rounded-xl border p-4 transition-all duration-200 ${
                        isReranked
                          ? "border-blue-500/20 bg-blue-500/[0.01]"
                          : "border-slate-800 hover:border-slate-700 bg-slate-950/20"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-[10px] font-bold text-slate-400">
                            #{index + 1}
                          </span>
                          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-slate-400">
                            {chunk.metadata.strategy}
                          </span>
                          {chunk.metadata.page !== undefined && (
                            <span className="text-[10px] text-slate-500">Page {chunk.metadata.page + 1}</span>
                          )}
                        </div>

                        {/* Rank changes / Scores */}
                        <div className="flex gap-1.5 items-center text-[10px] font-mono">
                          {chunk.metadata.relevance_score !== undefined && (
                            <span className={`px-1.5 py-0.5 rounded border ${getScoreColor(chunk.metadata.relevance_score)}`}>
                              Score: {chunk.metadata.relevance_score.toFixed(3)}
                            </span>
                          )}
                          {rankIndex !== -1 && (
                            <span className="px-1.5 py-0.5 rounded border border-blue-500/20 bg-blue-500/10 text-blue-400 font-bold">
                              Rerank: #{rankIndex + 1}
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-sm text-slate-400 line-clamp-3 group-hover:line-clamp-none transition-all duration-300 font-sans">
                        {chunk.content}
                      </p>
                      <div className="mt-2.5 flex items-center justify-between text-[10px] text-slate-500 border-t border-slate-900 pt-2">
                        <span className="truncate max-w-[200px]" title={chunk.metadata.source}>
                          {chunk.metadata.source}
                        </span>
                        {chunk.metadata.section && (
                          <span className="italic truncate max-w-[120px]">{chunk.metadata.section}</span>
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-center py-12 text-slate-500 text-sm">
                  Run a query to view retrieved context chunks.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom: Flowchart-like Pipeline Visualizer */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
        <div className="flex items-center gap-2 mb-6">
          <Cpu className="h-5 w-5 text-blue-400" />
          <h2 className="text-lg font-semibold text-slate-200">Pipeline Tracer Visualizer</h2>
          {pipelineTrace?.latency_ms && (
            <span className="ml-auto flex items-center gap-1 text-xs text-slate-400 bg-slate-800 px-3 py-1 rounded-full">
              <Clock className="h-3.5 w-3.5 text-slate-500" />
              <span>{pipelineTrace.latency_ms} ms Latency</span>
            </span>
          )}
        </div>

        {pipelineTrace ? (
          <div className="flex flex-col gap-6">
            {/* The Node Flowchart */}
            <div className="grid grid-cols-1 md:grid-cols-6 gap-4 items-center relative">
              {/* Step 1: Input Query */}
              <button
                onClick={() => setExpandedStep(expandedStep === "input" ? null : "input")}
                className={`flex flex-col items-center justify-center p-4 rounded-xl border text-center transition-all ${
                  expandedStep === "input" ? "border-blue-500 bg-blue-500/5 shadow-md" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <HelpCircle className="h-6 w-6 text-blue-400 mb-2" />
                <span className="text-xs font-semibold text-slate-300">1. Original Query</span>
                <span className="text-[10px] text-slate-500 font-mono mt-1">Payload input</span>
              </button>

              <div className="hidden md:flex items-center justify-center text-slate-700">
                <ArrowRight className="h-5 w-5" />
              </div>

              {/* Step 2: Transformations */}
              <button
                onClick={() => setExpandedStep(expandedStep === "transform" ? null : "transform")}
                className={`flex flex-col items-center justify-center p-4 rounded-xl border text-center transition-all ${
                  expandedStep === "transform" ? "border-blue-500 bg-blue-500/5 shadow-md" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <SlidersHorizontal className="h-6 w-6 text-amber-400 mb-2" />
                <span className="text-xs font-semibold text-slate-300">2. Query Transform</span>
                <span className="text-[10px] text-slate-500 font-mono mt-1">
                  {pipelineTrace.transformed_queries && pipelineTrace.transformed_queries.length > 0
                    ? `${pipelineTrace.transformed_queries.length} variants`
                    : "No transform"}
                </span>
              </button>

              <div className="hidden md:flex items-center justify-center text-slate-700">
                <ArrowRight className="h-5 w-5" />
              </div>

              {/* Step 3: Retrieval */}
              <button
                onClick={() => setExpandedStep(expandedStep === "retrieval" ? null : "retrieval")}
                className={`flex flex-col items-center justify-center p-4 rounded-xl border text-center transition-all ${
                  expandedStep === "retrieval" ? "border-blue-500 bg-blue-500/5 shadow-md" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <Database className="h-6 w-6 text-emerald-400 mb-2" />
                <span className="text-xs font-semibold text-slate-300">3. Raw Retrieval</span>
                <span className="text-[10px] text-slate-500 font-mono mt-1">
                  {pipelineTrace.retrieved_chunks?.length || 0} chunks fetched
                </span>
              </button>

              <div className="hidden md:flex items-center justify-center text-slate-700">
                <ArrowRight className="h-5 w-5" />
              </div>

              {/* Step 4: Reranking */}
              <button
                onClick={() => setExpandedStep(expandedStep === "rerank" ? null : "rerank")}
                className={`flex flex-col items-center justify-center p-4 rounded-xl border text-center transition-all ${
                  expandedStep === "rerank" ? "border-blue-500 bg-blue-500/5 shadow-md" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <Layers className="h-6 w-6 text-purple-400 mb-2" />
                <span className="text-xs font-semibold text-slate-300">4. Re-Ranking</span>
                <span className="text-[10px] text-slate-500 font-mono mt-1">
                  {pipelineTrace.reranked_chunks && pipelineTrace.reranked_chunks.length > 0
                    ? `${pipelineTrace.reranked_chunks.length} outputs`
                    : "Skipped rerank"}
                </span>
              </button>

              <div className="hidden md:flex items-center justify-center text-slate-700">
                <ArrowRight className="h-5 w-5" />
              </div>

              {/* Step 5: Generation */}
              <button
                onClick={() => setExpandedStep(expandedStep === "generation" ? null : "generation")}
                className={`flex flex-col items-center justify-center p-4 rounded-xl border text-center transition-all col-span-1 md:col-span-2 ${
                  expandedStep === "generation" ? "border-blue-500 bg-blue-500/5 shadow-md" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <Cpu className="h-6 w-6 text-pink-400 mb-2" />
                <span className="text-xs font-semibold text-slate-300">5. LLM Synthesis</span>
                <span className="text-[10px] text-slate-500 font-mono mt-1">
                  Context length: {pipelineTrace.retrieved_chunks?.reduce((acc, c) => acc + c.content.length, 0) || 0} chars
                </span>
              </button>
            </div>

            {/* Step Payload Inspector details */}
            {expandedStep && (
              <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950 p-6 animate-slideDown">
                {expandedStep === "input" && (
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Original User Input Query</h4>
                    <p className="text-sm text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800/60 font-mono">
                      {pipelineTrace.original_query}
                    </p>
                  </div>
                )}

                {expandedStep === "transform" && (
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Query Expansion / Transformations</h4>
                    {pipelineTrace.transformed_queries && pipelineTrace.transformed_queries.length > 0 ? (
                      <div className="space-y-2">
                        {pipelineTrace.transformed_queries.map((q, i) => (
                          <div key={i} className="text-sm text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800/60 font-mono flex items-start gap-2">
                            <span className="text-blue-500 font-semibold">#{i + 1}</span>
                            <span>{q}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500 italic">No query expansions were performed for strategy: {pipelineTrace.strategy}</p>
                    )}
                  </div>
                )}

                {expandedStep === "retrieval" && (
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Raw Retrieved Context Chunks</h4>
                    <div className="space-y-2 max-h-[300px] overflow-y-auto">
                      {pipelineTrace.retrieved_chunks?.map((chunk, i) => (
                        <div key={i} className="text-xs text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800/60">
                          <div className="flex justify-between mb-1">
                            <span className="text-blue-400 font-semibold">Chunk #{i + 1} ({chunk.metadata.source})</span>
                            <span className="font-mono text-slate-500">Score: {chunk.metadata.relevance_score || "N/A"}</span>
                          </div>
                          <p>{chunk.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {expandedStep === "rerank" && (
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Re-Ranking Score Shifts</h4>
                    {pipelineTrace.reranked_chunks && pipelineTrace.reranked_chunks.length > 0 ? (
                      <div className="space-y-2 max-h-[300px] overflow-y-auto">
                        {pipelineTrace.reranked_chunks.map((chunk, i) => (
                          <div key={i} className="text-xs text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800/60">
                            <div className="flex justify-between mb-1">
                              <span className="text-purple-400 font-semibold">Post-Rerank Rank #{i + 1}</span>
                              <span className="font-mono text-slate-500">Re-Score: {chunk.metadata.relevance_score?.toFixed(4)}</span>
                            </div>
                            <p>{chunk.content}</p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500 italic">Re-ranking was not triggered or configured for this strategy run.</p>
                    )}
                  </div>
                )}

                {expandedStep === "generation" && (
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Final Context Assembled & Prompt Template</h4>
                    <div className="text-xs text-slate-400 bg-slate-900 p-4 rounded-lg border border-slate-800/60 font-mono whitespace-pre-wrap leading-relaxed max-h-[300px] overflow-y-auto">
                      {`SYSTEM PROMPT: Use the following retrieved context to answer the user's question.

CONTEXT:
${pipelineTrace.retrieved_chunks?.map((c, idx) => `[${idx + 1}] ${c.content}`).join("\n\n")}

QUESTION:
${pipelineTrace.original_query}

ANSWER:`}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-slate-500 text-sm">
            Visualizer will build a step-by-step trace of your query execution once you run a query.
          </div>
        )}
      </div>
    </div>
  );
}
