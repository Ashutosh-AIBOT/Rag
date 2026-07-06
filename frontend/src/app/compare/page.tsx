"use client";

import { useState, useEffect, useRef } from "react";
import {
  Scale,
  Sparkles,
  ArrowRight,
  Clock,
  Cpu,
  CheckCircle,
  HelpCircle,
  FileText,
  AlertCircle,
  Filter
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000/api";

interface Trace {
  original_query: string;
  strategy: string;
  transformed_queries?: string[];
  retrieved_chunks?: {
    content: string;
    metadata: {
      source?: string;
      page?: number;
      [key: string]: any;
    };
  }[];
  reranked_chunks?: any[];
  answer?: string;
  latency_ms?: number;
}

interface CompareResult {
  query_id_a: string;
  answer_a: string;
  sources_a: string[];
  trace_a: Trace;
  latency_ms_a: number;
  query_id_b: string;
  answer_b: string;
  sources_b: string[];
  trace_b: Trace;
  latency_ms_b: number;
}

export default function ComparePage() {
  const [question, setQuestion] = useState("");
  const [strategyA, setStrategyA] = useState("vector");
  const [strategyB, setStrategyB] = useState("hybrid-rerank");
  
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState("");

  // Target document filters
  const [documents, setDocuments] = useState<any[]>([]);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [showDocSelector, setShowDocSelector] = useState(false);
  const [docSearch, setDocSearch] = useState("");

  // Background task state
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [jobStatus, setJobStatus] = useState("idle");

  const abortControllerRef = useRef<AbortController | null>(null);
  const pollingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/documents`)
      .then((r) => r.json())
      .then((d) => setDocuments(d.documents || []))
      .catch(() => {});

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
      }
    };
  }, []);

  const startWebSocket = (id: string, signal: AbortSignal) => {
    let wsUrl = API_URL.replace(/^http/, "ws") + `/jobs/${id}/ws`;
    if (wsUrl.startsWith("ws://api/")) {
      wsUrl = "ws://localhost:8000/api" + `/jobs/${id}/ws`;
    }
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      if (signal.aborted) {
        socket.close();
        return;
      }
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          setIsLoading(false);
          setJobId(null);
          socket.close();
          return;
        }

        setJobStatus(data.status);
        setProgress(data.progress);

        if (data.status === "completed") {
          setResult(data.result);
          setIsLoading(false);
          setJobId(null);
          socket.close();
        } else if (data.status === "failed") {
          setError(data.error || "Job failed.");
          setIsLoading(false);
          setJobId(null);
          socket.close();
        } else if (data.status === "cancelled") {
          setError("Comparison was cancelled.");
          setIsLoading(false);
          setJobId(null);
          socket.close();
        }
      } catch (err) {
        console.error("Failed to parse socket data", err);
      }
    };

    socket.onerror = (err) => {
      console.warn("WebSocket error, falling back to polling", err);
      socket.close();
      startPolling(id, signal);
    };

    signal.addEventListener("abort", () => {
      socket.close();
    });
  };

  const startPolling = (id: string, signal: AbortSignal) => {
    const poll = async () => {
      if (signal.aborted) return;
      try {
        const response = await fetch(`${API_URL}/jobs/${id}`, { signal });
        if (!response.ok) {
          throw new Error("Failed to fetch job status.");
        }
        const data = await response.json();
        setJobStatus(data.status);
        setProgress(data.progress);
        
        if (data.status === "completed") {
          setResult(data.result);
          setIsLoading(false);
          setJobId(null);
        } else if (data.status === "failed") {
          setError(data.error || "Job failed.");
          setIsLoading(false);
          setJobId(null);
        } else if (data.status === "cancelled") {
          setError("Comparison was cancelled.");
          setIsLoading(false);
          setJobId(null);
        } else {
          // Poll again in 1 second
          pollingTimeoutRef.current = setTimeout(poll, 1000);
        }
      } catch (e: any) {
        if (e.name !== "AbortError") {
          setError(e.message || "Error polling job status.");
          setIsLoading(false);
          setJobId(null);
        }
      }
    };
    
    pollingTimeoutRef.current = setTimeout(poll, 1000);
  };

  const handleCancel = async () => {
    if (jobId) {
      try {
        await fetch(`${API_URL}/jobs/${jobId}/cancel`, { method: "POST" });
      } catch (e) {
        console.error("Cancel API call failed:", e);
      }
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
    }
    setIsLoading(false);
    setJobStatus("cancelled");
    setJobId(null);
    setProgress(0);
  };

  const handleCompare = async () => {
    if (!question.trim()) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (pollingTimeoutRef.current) {
      clearTimeout(pollingTimeoutRef.current);
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError("");
    setResult(null);
    setJobStatus("pending");
    setProgress(0);

    try {
      const res = await fetch(`${API_URL}/query/compare/async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          strategy_a: strategyA,
          strategy_b: strategyB,
          filters: selectedSources.length > 0 ? { source: selectedSources } : undefined,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`Comparison request failed with status: ${res.status}`);
      }

      const data = await res.json();
      setJobId(data.job_id);
      setJobStatus("pending");
      startWebSocket(data.job_id, controller.signal);
    } catch (e: any) {
      if (e.name === "AbortError") {
        console.log("Comparison request aborted.");
      } else {
        console.error(e);
        setError(e.message || "An error occurred during comparison.");
        setIsLoading(false);
      }
    }
  };

  const getTokensEstimate = (query: string, trace?: Trace) => {
    if (!trace) return 0;
    const q_len = query.length;
    const c_len = trace.retrieved_chunks?.reduce((acc, chunk) => acc + chunk.content.length, 0) || 0;
    const a_len = trace.answer?.length || 0;
    return Math.round((q_len + c_len + a_len) / 4);
  };

  // Find overlapping chunks (retrieved by both strategies)
  const getOverlapChunks = () => {
    if (!result?.trace_a?.retrieved_chunks || !result?.trace_b?.retrieved_chunks) return [];
    const contentsA = new Set(result.trace_a.retrieved_chunks.map(c => c.content));
    return result.trace_b.retrieved_chunks.filter(c => contentsA.has(c.content));
  };

  const overlap = getOverlapChunks();

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 flex flex-col gap-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
          <Scale className="h-6 w-6 text-blue-500" />
          <span>A/B Strategy Comparison</span>
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Run Strategy A and Strategy B side-by-side on the same query to compare performance, latency, and retrieve metrics.
        </p>
      </div>

      {/* Inputs Form */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur-sm shadow-xl">
        <div className="flex flex-col gap-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Comparison Query</label>
            <input
              type="text"
              placeholder="Enter a complex, factual or multi-hop query to compare..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCompare()}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 py-3.5 px-4 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-base"
              disabled={isLoading}
            />
          </div>

          {/* Collapsible Document Selector */}
          <div className="border-t border-b border-slate-800/60 py-3 my-1">
            <button
              type="button"
              onClick={() => setShowDocSelector(!showDocSelector)}
              className="flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors uppercase tracking-wider"
            >
              <Filter className="h-3.5 w-3.5 text-blue-400" />
              <span>Target Documents ({selectedSources.length === 0 ? "All" : `${selectedSources.length} selected`})</span>
            </button>
            
            {showDocSelector && (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 animate-fadeIn">
                <div className="flex flex-col gap-2">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setSelectedSources(documents.map(d => d.filename))}
                      className="text-[10px] font-bold text-blue-400 hover:text-blue-300 uppercase tracking-wider"
                    >
                      Select All
                    </button>
                    <span className="text-slate-800 text-[10px]">|</span>
                    <button
                      type="button"
                      onClick={() => setSelectedSources([])}
                      className="text-[10px] font-bold text-slate-500 hover:text-slate-400 uppercase tracking-wider"
                    >
                      Clear Selection
                    </button>
                  </div>
                  <input
                    type="text"
                    placeholder="Search matching files..."
                    value={docSearch}
                    onChange={(e) => setDocSearch(e.target.value)}
                    className="w-full rounded-lg border border-slate-850 bg-slate-950 px-3 py-1.5 text-xs text-slate-350 focus:border-blue-500 focus:outline-none"
                  />
                </div>

                <div className="rounded-xl border border-slate-850 bg-slate-950/40 p-2.5 max-h-[140px] overflow-y-auto space-y-1 scrollbar-thin scrollbar-thumb-slate-800">
                  {documents.filter(d => d.filename.toLowerCase().includes(docSearch.toLowerCase())).length > 0 ? (
                    documents.filter(d => d.filename.toLowerCase().includes(docSearch.toLowerCase())).map((d) => {
                      const isChecked = selectedSources.includes(d.filename);
                      return (
                        <label
                          key={d.id}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-slate-350 cursor-pointer transition-colors hover:bg-slate-900/65 ${
                            isChecked ? "bg-blue-600/10 text-blue-350 font-medium" : ""
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => {
                              if (isChecked) {
                                setSelectedSources(selectedSources.filter(s => s !== d.filename));
                              } else {
                                setSelectedSources([...selectedSources, d.filename]);
                              }
                            }}
                            className="rounded border-slate-800 bg-slate-950 text-blue-600 focus:ring-0 focus:ring-offset-0 h-3.5 w-3.5"
                          />
                          <span className="truncate text-slate-300" title={d.filename}>{d.filename}</span>
                        </label>
                      );
                    })
                  ) : (
                    <div className="text-center py-6 text-xs text-slate-600 italic">No matching files found.</div>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-4 items-end">
            <div className="flex-1 w-full">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Strategy A</label>
              <select
                value={strategyA}
                onChange={(e) => setStrategyA(e.target.value)}
                className="w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-3.5 text-sm font-medium text-slate-300 focus:border-blue-500 focus:outline-none"
              >
                <option value="vector">Basic Vector</option>
                <option value="hybrid">Hybrid (BM25 + Vector)</option>
                <option value="hybrid-rerank">Hybrid + Rerank</option>
                <option value="parent-child">Parent-Child Retriever</option>
                <option value="multi-query">Multi-Query Expansion</option>
                <option value="hyde">HyDE (Hypothetical Doc)</option>
                <option value="step-back">Step-Back Prompting</option>
                <option value="decomposition">Query Decomposition</option>
              </select>
            </div>

            <div className="hidden sm:flex items-center justify-center text-slate-700 pb-4">
              <ArrowRight className="h-5 w-5" />
            </div>

            <div className="flex-1 w-full">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5">Strategy B</label>
              <select
                value={strategyB}
                onChange={(e) => setStrategyB(e.target.value)}
                className="w-full rounded-xl border border-slate-800 bg-slate-950 px-4 py-3.5 text-sm font-medium text-slate-300 focus:border-blue-500 focus:outline-none"
              >
                <option value="vector">Basic Vector</option>
                <option value="hybrid">Hybrid (BM25 + Vector)</option>
                <option value="hybrid-rerank">Hybrid + Rerank (Default)</option>
                <option value="parent-child">Parent-Child Retriever</option>
                <option value="multi-query">Multi-Query Expansion</option>
                <option value="hyde">HyDE (Hypothetical Doc)</option>
                <option value="step-back">Step-Back Prompting</option>
                <option value="decomposition">Query Decomposition</option>
              </select>
            </div>

            <button
              onClick={handleCompare}
              disabled={isLoading || !question.trim()}
              className="w-full sm:w-auto flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-8 py-3.5 text-sm font-semibold text-white shadow-lg shadow-blue-600/20 transition-all hover:bg-blue-500 hover:shadow-blue-500/30 disabled:bg-slate-800 disabled:text-slate-500 disabled:shadow-none"
            >
              {isLoading ? (
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-white" />
              ) : (
                <span>Compare</span>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Background Job Progress Dashboard */}
      {isLoading && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl flex flex-col gap-4 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-blue-500 animate-ping" />
              <h3 className="text-sm font-semibold text-slate-200">
                Background worker executing strategy comparison...
              </h3>
            </div>
            <button
              onClick={handleCancel}
              className="text-xs font-semibold text-rose-400 hover:text-rose-350 transition-colors uppercase tracking-wider bg-rose-500/10 border border-rose-500/20 px-3 py-1.5 rounded-lg"
            >
              Cancel Job
            </button>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-500">
              <span>Status: <strong className="text-blue-400 uppercase">{jobStatus}</strong></span>
              <span>{Math.round(progress * 100)}% Complete</span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-500 ease-out"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 text-rose-400">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Comparison Layout */}
      {result && (
        <div className="flex flex-col gap-8">
          {/* Performance Dashboard Table */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
            <h3 className="text-base font-semibold text-slate-200 mb-4">Pipeline Metrics Leaderboard</h3>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm text-slate-400">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="py-3 px-4 font-semibold text-slate-200">Metric</th>
                    <th className="py-3 px-4 font-semibold text-blue-400">Strategy A: {strategyA.toUpperCase()}</th>
                    <th className="py-3 px-4 font-semibold text-indigo-400">Strategy B: {strategyB.toUpperCase()}</th>
                    <th className="py-3 px-4 font-semibold text-slate-200">Delta / Winner</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr>
                    <td className="py-3 px-4 font-medium text-slate-300">Retrieval Latency</td>
                    <td className="py-3 px-4 font-mono text-blue-300">{result.latency_ms_a} ms</td>
                    <td className="py-3 px-4 font-mono text-indigo-300">{result.latency_ms_b} ms</td>
                    <td className="py-3 px-4">
                      {result.latency_ms_a === result.latency_ms_b ? (
                        <span className="text-slate-500">Tie</span>
                      ) : result.latency_ms_a < result.latency_ms_b ? (
                        <span className="text-emerald-400">A is {(result.latency_ms_b - result.latency_ms_a).toFixed(0)}ms faster</span>
                      ) : (
                        <span className="text-emerald-400">B is {(result.latency_ms_a - result.latency_ms_b).toFixed(0)}ms faster</span>
                      )}
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-slate-300">Context Chunks Retrieved</td>
                    <td className="py-3 px-4 font-mono">{result.trace_a?.retrieved_chunks?.length || 0} chunks</td>
                    <td className="py-3 px-4 font-mono">{result.trace_b?.retrieved_chunks?.length || 0} chunks</td>
                    <td className="py-3 px-4 text-slate-400">
                      Overlap: {overlap.length} chunks
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-slate-300">Est. Total Tokens Used</td>
                    <td className="py-3 px-4 font-mono">{getTokensEstimate(question, result.trace_a)} tokens</td>
                    <td className="py-3 px-4 font-mono">{getTokensEstimate(question, result.trace_b)} tokens</td>
                    <td className="py-3 px-4 text-slate-500">
                      {Math.abs(getTokensEstimate(question, result.trace_a) - getTokensEstimate(question, result.trace_b))} token diff
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Side-by-Side Answers */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Strategy A Answer */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md flex flex-col min-h-[300px]">
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="h-5 w-5 text-blue-400" />
                <h3 className="text-base font-semibold text-slate-200">Answer A ({strategyA})</h3>
              </div>
              <p className="flex-1 text-slate-300 text-sm leading-relaxed whitespace-pre-wrap font-sans">
                {result.answer_a}
              </p>
              {result.sources_a.length > 0 && (
                <div className="mt-4 pt-4 border-t border-slate-800 text-xs text-slate-500">
                  Sources: {Array.from(new Set(result.sources_a)).join(", ")}
                </div>
              )}
            </div>

            {/* Strategy B Answer */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md flex flex-col min-h-[300px]">
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="h-5 w-5 text-indigo-400" />
                <h3 className="text-base font-semibold text-slate-200">Answer B ({strategyB})</h3>
              </div>
              <p className="flex-1 text-slate-300 text-sm leading-relaxed whitespace-pre-wrap font-sans">
                {result.answer_b}
              </p>
              {result.sources_b.length > 0 && (
                <div className="mt-4 pt-4 border-t border-slate-800 text-xs text-slate-500">
                  Sources: {Array.from(new Set(result.sources_b)).join(", ")}
                </div>
              )}
            </div>
          </div>

          {/* Side-by-Side Chunk Retrieval List */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Strategy A Chunks */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
              <h4 className="text-sm font-semibold text-slate-300 mb-4">Strategy A Retrieved Chunks</h4>
              <div className="space-y-3 max-h-[360px] overflow-y-auto pr-1">
                {result.trace_a?.retrieved_chunks?.map((chunk, idx) => {
                  const isShared = overlap.some(oc => oc.content === chunk.content);
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border text-xs leading-relaxed ${
                        isShared
                          ? "border-emerald-500/20 bg-emerald-500/[0.02] text-slate-300"
                          : "border-slate-850 bg-slate-950/40 text-slate-400"
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-semibold text-slate-500">Chunk #{idx + 1}</span>
                        {isShared && <span className="text-[10px] text-emerald-400 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/10">Overlap</span>}
                      </div>
                      <p>{chunk.content}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Strategy B Chunks */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
              <h4 className="text-sm font-semibold text-slate-300 mb-4">Strategy B Retrieved Chunks</h4>
              <div className="space-y-3 max-h-[360px] overflow-y-auto pr-1">
                {result.trace_b?.retrieved_chunks?.map((chunk, idx) => {
                  const isShared = overlap.some(oc => oc.content === chunk.content);
                  return (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border text-xs leading-relaxed ${
                        isShared
                          ? "border-emerald-500/20 bg-emerald-500/[0.02] text-slate-300"
                          : "border-slate-850 bg-slate-950/40 text-slate-400"
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-semibold text-slate-500">Chunk #{idx + 1}</span>
                        {isShared && <span className="text-[10px] text-emerald-400 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/10">Overlap</span>}
                      </div>
                      <p>{chunk.content}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
