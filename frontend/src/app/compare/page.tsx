"use client";

import { useState } from "react";
import {
  Scale,
  Sparkles,
  ArrowRight,
  Clock,
  Cpu,
  CheckCircle,
  HelpCircle,
  FileText,
  AlertCircle
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

  const handleCompare = async () => {
    if (!question.trim()) return;

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/query/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          strategy_a: strategyA,
          strategy_b: strategyB,
        }),
      });

      if (!res.ok) {
        throw new Error(`Comparison request failed with status: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      console.error(e);
      setError(e.message || "An error occurred during comparison.");
    } finally {
      setIsLoading(false);
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
