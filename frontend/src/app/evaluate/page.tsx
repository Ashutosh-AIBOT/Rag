"use client";

import { useState, useEffect } from "react";
import {
  BarChart2,
  Play,
  TrendingUp,
  AlertTriangle,
  Award,
  CheckCircle,
  HelpCircle,
  FileText,
  Clock,
  Sparkles,
  Search,
  Eye
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

const API_URL = "http://127.0.0.1:8000/api";

interface EvalPair {
  question: string;
  reference_answer: string;
  strategy: string;
}

interface EvalResult {
  id: string;
  query_id: string;
  faithfulness: number;
  relevancy: number;
  precision: number;
  recall: number;
  created_at: string;
  // Extracted during list
  question?: string;
  reference_answer?: string;
  actual_answer?: string;
  strategy?: string;
}

export default function EvaluatePage() {
  const [dataset, setDataset] = useState<EvalPair[]>([]);
  const [evalResults, setEvalResults] = useState<EvalResult[]>([]);
  
  const [isRunning, setIsRunning] = useState(false);
  const [progressText, setProgressText] = useState("");
  const [error, setError] = useState("");

  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [selectedTrace, setSelectedTrace] = useState<any | null>(null);
  const [isLoadingTrace, setIsLoadingTrace] = useState(false);

  useEffect(() => {
    fetchDefaultDataset();
    fetchPastResults();
  }, []);

  const fetchDefaultDataset = async () => {
    try {
      const res = await fetch(`${API_URL}/evaluate/dataset`);
      if (res.ok) {
        const data = await res.json();
        setDataset(data || []);
      }
    } catch (e) {
      console.error("Failed to load dataset", e);
    }
  };

  const fetchPastResults = async () => {
    try {
      const res = await fetch(`${API_URL}/evaluate/results`);
      if (res.ok) {
        const data = await res.json();
        setEvalResults(data.results || []);
      }
    } catch (e) {
      console.error("Failed to load past eval results", e);
    }
  };

  const runEvaluation = async () => {
    if (dataset.length === 0) {
      setError("No evaluation dataset loaded.");
      return;
    }

    setIsRunning(true);
    setError("");
    setProgressText("Starting batch execution of 15 Q&A pairs across strategies...");

    try {
      // We run the batch evaluation endpoint
      const res = await fetch(`${API_URL}/evaluate/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: dataset.map(item => ({
            question: item.question,
            reference_answer: item.reference_answer,
            strategy: item.strategy,
          }))
        }),
      });

      if (!res.ok) {
        throw new Error(`Batch evaluation failed with status: ${res.status}`);
      }

      setProgressText("Computing LLM-as-judge scores (Faithfulness, Relevancy, Precision, Recall)...");
      await res.json();
      setProgressText("Evaluation completed successfully!");
      fetchPastResults();
    } catch (e: any) {
      console.error(e);
      setError(e.message || "An error occurred during evaluation.");
    } finally {
      setIsRunning(false);
    }
  };

  const handleViewTrace = async (queryId: string) => {
    setSelectedTraceId(queryId);
    setIsLoadingTrace(true);
    setSelectedTrace(null);
    try {
      const res = await fetch(`${API_URL}/query/${queryId}/pipeline`);
      if (res.ok) {
        const data = await res.json();
        setSelectedTrace(data.pipeline_trace);
      }
    } catch (e) {
      console.error("Failed to fetch pipeline trace", e);
    } finally {
      setIsLoadingTrace(false);
    }
  };

  // Compile aggregate metrics per strategy
  const getStrategyAggregates = () => {
    const strategies = Array.from(new Set(evalResults.map(r => r.strategy || "hybrid-rerank")));
    return strategies.map(strat => {
      const filtered = evalResults.filter(r => (r.strategy || "hybrid-rerank") === strat);
      const count = filtered.length;
      return {
        strategy: strat.toUpperCase(),
        faithfulness: parseFloat((filtered.reduce((acc, r) => acc + r.faithfulness, 0) / count).toFixed(2)),
        relevancy: parseFloat((filtered.reduce((acc, r) => acc + r.relevancy, 0) / count).toFixed(2)),
        precision: parseFloat((filtered.reduce((acc, r) => acc + r.precision, 0) / count).toFixed(2)),
        recall: parseFloat((filtered.reduce((acc, r) => acc + r.recall, 0) / count).toFixed(2)),
        count
      };
    });
  };

  const chartData = getStrategyAggregates();

  // Strategy leaderboard sorted by average overall score
  const leaderboard = [...chartData].sort((a, b) => {
    const avgA = (a.faithfulness + a.relevancy + a.precision + a.recall) / 4;
    const avgB = (b.faithfulness + b.relevancy + b.precision + b.recall) / 4;
    return avgB - avgA;
  });

  // Failures: any question where any metric < 0.7
  const failures = evalResults.filter(
    r => r.faithfulness < 0.7 || r.relevancy < 0.7 || r.precision < 0.7 || r.recall < 0.7
  );

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 flex-1 flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
            <BarChart2 className="h-6 w-6 text-blue-500" />
            <span>RAG Evaluation & Metrics Dashboard</span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Standardized evaluation of retrieval accuracy and answer quality using LLM-as-judge metrics.
          </p>
        </div>

        <button
          onClick={runEvaluation}
          disabled={isRunning || dataset.length === 0}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-600/20 transition-all hover:from-blue-500 hover:to-indigo-500 disabled:bg-slate-800 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 disabled:shadow-none"
        >
          <Play className="h-4 w-4" />
          <span>{isRunning ? "Evaluating..." : "Run System Evaluation"}</span>
        </button>
      </div>

      {isRunning && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4 flex items-center gap-3">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-white flex-shrink-0" />
          <span className="text-sm font-medium text-blue-400">{progressText}</span>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 text-rose-400">
          <AlertTriangle className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Grid: Charts & Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Side: Recharts Bar Chart */}
        <div className="lg:col-span-8 rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md min-h-[400px] flex flex-col">
          <h3 className="text-base font-semibold text-slate-200 mb-6 flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-blue-400" />
            <span>Retrieval Strategies Comparison</span>
          </h3>

          <div className="flex-1 w-full min-h-[300px]">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="strategy" stroke="#94a3b8" fontSize={11} />
                  <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0b0f19", borderColor: "#1e293b", color: "#f8fafc" }}
                  />
                  <Legend />
                  <Bar dataKey="faithfulness" fill="#3b82f6" name="Faithfulness" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="relevancy" fill="#6366f1" name="Answer Relevancy" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="precision" fill="#10b981" name="Precision" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="recall" fill="#f59e0b" name="Recall" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center py-24 text-slate-500 text-sm">
                <span>No evaluation data available. Run the evaluation dataset above to generate chart.</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Leaderboard */}
        <div className="lg:col-span-4 rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md min-h-[400px]">
          <h3 className="text-base font-semibold text-slate-200 mb-6 flex items-center gap-2">
            <Award className="h-5 w-5 text-indigo-400" />
            <span>Strategy Leaderboard</span>
          </h3>

          <div className="space-y-4">
            {leaderboard.length > 0 ? (
              leaderboard.map((item, index) => {
                const overallScore = (item.faithfulness + item.relevancy + item.precision + item.recall) / 4;
                return (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 rounded-xl border border-slate-800 bg-slate-950/40"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-slate-400">#{index + 1}</span>
                        <span className="text-sm font-semibold text-slate-200">{item.strategy}</span>
                      </div>
                      <span className="text-[10px] text-slate-500">{item.count} test queries executed</span>
                    </div>

                    <div className="text-right">
                      <span className="text-base font-bold text-blue-400">{(overallScore * 100).toFixed(0)}%</span>
                      <p className="text-[10px] text-slate-500 font-semibold uppercase">Avg Score</p>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-12 text-slate-500 text-sm italic">
                Leaderboard will update once data is present.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Grid 2: Failures & Past Results Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Per-Question Results Table */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
          <h3 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <FileText className="h-5 w-5 text-blue-400" />
            <span>Complete Q&A Evaluation Log</span>
          </h3>

          <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
            <table className="w-full border-collapse text-left text-xs text-slate-400">
              <thead>
                <tr className="border-b border-slate-800 text-slate-200">
                  <th className="py-2 px-3 font-semibold">Question</th>
                  <th className="py-2 px-3 font-semibold">Strategy</th>
                  <th className="py-2 px-3 font-semibold">Faith</th>
                  <th className="py-2 px-3 font-semibold">Relev</th>
                  <th className="py-2 px-3 font-semibold">Prec</th>
                  <th className="py-2 px-3 font-semibold">Recall</th>
                  <th className="py-2 px-3 font-semibold">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40">
                {evalResults.length > 0 ? (
                  evalResults.map((row) => (
                    <tr key={row.id} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-2.5 px-3 truncate max-w-[200px]" title={row.question}>
                        {row.question}
                      </td>
                      <td className="py-2.5 px-3 uppercase font-semibold text-[10px] text-slate-500">
                        {row.strategy || "N/A"}
                      </td>
                      <td className="py-2.5 px-3 font-mono">{row.faithfulness.toFixed(2)}</td>
                      <td className="py-2.5 px-3 font-mono">{row.relevancy.toFixed(2)}</td>
                      <td className="py-2.5 px-3 font-mono">{row.precision.toFixed(2)}</td>
                      <td className="py-2.5 px-3 font-mono">{row.recall.toFixed(2)}</td>
                      <td className="py-2.5 px-3">
                        <button
                          onClick={() => handleViewTrace(row.query_id)}
                          className="flex items-center gap-1 text-[10px] font-semibold text-blue-400 hover:text-blue-300"
                        >
                          <Eye className="h-3 w-3" />
                          <span>Trace</span>
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="text-center py-12 text-slate-500">
                      No evaluation log available.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Failure Analysis & Debug Tracer */}
        <div className="lg:col-span-5 rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md min-h-[300px]">
          <h3 className="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            <span>Failure Analysis Trace</span>
          </h3>

          {selectedTraceId ? (
            <div className="space-y-4">
              <div className="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-800">
                <span className="text-[10px] text-slate-500 font-mono">Trace ID: {selectedTraceId}</span>
                <button
                  onClick={() => setSelectedTraceId(null)}
                  className="text-xs text-slate-400 hover:text-slate-200"
                >
                  Clear
                </button>
              </div>

              {isLoadingTrace ? (
                <div className="flex flex-col items-center justify-center py-12 gap-2">
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-white" />
                  <span className="text-xs text-slate-500">Retrieving trace log...</span>
                </div>
              ) : selectedTrace ? (
                <div className="space-y-3">
                  <div>
                    <span className="text-[10px] font-bold text-slate-500">ORIGINAL QUERY</span>
                    <p className="text-xs text-slate-300 bg-slate-950 p-2.5 rounded border border-slate-850 font-mono">{selectedTrace.original_query}</p>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-500">LLM GENERATED ANSWER</span>
                    <p className="text-xs text-slate-300 bg-slate-950 p-2.5 rounded border border-slate-850 leading-relaxed font-sans">{selectedTrace.answer}</p>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-500">RETRIEVED CHUNKS CONTEXT</span>
                    <div className="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
                      {selectedTrace.retrieved_chunks?.map((c: any, i: number) => (
                        <div key={i} className="text-[10px] text-slate-400 bg-slate-950 p-2 rounded border border-slate-900 leading-normal">
                          <span className="font-semibold text-slate-500 block">Chunk #{i+1} ({c.metadata.source})</span>
                          <p>{c.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-500 italic">Failed to retrieve trace data.</div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20 text-slate-500 text-sm text-center">
              <HelpCircle className="h-8 w-8 text-slate-600 mb-2" />
              <span>Click the "Trace" button on any row in the log to inspect its step-by-step pipeline execution here.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
