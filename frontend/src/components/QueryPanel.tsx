"use client";

import { Search, SlidersHorizontal, ChevronDown, ChevronUp } from "lucide-react";

const STRATEGIES = [
  { value: "vector", label: "Basic Vector" },
  { value: "hybrid", label: "Hybrid (BM25 + Vector)" },
  { value: "hybrid-rerank", label: "Hybrid + Rerank (Default)" },
  { value: "parent-child", label: "Parent-Child Retriever" },
  { value: "multi-query", label: "Multi-Query Expansion" },
  { value: "hyde", label: "HyDE (Hypothetical Doc)" },
  { value: "step-back", label: "Step-Back Prompting" },
  { value: "decomposition", label: "Query Decomposition" },
];

interface QueryPanelProps {
  question: string;
  setQuestion: (v: string) => void;
  strategy: string;
  setStrategy: (v: string) => void;
  isLoading: boolean;
  onAsk: () => void;
  showFilters: boolean;
  setShowFilters: (v: boolean) => void;
}

export default function QueryPanel({
  question, setQuestion, strategy, setStrategy, isLoading, onAsk, showFilters, setShowFilters,
}: QueryPanelProps) {
  return (
    <div className="flex flex-col md:flex-row gap-4 items-center">
      <div className="relative flex-1 w-full">
        <Search className="absolute left-4 top-3.5 h-5 w-5 text-slate-400" />
        <input
          type="text"
          placeholder="Ask the LangChain RAG engine a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAsk()}
          className="w-full rounded-xl border border-slate-800 bg-slate-950 py-3.5 pl-12 pr-4 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-base"
          disabled={isLoading}
        />
      </div>
      <StrategySelector strategy={strategy} setStrategy={setStrategy} />
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
        onClick={onAsk}
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
  );
}

export function StrategySelector({ strategy, setStrategy }: { strategy: string; setStrategy: (v: string) => void }) {
  return (
    <select
      value={strategy}
      onChange={(e) => setStrategy(e.target.value)}
      className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3.5 text-sm font-medium text-slate-300 focus:border-blue-500 focus:outline-none"
    >
      {STRATEGIES.map((s) => (
        <option key={s.value} value={s.value}>{s.label}</option>
      ))}
    </select>
  );
}

export { STRATEGIES };
