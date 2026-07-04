"use client";

import { Sparkles, FileText } from "lucide-react";

interface Chunk {
  content: string;
  metadata: Record<string, any>;
}

interface AnswerDisplayProps {
  answer: string;
  isLoading: boolean;
  chunks?: Chunk[];
}

export default function AnswerDisplay({ answer, isLoading, chunks }: AnswerDisplayProps) {
  const uniqueSources = chunks
    ? Array.from(new Set(chunks.map((c) => c.metadata.source).filter(Boolean)))
    : [];

  return (
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
        {answer || (isLoading ? "Thinking..." : "Your generated answer will appear here. Ask a query above to begin.")}
      </div>

      {uniqueSources.length > 0 && (
        <div className="mt-6 pt-6 border-t border-slate-800/80">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Sources Cited</h3>
          <div className="flex flex-wrap gap-2">
            {uniqueSources.map((src, i) => (
              <div key={i} className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-xs text-slate-400">
                <FileText className="h-3 w-3 text-blue-500" />
                <span>{src}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
