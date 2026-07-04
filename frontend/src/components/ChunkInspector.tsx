"use client";

import { Layers } from "lucide-react";

interface Chunk {
  content: string;
  metadata: Record<string, any>;
}

interface ChunkInspectorProps {
  retrievedChunks: Chunk[];
  rerankedChunks: Chunk[];
}

export default function ChunkInspector({ retrievedChunks, rerankedChunks }: ChunkInspectorProps) {
  const getScoreColor = (score?: number) => {
    if (score === undefined) return "text-slate-400 border-slate-800 bg-slate-900";
    if (score >= 0.7 || score > 5.0) return "text-emerald-400 border-emerald-500/20 bg-emerald-500/5";
    if (score >= 0.4 || score > 2.0) return "text-amber-400 border-amber-500/20 bg-amber-500/5";
    return "text-rose-400 border-rose-500/20 bg-rose-500/5";
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-indigo-400" />
          <h2 className="text-lg font-semibold text-slate-200">Retrieved Chunks</h2>
        </div>
        <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-semibold text-slate-400">
          {retrievedChunks.length} Retrieved
        </span>
      </div>

      <div className="space-y-4 max-h-[480px] overflow-y-auto pr-1">
        {retrievedChunks.length > 0 ? (
          retrievedChunks.map((chunk, index) => {
            const isReranked = rerankedChunks.some(
              (rc) => rc.metadata.doc_id === chunk.metadata.doc_id
            );
            const rankIndex = rerankedChunks.findIndex(
              (rc) => rc.metadata.doc_id === chunk.metadata.doc_id
            );

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
  );
}
