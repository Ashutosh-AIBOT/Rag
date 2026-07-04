"use client";

import { useState } from "react";
import { Layers, Shuffle, Columns } from "lucide-react";

interface Chunk {
  content: string;
  metadata: Record<string, any>;
}

interface ChunkInspectorProps {
  retrievedChunks: Chunk[];
  rerankedChunks: Chunk[];
}

export default function ChunkInspector({ retrievedChunks, rerankedChunks }: ChunkInspectorProps) {
  const [viewMode, setViewMode] = useState<"list" | "overlap">("list");

  const getScoreColor = (score?: number) => {
    if (score === undefined) return "text-slate-400 border-slate-800 bg-slate-900";
    if (score >= 0.7 || score > 5.0) return "text-emerald-400 border-emerald-500/20 bg-emerald-500/5";
    if (score >= 0.4 || score > 2.0) return "text-amber-400 border-amber-500/20 bg-amber-500/5";
    return "text-rose-400 border-rose-500/20 bg-rose-500/5";
  };

  // Find suffix-prefix exact text overlaps between chunks
  const getOverlaps = () => {
    const overlaps: Array<{ from: number; to: number; text: string }> = [];
    for (let i = 0; i < retrievedChunks.length; i++) {
      for (let j = 0; j < retrievedChunks.length; j++) {
        if (i === j) continue;
        const text1 = retrievedChunks[i].content;
        const text2 = retrievedChunks[j].content;
        
        const clean1 = text1.trim().replace(/\s+/g, " ");
        const clean2 = text2.trim().replace(/\s+/g, " ");
        const minLen = Math.min(clean1.length, clean2.length, 300);
        let foundOverlap = "";
        for (let len = minLen; len >= 15; len--) {
          const suffix = clean1.slice(-len);
          if (clean2.startsWith(suffix)) {
            foundOverlap = suffix;
            break;
          }
        }
        if (foundOverlap) {
          overlaps.push({
            from: i + 1,
            to: j + 1,
            text: foundOverlap,
          });
        }
      }
    }
    return overlaps;
  };

  // Group chunks by page to show document layout density
  const getPageDensity = () => {
    const pageMap: Record<number, Array<{ id: number; strategy: string }>> = {};
    retrievedChunks.forEach((chunk, idx) => {
      const page = chunk.metadata.page ?? 0;
      if (!pageMap[page]) pageMap[page] = [];
      pageMap[page].push({ id: idx + 1, strategy: chunk.metadata.strategy });
    });
    return pageMap;
  };

  const overlaps = getOverlaps();
  const pageDensity = getPageDensity();

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-indigo-400" />
          <h2 className="text-lg font-semibold text-slate-200">Retrieved Chunks</h2>
        </div>
        
        <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-850">
          <button
            onClick={() => setViewMode("list")}
            className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs font-semibold uppercase tracking-wider transition-all ${
              viewMode === "list"
                ? "bg-blue-600 text-white shadow"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <Columns className="h-3.5 w-3.5" />
            <span>List</span>
          </button>
          <button
            onClick={() => setViewMode("overlap")}
            className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs font-semibold uppercase tracking-wider transition-all ${
              viewMode === "overlap"
                ? "bg-blue-600 text-white shadow"
                : "text-slate-500 hover:text-slate-300"
            }`}
            disabled={retrievedChunks.length === 0}
          >
            <Shuffle className="h-3.5 w-3.5" />
            <span>Overlap Map</span>
          </button>
        </div>
      </div>

      <div className="space-y-4 max-h-[480px] overflow-y-auto pr-1">
        {retrievedChunks.length > 0 ? (
          viewMode === "list" ? (
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
            <div className="space-y-6">
              {/* Document Page Density representation */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-450 border-b border-slate-800 pb-1.5">
                  Page-level Distribution & Densities
                </h4>
                <div className="space-y-2">
                  {Object.keys(pageDensity).map((pageStr) => {
                    const pageNum = parseInt(pageStr);
                    return (
                      <div key={pageNum} className="flex items-center gap-3 bg-slate-950/40 p-2.5 rounded-lg border border-slate-900">
                        <span className="text-xs text-slate-500 font-semibold w-16">Page {pageNum + 1}</span>
                        <div className="flex flex-wrap gap-1.5 flex-1">
                          {pageDensity[pageNum].map((item) => (
                            <span
                              key={item.id}
                              className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-900/10 border border-blue-550/20 text-blue-400"
                              title={`Retrieved Chunk #${item.id} via ${item.strategy}`}
                            >
                              Chunk #{item.id} ({item.strategy})
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Exact suffix-prefix boundary overlaps */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-450 border-b border-slate-800 pb-1.5">
                  Boundary Token Overlaps
                </h4>
                {overlaps.length > 0 ? (
                  <div className="space-y-3">
                    {overlaps.map((overlap, idx) => (
                      <div key={idx} className="bg-amber-500/[0.02] border border-amber-500/10 p-3 rounded-lg text-xs space-y-1.5">
                        <div className="flex justify-between items-center text-[10px] text-amber-400 font-bold uppercase tracking-wider">
                          <span>Transition: Chunk #{overlap.from} ➔ Chunk #{overlap.to}</span>
                          <span>{overlap.text.length} Overlapping Chars</span>
                        </div>
                        <p className="font-mono text-[11px] text-slate-450 bg-slate-950 p-2 rounded leading-relaxed border border-slate-900 select-all">
                          {overlap.text}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-slate-500 text-xs italic">
                    No character-level boundary overlaps detected in these retrieved segments.
                  </div>
                )}
              </div>
            </div>
          )
        ) : (
          <div className="text-center py-12 text-slate-500 text-sm">
            Run a query to view retrieved context chunks.
          </div>
        )}
      </div>
    </div>
  );
}
