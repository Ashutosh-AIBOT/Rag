"use client";

import { useState } from "react";
import { Cpu, Clock, HelpCircle, SlidersHorizontal, Database, Layers, ArrowRight } from "lucide-react";

interface Chunk { content: string; metadata: Record<string, any>; }
interface PipelineTrace {
  original_query: string; strategy: string;
  transformed_queries?: string[]; retrieved_chunks?: Chunk[];
  reranked_chunks?: Chunk[]; answer?: string; latency_ms?: number;
}

export default function PipelineVisualizer({ trace }: { trace: PipelineTrace | null }) {
  const [step, setStep] = useState<string | null>(null);
  if (!trace) return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
      <div className="flex items-center gap-2 mb-4"><Cpu className="h-5 w-5 text-blue-400" /><h2 className="text-lg font-semibold text-slate-200">Pipeline Tracer</h2></div>
      <p className="text-center py-8 text-slate-500 text-sm">Run a query to see the pipeline trace.</p>
    </div>
  );

  const nodes = [
    { id: "input", label: "1. Query", Icon: HelpCircle, color: "text-blue-400", sub: "Input" },
    { id: "transform", label: "2. Transform", Icon: SlidersHorizontal, color: "text-amber-400", sub: trace.transformed_queries?.length ? `${trace.transformed_queries.length} variants` : "None" },
    { id: "retrieval", label: "3. Retrieval", Icon: Database, color: "text-emerald-400", sub: `${trace.retrieved_chunks?.length || 0} chunks` },
    { id: "rerank", label: "4. Rerank", Icon: Layers, color: "text-purple-400", sub: trace.reranked_chunks?.length ? `${trace.reranked_chunks.length} out` : "Skipped" },
    { id: "generation", label: "5. LLM", Icon: Cpu, color: "text-pink-400", sub: `${trace.retrieved_chunks?.reduce((a, c) => a + c.content.length, 0) || 0} chars` },
  ];

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/20 p-6 shadow-md">
      <div className="flex items-center gap-2 mb-6">
        <Cpu className="h-5 w-5 text-blue-400" /><h2 className="text-lg font-semibold text-slate-200">Pipeline Tracer</h2>
        {trace.latency_ms && <span className="ml-auto text-xs text-slate-400 bg-slate-800 px-3 py-1 rounded-full flex items-center gap-1"><Clock className="h-3.5 w-3.5" />{trace.latency_ms} ms</span>}
      </div>
      <div className="flex flex-wrap gap-3 items-center">
        {nodes.map((n, i) => (<div key={n.id} className="flex items-center gap-3">
          <button onClick={() => setStep(step === n.id ? null : n.id)} className={`flex flex-col items-center p-3 rounded-xl border min-w-[100px] transition-all ${step === n.id ? "border-blue-500 bg-blue-500/5" : "border-slate-800 bg-slate-950/40 hover:border-slate-700"}`}>
            <n.Icon className={`h-5 w-5 ${n.color} mb-1`} /><span className="text-[11px] font-semibold text-slate-300">{n.label}</span><span className="text-[10px] text-slate-500 font-mono">{n.sub}</span>
          </button>
          {i < nodes.length - 1 && <ArrowRight className="h-4 w-4 text-slate-700 hidden sm:block" />}
        </div>))}
      </div>
      {step && <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950 p-5">
        {step === "input" && <div><h4 className="text-sm font-semibold text-slate-300 mb-2">Original Query</h4><p className="text-sm text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800/60 font-mono">{trace.original_query}</p></div>}
        {step === "transform" && <div><h4 className="text-sm font-semibold text-slate-300 mb-2">Query Transformations</h4>{trace.transformed_queries?.length ? trace.transformed_queries.map((q, i) => <p key={i} className="text-sm text-slate-400 bg-slate-900 p-2 rounded-lg border border-slate-800/60 font-mono mb-1">#{i+1} {q}</p>) : <p className="text-sm text-slate-500 italic">No transforms for {trace.strategy}</p>}</div>}
        {step === "retrieval" && <div><h4 className="text-sm font-semibold text-slate-300 mb-2">Retrieved Chunks</h4><div className="space-y-1.5 max-h-[280px] overflow-y-auto">{trace.retrieved_chunks?.map((c, i) => <div key={i} className="text-xs text-slate-400 bg-slate-900 p-2 rounded-lg border border-slate-800/60"><span className="text-blue-400 font-semibold">#{i+1}</span> {c.content}</div>)}</div></div>}
        {step === "rerank" && <div><h4 className="text-sm font-semibold text-slate-300 mb-2">Re-Ranked Chunks</h4>{trace.reranked_chunks?.length ? <div className="space-y-1.5 max-h-[280px] overflow-y-auto">{trace.reranked_chunks.map((c, i) => <div key={i} className="text-xs text-slate-400 bg-slate-900 p-2 rounded-lg border border-slate-800/60"><span className="text-purple-400 font-semibold">#{i+1}</span> score:{c.metadata.relevance_score?.toFixed(3)} — {c.content}</div>)}</div> : <p className="text-sm text-slate-500 italic">Reranking not triggered.</p>}</div>}
        {step === "generation" && <div><h4 className="text-sm font-semibold text-slate-300 mb-2">Context Sent to LLM</h4><pre className="text-xs text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-800/60 font-mono whitespace-pre-wrap max-h-[280px] overflow-y-auto">{`CONTEXT:\n${trace.retrieved_chunks?.map((c, i) => `[${i+1}] ${c.content}`).join("\n\n")}\n\nQUESTION: ${trace.original_query}`}</pre></div>}
      </div>}
    </div>
  );
}
