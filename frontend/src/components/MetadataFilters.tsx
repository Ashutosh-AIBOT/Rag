"use client";

import { useState } from "react";

interface Doc { id: string; filename: string; }

interface MetadataFiltersProps {
  documents: Doc[];
  selectedSources: string[]; setSelectedSources: (v: string[]) => void;
  filterPageStart: number | ""; setFilterPageStart: (v: number | "") => void;
  filterPageEnd: number | ""; setFilterPageEnd: (v: number | "") => void;
  filterSection: string; setFilterSection: (v: string) => void;
  filterTags: string; setFilterTags: (v: string) => void;
  filterStrategy: string; setFilterStrategy: (v: string) => void;
  k: number; setK: (v: number) => void;
  rerank: boolean; setRerank: (v: boolean) => void;
}

export default function MetadataFilters(props: MetadataFiltersProps) {
  const [docSearch, setDocSearch] = useState("");
  const inputCls = "w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none";
  const labelCls = "block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5";

  const filteredDocs = props.documents.filter(d =>
    d.filename.toLowerCase().includes(docSearch.toLowerCase())
  );

  const toggleSource = (filename: string) => {
    if (props.selectedSources.includes(filename)) {
      props.setSelectedSources(props.selectedSources.filter(s => s !== filename));
    } else {
      props.setSelectedSources([...props.selectedSources, filename]);
    }
  };

  const handleSelectAll = () => {
    props.setSelectedSources(props.documents.map(d => d.filename));
  };

  const handleClearAll = () => {
    props.setSelectedSources([]);
  };

  return (
    <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 pt-6 border-t border-slate-800/80">
      {/* Searchable Document Multi-Selector */}
      <div className="md:col-span-2 flex flex-col">
        <div className="flex items-center justify-between mb-1.5">
          <label className={labelCls}>Filter by Source Documents</label>
          <div className="flex gap-2 text-[10px] font-bold uppercase tracking-wider">
            <button type="button" onClick={handleSelectAll} className="text-blue-400 hover:text-blue-300">Select All</button>
            <span className="text-slate-850">|</span>
            <button type="button" onClick={handleClearAll} className="text-slate-500 hover:text-slate-400">Clear ({props.selectedSources.length})</button>
          </div>
        </div>
        
        <div className="rounded-xl border border-slate-800 bg-slate-950 p-3 space-y-2 flex flex-col h-[180px]">
          <input
            type="text"
            placeholder="Search matching corpus files..."
            value={docSearch}
            onChange={(e) => setDocSearch(e.target.value)}
            className="w-full rounded-lg border border-slate-850 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-300 focus:border-blue-500 focus:outline-none"
          />
          <div className="flex-1 overflow-y-auto space-y-1 pr-1 scrollbar-thin scrollbar-thumb-slate-800">
            {filteredDocs.length > 0 ? (
              filteredDocs.map((d) => {
                const isChecked = props.selectedSources.includes(d.filename);
                return (
                  <label
                    key={d.id}
                    className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-slate-350 cursor-pointer transition-all hover:bg-slate-900/80 ${
                      isChecked ? "bg-blue-600/10 text-blue-300 font-bold border border-blue-500/10" : "border border-transparent"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => toggleSource(d.filename)}
                      className="rounded border-slate-800 bg-slate-950 text-blue-600 focus:ring-0 focus:ring-offset-0 h-3.5 w-3.5"
                    />
                    <span className="truncate" title={d.filename}>{d.filename}</span>
                  </label>
                );
              })
            ) : (
              <div className="text-center py-10 text-xs text-slate-600 italic">No files match your query.</div>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-2">
          <div><label className={labelCls}>Page Start</label><input type="number" placeholder="Min" value={props.filterPageStart} onChange={(e) => props.setFilterPageStart(e.target.value !== "" ? Number(e.target.value) : "")} className={inputCls} /></div>
          <div><label className={labelCls}>Page End</label><input type="number" placeholder="Max" value={props.filterPageEnd} onChange={(e) => props.setFilterPageEnd(e.target.value !== "" ? Number(e.target.value) : "")} className={inputCls} /></div>
        </div>
        <div><label className={labelCls}>Section Name</label><input type="text" placeholder="e.g. Abstract" value={props.filterSection} onChange={(e) => props.setFilterSection(e.target.value)} className={inputCls} /></div>
      </div>

      <div><label className={labelCls}>User Tags</label><input type="text" placeholder="e.g. key-terms" value={props.filterTags} onChange={(e) => props.setFilterTags(e.target.value)} className={inputCls} /></div>
      
      <div>
        <label className={labelCls}>Chunking Strategy</label>
        <select value={props.filterStrategy} onChange={(e) => props.setFilterStrategy(e.target.value)} className={inputCls}>
          <option value="">All</option>
          <option value="recursive">Recursive</option>
          <option value="parent-child">Parent-Child</option>
          <option value="section">Section</option>
          <option value="semantic">Semantic</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div><label className={labelCls}>Retrieve K</label><input type="number" value={props.k} onChange={(e) => props.setK(Number(e.target.value))} className={inputCls} /></div>
        <div className="flex items-center gap-2 pt-6">
          <input type="checkbox" id="rerank-chk" checked={props.rerank} onChange={(e) => props.setRerank(e.target.checked)} className="rounded border-slate-800 bg-slate-950 text-blue-600" />
          <label htmlFor="rerank-chk" className="text-xs text-slate-400 select-none cursor-pointer">Cross Rerank</label>
        </div>
      </div>
    </div>
  );
}
