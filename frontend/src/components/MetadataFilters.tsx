"use client";

interface Doc { id: string; filename: string; }

interface MetadataFiltersProps {
  documents: Doc[];
  filterSource: string; setFilterSource: (v: string) => void;
  filterPageStart: number | ""; setFilterPageStart: (v: number | "") => void;
  filterPageEnd: number | ""; setFilterPageEnd: (v: number | "") => void;
  filterSection: string; setFilterSection: (v: string) => void;
  filterTags: string; setFilterTags: (v: string) => void;
  filterStrategy: string; setFilterStrategy: (v: string) => void;
  k: number; setK: (v: number) => void;
  rerank: boolean; setRerank: (v: boolean) => void;
}

export default function MetadataFilters(props: MetadataFiltersProps) {
  const inputCls = "w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 focus:border-blue-500 focus:outline-none";
  const labelCls = "block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5";

  return (
    <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 pt-4 border-t border-slate-800/80">
      <div>
        <label className={labelCls}>Source Document</label>
        <select value={props.filterSource} onChange={(e) => props.setFilterSource(e.target.value)} className={inputCls}>
          <option value="">All Documents</option>
          {props.documents.map((d) => <option key={d.id} value={d.filename}>{d.filename}</option>)}
        </select>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div><label className={labelCls}>Page Start</label><input type="number" placeholder="Min" value={props.filterPageStart} onChange={(e) => props.setFilterPageStart(e.target.value !== "" ? Number(e.target.value) : "")} className={inputCls} /></div>
        <div><label className={labelCls}>Page End</label><input type="number" placeholder="Max" value={props.filterPageEnd} onChange={(e) => props.setFilterPageEnd(e.target.value !== "" ? Number(e.target.value) : "")} className={inputCls} /></div>
      </div>
      <div><label className={labelCls}>Section Name</label><input type="text" placeholder="e.g. Abstract" value={props.filterSection} onChange={(e) => props.setFilterSection(e.target.value)} className={inputCls} /></div>
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
