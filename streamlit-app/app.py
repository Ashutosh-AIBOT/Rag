import streamlit as st
import requests
import time

# Set up page configurations
st.set_page_config(
    page_title="Advanced RAG Portal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend server endpoint URL
BACKEND_URL = "http://localhost:8000/api"

# Helper to fetch active documents
def fetch_documents():
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# Helper to fetch evaluation history
def fetch_eval_history():
    try:
        response = requests.get(f"{BACKEND_URL}/evaluate/results")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# Fetch docs for filters
documents_list = fetch_documents()

# Custom Styling for premium look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Design Sidebar
st.sidebar.title("⚙️ RAG Configurations")

# Active Metadata Filters
st.sidebar.subheader("🎯 Metadata Filters")
doc_options = {"All Documents": None}
for d in documents_list:
    doc_options[d.get('filename')] = d.get('id')

selected_doc_name = st.sidebar.selectbox("Filter by Document Source", list(doc_options.keys()))
selected_doc_id = doc_options[selected_doc_name]

# Chunking Strategy Filter
selected_chunk_strategy = st.sidebar.selectbox(
    "Filter by Chunking Strategy",
    ["All", "recursive", "parent-child", "section", "semantic"]
)
chunk_strategy_val = None if selected_chunk_strategy == "All" else selected_chunk_strategy

# Tag filtering
tags_filter_input = st.sidebar.text_input("Filter by Tags (comma separated)", "")

# Page number filtering
page_filter_input = st.sidebar.text_input("Filter by Page Number (e.g. 5)", "")

# Build filters dict
filters_dict = {}
if selected_doc_id:
    filters_dict["doc_id"] = selected_doc_id
if chunk_strategy_val:
    filters_dict["chunk_strategy"] = chunk_strategy_val
if tags_filter_input:
    filters_dict["tags"] = tags_filter_input
if page_filter_input:
    filters_dict["page"] = page_filter_input

# Check health
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 System Health")
if st.sidebar.button("Check Backend Health"):
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            status = response.json()
            st.sidebar.success(f"Backend Status: {status.get('status').upper()}")
            st.sidebar.json(status)
        else:
            st.sidebar.error(f"Health check failed (HTTP {response.status_code})")
    except Exception as e:
        st.sidebar.error(f"Cannot reach backend server. Error: {e}")

# Design Main UI
st.title("🤖 Advanced Hybrid RAG Platform")
st.markdown("A modular RAG assignment platform demonstrating multiple chunking strategies, hybrid search, RRF merging, cross-encoder re-ranking, and query transformations.")
st.markdown("---")

tab_chat, tab_compare, tab_eval, tab_upload, tab_manage, tab_stats = st.tabs([
    "💬 Ask & Query (RAG)", 
    "⚖️ Compare Strategies (A/B)", 
    "🧪 Evaluation (LLM Judge)",
    "📤 Upload & Index Chunks", 
    "📚 Manage Documents",
    "📊 Platform Stats & Diagnostics"
])

# RAG Strategy list
strategy_mapping = {
    "Dense Similarity Search Only": "dense",
    "Sparse Keyword (BM25) Search Only": "sparse",
    "Hybrid (Dense + Sparse with RRF)": "hybrid",
    "Hybrid + Re-ranking (Cross-Encoder)": "hybrid_rerank",
    "Multi-Query Expansion + Hybrid + Re-ranking": "multiquery_hybrid_rerank",
    "HyDE (Hypothetical Doc Embeddings) + Hybrid + Re-ranking": "hyde_hybrid_rerank",
    "Query Decomposition + Hybrid + Re-ranking": "decomposed_hybrid_rerank",
    "Step-Back Prompting + Hybrid + Re-ranking": "step_back_hybrid_rerank"
}

# =============================================================================
# TAB: ASK & QUERY (RAG)
# =============================================================================
with tab_chat:
    st.subheader("Query the Vector Index")
    
    col_input, col_strat = st.columns([2, 1])
    with col_input:
        query_text = st.text_input("Enter your prompt / question", "What is self-attention?")
    with col_strat:
        selected_strategy_label = st.selectbox("Select Retrieval & Generation Strategy", list(strategy_mapping.keys()), index=3)
        strategy_val = strategy_mapping[selected_strategy_label]
        
    if st.button("🔍 Run RAG Pipeline", key="run_rag_btn"):
        if not query_text.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Executing RAG Pipeline (retrieval, re-ranking, LLM chain)..."):
                payload = {
                    "query": query_text,
                    "strategy": strategy_val,
                    "filters": filters_dict
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/query", json=payload)
                    if response.status_code == 200:
                        res_data = response.json()
                        answer = res_data.get("answer")
                        trace = res_data.get("trace", {})
                        
                        st.markdown("### 💡 Answer")
                        st.info(answer)
                        
                        # Metrics
                        st.markdown("### 📊 Execution Performance")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{trace.get("latency_ms", "N/A")} ms</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{trace.get("token_usage", {}).get("input", 0)}</div><div class="metric-label">Input Tokens (est.)</div></div>', unsafe_allow_html=True)
                        with m3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{trace.get("token_usage", {}).get("output", 0)}</div><div class="metric-label">Output Tokens (est.)</div></div>', unsafe_allow_html=True)
                        with m4:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{trace.get("token_usage", {}).get("total", 0)}</div><div class="metric-label">Total Tokens (est.)</div></div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Transparent Trace
                        with st.expander("🛠️ View Pipeline Execution Trace (Step-by-Step)"):
                            # 1. Query Transformations
                            if trace.get("transformations"):
                                st.markdown("#### 1. Query Transformations")
                                for k, val in trace.get("transformations").items():
                                    st.write(f"**{k.replace('_', ' ').title()}:**")
                                    if isinstance(val, list):
                                        for q in val:
                                            st.write(f"- `{q}`")
                                    else:
                                        st.write(f"`{val}`")
                                st.markdown("---")
                                
                            # 2. Initial Retrieval
                            st.markdown("#### 2. Initial Retrieved Chunks (Top 20)")
                            if trace.get("initial_retrieval"):
                                for idx, item in enumerate(trace.get("initial_retrieval")):
                                    meta = item.get("metadata", {})
                                    score = item.get("score", 0.0)
                                    st.markdown(f"**Chunk {idx+1}:** (Score: `{score:.4f}` | Source: `{meta.get('filename')}` | Strategy: `{meta.get('chunk_strategy')}` | Page: `{meta.get('page', 'N/A')}`)")
                                    st.text_area(f"Content {idx+1}", item.get("content"), height=100, key=f"init_content_{idx}")
                            else:
                                st.write("No document chunks retrieved.")
                            st.markdown("---")
                                
                            # 3. Re-ranked Retrieval
                            if trace.get("re_ranked_retrieval"):
                                st.markdown("#### 3. Re-ranked Chunks (Top 5)")
                                for idx, item in enumerate(trace.get("re_ranked_retrieval")):
                                    meta = item.get("metadata", {})
                                    score = item.get("score", 0.0)
                                    st.markdown(f"**Chunk {idx+1}:** (Re-rank Score: `{score:.4f}` | Source: `{meta.get('filename')}` | Strategy: `{meta.get('chunk_strategy')}` | Page: `{meta.get('page', 'N/A')}`)")
                                    st.text_area(f"Re-ranked Content {idx+1}", item.get("content"), height=100, key=f"rerank_content_{idx}")
                                st.markdown("---")
                                
                            # 4. Final Assembled Prompt
                            st.markdown("#### 4. Final Context System Prompt")
                            st.text_area("Formatted System Prompt", trace.get("prompt"), height=250)
                            
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

# =============================================================================
# TAB: COMPARE STRATEGIES (A/B)
# =============================================================================
with tab_compare:
    st.subheader("Side-by-Side RAG Comparison")
    st.markdown("Select two different strategies to execute the same query and compare their outputs, scores, and latency side-by-side.")
    
    compare_query = st.text_input("Enter comparison query", "Explain self-attention")
    
    col_a, col_b = st.columns(2)
    with col_a:
        strategy_a_label = st.selectbox("Strategy A", list(strategy_mapping.keys()), index=0)
        strategy_a_val = strategy_mapping[strategy_a_label]
    with col_b:
        strategy_b_label = st.selectbox("Strategy B", list(strategy_mapping.keys()), index=3)
        strategy_b_val = strategy_mapping[strategy_b_label]
        
    if st.button("⚖️ Run A/B Comparison", key="compare_btn"):
        if not compare_query.strip():
            st.warning("Please enter a comparison query.")
        else:
            with st.spinner("Running side-by-side comparison..."):
                payload = {
                    "query": compare_query,
                    "strategy_a": strategy_a_val,
                    "strategy_b": strategy_b_val,
                    "filters": filters_dict
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/query/compare", json=payload)
                    if response.status_code == 200:
                        res_data = response.json()
                        a_data = res_data.get("strategy_a", {})
                        b_data = res_data.get("strategy_b", {})
                        
                        col_res_a, col_res_b = st.columns(2)
                        
                        # STRATEGY A Column
                        with col_res_a:
                            st.markdown(f"### 🛡️ Strategy A: {strategy_a_label}")
                            st.info(a_data.get("answer"))
                            
                            st.markdown("#### Performance Metrics")
                            st.write(f"- **Latency:** `{a_data.get('trace', {}).get('latency_ms', 'N/A')} ms`")
                            st.write(f"- **Total Tokens:** `{a_data.get('trace', {}).get('token_usage', {}).get('total', 0)}`")
                            
                            st.markdown("#### Top Chunks Used")
                            for idx, chunk in enumerate(a_data.get("trace", {}).get("context_assembled", [])):
                                meta = chunk.get("metadata", {})
                                st.markdown(f"**Chunk {idx+1}:** `{meta.get('filename')}` | Page `{meta.get('page')}`")
                                st.caption(chunk.get("content")[:200] + "...")
                        
                        # STRATEGY B Column
                        with col_res_b:
                            st.markdown(f"### 🎯 Strategy B: {strategy_b_label}")
                            st.info(b_data.get("answer"))
                            
                            st.markdown("#### Performance Metrics")
                            st.write(f"- **Latency:** `{b_data.get('trace', {}).get('latency_ms', 'N/A')} ms`")
                            st.write(f"- **Total Tokens:** `{b_data.get('trace', {}).get('token_usage', {}).get('total', 0)}`")
                            
                            st.markdown("#### Top Chunks Used")
                            for idx, chunk in enumerate(b_data.get("trace", {}).get("context_assembled", [])):
                                meta = chunk.get("metadata", {})
                                st.markdown(f"**Chunk {idx+1}:** `{meta.get('filename')}` | Page `{meta.get('page')}`")
                                st.caption(chunk.get("content")[:200] + "...")
                                
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

# =============================================================================
# TAB: EVALUATION (LLM JUDGE)
# =============================================================================
with tab_eval:
    st.subheader("🧪 Quantitative LLM-as-Judge Evaluation")
    
    eval_mode = st.radio("Select Evaluation Mode", ["Single Q&A Evaluation", "Batch Evaluation (15-Q Dataset)"])
    
    if eval_mode == "Batch Evaluation (15-Q Dataset)":
        st.markdown("Run a quantitative evaluation across the built-in 15-question evaluation dataset to measure Faithfulness, Relevancy, Context Precision, and Context Recall (scored from 0.0 to 1.0).")
        col_eval_strat, col_eval_action = st.columns([2, 1])
        with col_eval_strat:
            eval_strategy_label = st.selectbox("Select Strategy to Evaluate", list(strategy_mapping.keys()), index=3, key="eval_strat_select")
            eval_strategy_val = strategy_mapping[eval_strategy_label]
        with col_eval_action:
            st.write("") # Spacing
            run_eval_clicked = st.button("🧪 Run Batch Eval", key="run_eval_btn")
            
        if run_eval_clicked:
            with st.spinner("Executing batch evaluation runs through the judge LLM..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/evaluate/batch", json={"strategy": eval_strategy_val})
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success("✅ Evaluation run completed successfully!")
                        
                        e_m1, e_m2, e_m3, e_m4, e_m5 = st.columns(5)
                        with e_m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{res_data.get("avg_faithfulness", 0.0):.2f}</div><div class="metric-label">Avg Faithfulness</div></div>', unsafe_allow_html=True)
                        with e_m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{res_data.get("avg_relevancy", 0.0):.2f}</div><div class="metric-label">Avg Relevancy</div></div>', unsafe_allow_html=True)
                        with e_m3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{res_data.get("avg_precision", 0.0):.2f}</div><div class="metric-label">Avg Precision</div></div>', unsafe_allow_html=True)
                        with e_m4:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{res_data.get("avg_recall", 0.0):.2f}</div><div class="metric-label">Avg Recall</div></div>', unsafe_allow_html=True)
                        with e_m5:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{res_data.get("total_latency_ms", 0) / 1000:.2f} s</div><div class="metric-label">Total Duration</div></div>', unsafe_allow_html=True)
                            
                        # Detailed results
                        st.markdown("### 📋 Detailed Score Breakdowns")
                        for q_id, info in res_data.get("detailed", {}).items():
                            with st.expander(f"Question {q_id}: {info.get('question')}"):
                                st.write(f"**Category:** `{info.get('category')}`")
                                st.write(f"**Reference Answer:** *{info.get('reference')}*")
                                st.write(f"**Generated Answer:** *{info.get('answer')}*")
                                st.markdown(f"- **Faithfulness Score:** `{info.get('faithfulness'):.2f}` (Judge Reason: *{info.get('faithfulness_reason')}*)")
                                st.markdown(f"- **Relevancy Score:** `{info.get('relevancy'):.2f}` (Judge Reason: *{info.get('relevancy_reason')}*)")
                                st.markdown(f"- **Context Precision:** `{info.get('precision'):.2f}` (Judge Reason: *{info.get('precision_reason')}*)")
                                st.markdown(f"- **Context Recall:** `{info.get('recall'):.2f}` (Judge Reason: *{info.get('recall_reason')}*)")
                                st.markdown(f"- **Latency:** `{info.get('latency_ms')} ms`")
                    else:
                        st.error(f"Error running evaluation: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    
    else: # Single Q&A Evaluation
        st.markdown("Evaluate a custom question, generated answer, and context chunk to calculate Faithfulness, Relevancy, Context Precision, and Recall using the judge LLM.")
        with st.form("single_eval_form"):
            se_question = st.text_input("Question", "What is complexity per layer for Self-Attention?")
            se_answer = st.text_area("Generated Answer", "O(n^2 * d) where n is sequence length and d is representation dimension.")
            se_context = st.text_area("Retrieved Context Chunks (separate multiple chunks with double newlines)", "Self-attention layers relate all positions in a sequence. The complexity per layer is O(n^2 * d).")
            se_reference = st.text_input("Reference/Ground Truth Answer (Optional - required for Context Recall)", "O(n^2 * d)")
            
            submit_se = st.form_submit_button("🧪 Evaluate Q&A")
            
        if submit_se:
            if not se_question.strip() or not se_answer.strip() or not se_context.strip():
                st.warning("Please fill in Question, Answer, and Context fields.")
            else:
                with st.spinner("Evaluating Q&A with LLM Judge..."):
                    payload = {
                        "question": se_question,
                        "answer": se_answer,
                        "context": se_context,
                        "reference": se_reference.strip() if se_reference.strip() else None
                    }
                    try:
                        response = requests.post(f"{BACKEND_URL}/evaluate", json=payload)
                        if response.status_code == 200:
                            res = response.json()
                            st.success("✅ Custom Q&A evaluation completed!")
                            
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{res.get("faithfulness", 0.0):.2f}</div><div class="metric-label">Faithfulness</div></div>', unsafe_allow_html=True)
                            with c2:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{res.get("relevancy", 0.0):.2f}</div><div class="metric-label">Relevancy</div></div>', unsafe_allow_html=True)
                            with c3:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{res.get("context_precision", 0.0):.2f}</div><div class="metric-label">Context Precision</div></div>', unsafe_allow_html=True)
                            with c4:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{res.get("context_recall", 0.0):.2f}</div><div class="metric-label">Context Recall</div></div>', unsafe_allow_html=True)
                                
                            # Show reasons
                            st.markdown("### 📋 Judge Explanations")
                            reasons = res.get("reasons", {})
                            st.markdown(f"**Faithfulness Explanation:** *{reasons.get('faithfulness', 'N/A')}*")
                            st.markdown(f"**Relevancy Explanation:** *{reasons.get('relevancy', 'N/A')}*")
                            st.markdown(f"**Context Precision Explanation:** *{reasons.get('context_precision', 'N/A')}*")
                            st.markdown(f"**Context Recall Explanation:** *{reasons.get('context_recall', 'N/A')}*")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to connect: {e}")

    # History Table
    st.markdown("---")
    st.subheader("📜 Past Evaluation Run History")
    eval_history = fetch_eval_history()
    if not eval_history:
        st.info("No past evaluation runs found.")
    else:
        st.table(eval_history)

# =============================================================================
# TAB: UPLOAD & INDEX CHUNKS
# =============================================================================
with tab_upload:
    st.subheader("Ingest New Document")
    st.write("Upload a file to automatically trigger four chunking strategies (Recursive, Parent-Child, Section, and Semantic) and index them in ChromaDB.")
    
    # Show last upload success message if any
    if "upload_success" in st.session_state:
        st.success(st.session_state.upload_success)
        del st.session_state.upload_success
        
    uploaded_file = st.file_uploader("Upload PDF, DOCX, TXT, or MD file", type=["pdf", "docx", "txt", "md"])
    tags_input = st.text_input("Custom Tags (comma separated, e.g. invoice, finance, notes)", "")
    
    if st.button("🚀 Upload & Process"):
        if uploaded_file is not None:
            with st.spinner("Submitting ingestion task to background queue..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"tags": tags_input}
                
                try:
                    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files, data=data)
                    if response.status_code == 202:
                        res_data = response.json()
                        st.session_state.upload_success = f"✅ Ingestion task successfully scheduled for **{uploaded_file.name}**! (ID: `{res_data.get('document_id')}`). It will index in the background."
                        st.rerun()
                    else:
                        st.error(f"Failed to submit document. Status Code: {response.status_code}. Response: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")
        else:
            st.warning("Please select a file to upload first.")
            
    st.markdown("---")
    st.subheader("📋 Status of Ingested Documents")
    st.write("Current files stored in the platform:")
    
    # Reload documents
    docs = fetch_documents()
    if not docs:
        st.info("No documents have been uploaded yet.")
    else:
        for doc in docs:
            st.markdown(f"- **📄 {doc.get('filename')}** — `{doc.get('file_type').upper()}` | Size: `{doc.get('file_size')} bytes` | Pages: `{doc.get('total_pages')}` | Ingested: `{doc.get('uploaded_at')}`")

# =============================================================================================================
# TAB: MANAGE DOCUMENTS
# =============================================================================
with tab_manage:
    st.subheader("Ingested Documents")
    st.write("View or delete documents currently stored in the SQLite database and ChromaDB vector store.")
    
    if st.button("🔄 Refresh Document List"):
        st.rerun()
        
    if not documents_list:
        st.info("No documents are currently ingested in the platform.")
    else:
        for doc in documents_list:
            with st.expander(f"📄 {doc.get('filename')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Document ID:** `{doc.get('id')}`")
                    st.markdown(f"**File Format:** `{doc.get('file_type')}`")
                    st.markdown(f"**File Size:** `{doc.get('file_size')} bytes`")
                    st.markdown(f"**Ingested At:** `{doc.get('uploaded_at')}`")
                    tags = doc.get('tags')
                    if tags and tags != ['']:
                        st.markdown(f"**Tags:** " + " ".join([f"`{t}`" for t in tags]))
                    else:
                        st.markdown("**Tags:** *None*")
                with col2:
                    if st.button("🗑️ Delete Document", key=doc.get('id')):
                        with st.spinner("Deleting document and clearing vector chunks..."):
                            del_resp = requests.delete(f"{BACKEND_URL}/documents/{doc.get('id')}")
                            if del_resp.status_code == 200:
                                st.success("Deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete document.")

# =============================================================================
# TAB: PLATFORM STATS & DIAGNOSTICS (NEW)
# =============================================================================
with tab_stats:
    st.subheader("📊 Platform Statistics & Diagnostics")
    st.markdown("Real-time telemetry, direct chunk search, and historical query execution trace visualizer.")
    st.markdown("---")
    
    # 1. Platform Statistics
    st.markdown("### 📈 Live Usage Telemetry")
    try:
        stats_resp = requests.get(f"{BACKEND_URL}/stats")
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("total_queries", 0)}</div><div class="metric-label">Total Queries Run</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("avg_latency_ms", 0.0):.2f} ms</div><div class="metric-label">Avg Query Latency</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("total_tokens", 0)}</div><div class="metric-label">Total Token Footprint</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">${stats.get("estimated_cost_usd", 0.0):.5f}</div><div class="metric-label">Est. Provider Cost (USD)</div></div>', unsafe_allow_html=True)
        else:
            st.error("Failed to load statistics from backend.")
    except Exception as e:
        st.error(f"Cannot connect to stats API: {e}")
        
    st.markdown("---")
    
    # 2. Direct Chunk Search
    st.markdown("### 🔍 Direct Chunk Search Debugger")
    st.markdown("Execute a direct similarity query against the document chunks stored in ChromaDB/BM25 without generating an LLM response.")
    
    col_s_query, col_s_strat, col_s_k = st.columns([2, 1, 1])
    with col_s_query:
        search_query = st.text_input("Enter search text", "positional encoding", key="direct_search_query_input")
    with col_s_strat:
        search_strategy = st.selectbox("Search Strategy", ["dense", "sparse", "hybrid"], key="direct_search_strategy")
    with col_s_k:
        search_k = st.number_input("k (Number of chunks)", min_value=1, max_value=20, value=3, key="direct_search_k")
        
    if st.button("🔍 Search Vector Database", key="direct_search_btn"):
        if not search_query.strip():
            st.warning("Please enter search text.")
        else:
            with st.spinner("Searching vector index..."):
                try:
                    params = {
                        "query": search_query,
                        "strategy": search_strategy,
                        "k": search_k
                    }
                    if selected_doc_id:
                        params["doc_id"] = selected_doc_id
                    if chunk_strategy_val:
                        params["chunk_strategy"] = chunk_strategy_val
                    if tags_filter_input:
                        params["tags"] = tags_filter_input
                        
                    s_resp = requests.get(f"{BACKEND_URL}/chunks/search", params=params)
                    if s_resp.status_code == 200:
                        s_results = s_resp.json()
                        st.success(f"Found {len(s_results)} matching chunks.")
                        for idx, item in enumerate(s_results):
                            meta = item.get("metadata", {})
                            score = item.get("score", 0.0)
                            st.markdown(f"**Result {idx+1}:** (Score: `{score:.4f}` | Source: `{meta.get('filename')}` | Strategy: `{meta.get('chunk_strategy')}` | Page: `{meta.get('page', 'N/A')}`)")
                            st.text_area(f"Chunk Content {idx+1}", item.get("content"), height=100, key=f"s_result_{idx}")
                    else:
                        st.error(f"Search failed: {s_resp.text}")
                except Exception as e:
                    st.error(f"Search request failed: {e}")
                    
    st.markdown("---")
    
    # 3. Query History & Step-by-Step Trace Viewer
    st.markdown("### 🛠️ Historical Query Pipeline Trace Viewer")
    st.markdown("Select a query from the database history to inspect its transparent step-by-step pipeline trace.")
    
    try:
        hist_resp = requests.get(f"{BACKEND_URL}/query/history")
        if hist_resp.status_code == 200:
            queries_hist = hist_resp.json()
            if not queries_hist:
                st.info("No query logs found in database. Run some queries in the RAG tab first!")
            else:
                query_options = {
                    f"[{q.get('created_at')}] ID: {q.get('id')[:8]}... - '{q.get('query_text')[:60]}'": q.get("id")
                    for q in queries_hist
                }
                selected_q_label = st.selectbox("Select Past Query", list(query_options.keys()))
                selected_q_id = query_options[selected_q_label]
                
                if st.button("🛠️ Inspect Pipeline Trace Steps", key="inspect_trace_btn"):
                    with st.spinner("Fetching execution trace from DB..."):
                        t_resp = requests.get(f"{BACKEND_URL}/query/{selected_q_id}/pipeline")
                        if t_resp.status_code == 200:
                            trace_steps = t_resp.json()
                            st.success(f"Retrieved {len(trace_steps)} trace steps.")
                            
                            for step in trace_steps:
                                step_name = step.get("step_name")
                                timing = step.get("timing_ms")
                                data = step.get("step_data")
                                
                                with st.expander(f"⚙️ Step: {step_name.replace('_', ' ').title()} ({timing} ms)"):
                                    if step_name == "transformations":
                                        st.write(data)
                                    elif step_name in ["initial_retrieval", "re_ranked_retrieval", "context_assembled"]:
                                        for idx, chunk in enumerate(data):
                                            meta = chunk.get("metadata", {})
                                            st.markdown(f"**Chunk {idx+1}:** (Score/Metrics: `{chunk.get('score', 0.0):.4f}` | Source: `{meta.get('filename')}` | Strategy: `{meta.get('chunk_strategy')}` | Page: `{meta.get('page', 'N/A')}`)")
                                            st.text_area(f"{step_name.title()} Content {idx+1}", chunk.get("content"), height=100, key=f"trace_{step_name}_{selected_q_id}_{idx}")
                                    elif step_name == "prompt":
                                        st.text_area("Formatted System Prompt", data, height=250, key=f"trace_prompt_{selected_q_id}")
                                    else:
                                        st.json(data)
                        else:
                            st.error(f"Failed to fetch trace: {t_resp.text}")
        else:
            st.error("Failed to load query history.")
    except Exception as e:
        st.error(f"Cannot connect to query history API: {e}")
