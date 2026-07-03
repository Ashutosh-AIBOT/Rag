**Assignment \#7** 

**Advanced RAG with** 

**LangChain** 

Build a Production-Grade Retrieval System with Hybrid Search, Re-Ranking, Multi-Strategy Retrieval & RAG Evaluation 

**Organization:** Excellence Technologies Pvt Ltd 

**Phase:** Phase 2 \- LangChain & Advanced RAG 

**Backend:** FastAPI (Python) 

**Frontend:** Next.js (React) 

**Duration:** 4 Days 

**Difficulty:** Advanced 

**Advanced RAG Techniques Covered** 

Hybrid Search (Semantic \+ BM25) | Cross-Encoder Re-Ranking | Parent-Child Chunking Query Expansion | Metadata Filtering | Multi-Vector Retrieval | HyDE Reciprocal Rank Fusion | RAG Evaluation (Faithfulness, Relevancy, Recall) 

This is an advanced assignment. Strong completion of Assignments \#1-6 is expected.  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 2 

 **1\. Assignment Overview** 

In Assignment \#1, you built a basic RAG pipeline: upload documents, chunk, embed, store in a vector DB, retrieve, and generate answers. That was RAG 101\. In this assignment, you will build an Advanced RAG system using LangChain that addresses every limitation of basic RAG: poor retrieval precision, missing context, irrelevant chunks, no handling of complex queries, and no way to evaluate quality. 

You will implement hybrid search (semantic \+ keyword), re-ranking with cross-encoders, multiple chunking strategies (including parent-child), query transformation techniques (expansion, decomposition, HyDE), metadata filtering, Reciprocal Rank Fusion, and a comprehensive RAG evaluation pipeline. This is the assignment where you go from 'it kinda works' to 'it works reliably at production quality.' 

**What You Will Demonstrate** 

\- Deep understanding of LangChain: document loaders, text splitters, retrievers, chains, and LCEL (LangChain Expression Language) 

\- Implementing hybrid search that combines semantic (vector) and lexical (BM25/keyword) retrieval \- Cross-encoder re-ranking to improve retrieval precision after initial retrieval 

\- Advanced chunking strategies: recursive, semantic, parent-child (small chunks for retrieval, large chunks for context) 

\- Query transformation: expansion, decomposition, step-back prompting, and HyDE (Hypothetical Document Embeddings) 

\- Metadata filtering: filter chunks by source, date, section, page number before/during retrieval \- RAG evaluation: measuring faithfulness, answer relevancy, context precision, and context recall \- A/B comparison of retrieval strategies with quantitative metrics 

**Why This Matters** 

In production RAG systems, basic vector similarity search retrieves the right answer only \~60-70% of the time. Advanced techniques (hybrid search \+ re-ranking \+ query expansion) can push this to 85-95%. The difference between a 'cool demo' and a 'production-ready system' is entirely in these techniques. 

 **2\. Problem Statement** 

Build an Advanced RAG Platform where users can: 

**1\.** Upload documents (PDF, TXT, DOCX, Markdown) and process them through multiple chunking strategies simultaneously 

**2\.** Query documents using hybrid search (semantic \+ BM25) with automatic Reciprocal Rank Fusion **3\.** Apply re-ranking to retrieval results using a cross-encoder model before sending to the LLM **4\.** Choose and compare retrieval strategies: basic vector, hybrid, parent-child, multi-query, HyDE **5\.** Filter retrieval by metadata: source document, page number, section headers, date, custom tags **6\.** Evaluate RAG quality: run evaluation queries and see faithfulness, relevancy, precision, and recall scores 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 3 

**7\.** A/B test retrieval strategies: run the same query with two different strategies, compare results side-by-side with scores 

**8\.** Visualize the full RAG pipeline: see exactly which chunks were retrieved, how they were re-ranked, what context was sent to the LLM, and how the answer was generated 

**How This Differs from Assignment \#1** 

 **Aspect Assignment \#1 (Basic RAG) Assignment \#7 (Advanced RAG)**  Chunking Fixed-size character split Recursive, semantic, parent-child  Search Vector similarity only Hybrid (vector \+ BM25) \+ RRF  Re-ranking None Cross-encoder re-ranking 

 Query handling Direct query Expansion, decomposition, HyDE  Filtering None Metadata filters (source, page, date)  Evaluation None Faithfulness, relevancy, precision, recall  Framework Manual pipeline LangChain LCEL with composable chains  Transparency Black box Full pipeline visualization 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 4 

 **3\. Technical Requirements** 

**3.1 Document Ingestion with LangChain** 

Build a flexible, multi-strategy document ingestion pipeline using LangChain: 

\- **Document Loaders:** Use LangChain document loaders: PyPDFLoader (PDF), TextLoader (TXT), UnstructuredWordDocumentLoader (DOCX), UnstructuredMarkdownLoader 

(Markdown). Each loader extracts text with metadata (source, page\_number, file\_type). 

\- **Metadata Enrichment:** After loading, enrich each document with metadata: file name, file type, total pages, upload date, file size, user-defined tags. For PDFs, extract section headers (H1/H2) 

and attach as metadata to relevant chunks. 

\- **Multiple Chunking Strategies (run simultaneously):** Store chunks from ALL strategies in the vector DB with a 'strategy' metadata field. This allows A/B comparison 

at query time. 

Implement the following chunking strategies: 

\- **(a) Recursive Character Splitting:** LangChain's RecursiveCharacterTextSplitter. Chunk size 500 tokens, overlap 50 tokens. Splits on paragraph \-\> sentence \-\> word boundaries 

for clean breaks. 

\- **(b) Semantic Chunking:** Split based on embedding similarity. Consecutive sentences with similar embeddings stay together; a new chunk starts when similarity drops below a 

threshold. Produces semantically coherent chunks. 

\- **(c) Parent-Child Chunking:** Create small chunks (200 tokens) for retrieval precision. Each small chunk links to its parent chunk (1000 tokens) which provides full context. Retrieve by small 

chunk, but send parent chunk to LLM. 

\- **(d) Section-Based Chunking:** Split documents by detected section headers (H1, H2). Each section is a chunk. Best for structured documents like reports, manuals, and legal docs. 

**3.2 Hybrid Search (Semantic \+ BM25)** 

Implement a hybrid retrieval system combining two search paradigms: 

\- **Semantic Search (Dense):** Standard vector similarity search using embeddings (sentence-transformers all-MiniLM-L6-v2 or OpenAI text-embedding-3-small). Finds conceptually similar 

content even with different wording. 

\- **Keyword Search (Sparse/BM25):** BM25 lexical search using rank\_bm25 library or Elasticsearch. Finds exact keyword matches that semantic search might miss (e.g., specific product 

codes, names, numbers). 

\- **Reciprocal Rank Fusion (RRF):** Combine results from both searches using RRF: score \= sum(1 / (k \+ rank\_i)) for each document across all result lists. k=60 is standard. This 

produces a merged, ranked list that leverages both search types. 

\- **Weight Configuration:** Allow users to adjust the semantic vs. keyword weight (e.g., 70% semantic \+ 30% Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 5 

BM25, or 50/50). Different document types benefit from different ratios. 

\- **LangChain EnsembleRetriever:** Use LangChain's EnsembleRetriever to implement this cleanly. It handles combining multiple retrievers with configurable weights. 

**Why Hybrid Search?** 

Semantic search fails on exact matches: searching for 'error code E-4021' may return results about error handling in general. BM25 finds the exact code. Conversely, BM25 fails on conceptual queries: 'how to fix authentication issues' may not match a document that says 'resolve login problems.' Hybrid combines both strengths. 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 6 

**3.3 Cross-Encoder Re-Ranking** 

After initial retrieval, re-rank results using a more powerful model: 

\- **Initial Retrieval:** Retrieve top-20 chunks using hybrid search (fast but imprecise). 

\- **Re-Ranking:** Pass query \+ each retrieved chunk through a cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers). The cross-encoder scores query-document relevance much more accurately than bi-encoder similarity. 

\- **Final Top-K:** After re-ranking, take the top-5 chunks (re-ranked) to send to the LLM. This dramatically improves precision. 

\- **Score Transparency:** Show the re-ranking scores in the UI. Display: original rank, re-ranked position, relevance score. This helps users understand why certain chunks were selected. 

\- **Cohere Re-Rank (Alternative):** Alternatively, use Cohere's Rerank API (has a free tier) for re-ranking. This is easier to set up than running a local cross-encoder model. 

**3.4 Query Transformation Techniques** 

Implement multiple strategies to improve query effectiveness: 

\- **Multi-Query Expansion:** Use the LLM to generate 3-5 alternative phrasings of the user's question. Run retrieval for each phrasing and merge results (union \+ deduplicate). LangChain's 

MultiQueryRetriever does this. 

\- **Query Decomposition:** For complex questions, break them into sub-questions. E.g., 'Compare the pricing and features of Plan A vs Plan B' becomes: (1) 'What are the features of Plan A?', (2) 

'What are the features of Plan B?', (3) 'What is the pricing of Plan A?', (4) 'What is the 

pricing of Plan B?'. Retrieve for each sub-question, then synthesize. 

\- **Step-Back Prompting:** For specific questions, generate a more general 'step-back' question first. E.g., specific: 'What was revenue in Q3 2025?' \-\> step-back: 'What are the financial results 

for 2025?'. Retrieve context for both, improving recall. 

\- **HyDE (Hypothetical Document Embeddings):** Given a query, use the LLM to generate a hypothetical answer. Embed the hypothetical answer (not the query) and 

search with that embedding. This often retrieves more 

relevant chunks because the hypothetical answer is closer in 

embedding space to real answers. 

**3.5 Metadata Filtering** 

Allow precise filtering before and during retrieval: 

\- **Source Filter:** Filter by source document (e.g., 'only search in annual\_report\_2025.pdf'). Critical when multiple documents are ingested. 

\- **Page Range:** Filter by page number range (e.g., 'only pages 10-25'). Useful for large documents. \- **Section Filter:** Filter by section header (e.g., 'only the Financial Summary section'). 

\- **Date Filter:** Filter by document upload date or document date (if extracted from content). Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 7 

\- **Tag Filter:** Filter by user-defined tags assigned during upload (e.g., 'legal', 'finance', 'technical'). \- **Chunk Strategy Filter:** Filter by chunking strategy (e.g., 'only use parent-child chunks' or 'only semantic chunks'). 

\- **Combined Filters:** Support AND/OR combinations: source='report.pdf' AND section='Financial' AND strategy='parent-child'. 

**3.6 RAG Evaluation Pipeline** 

Build a comprehensive evaluation system to measure RAG quality: 

\- **Faithfulness:** Does the answer contain only information supported by the retrieved context? Score 0-1. Detect hallucinated facts not present in the context. Use LLM-as-judge: extract claims from the answer, check each claim against the context. 

\- **Answer Relevancy:** Is the answer actually relevant to the question? Score 0-1. An answer can be faithful but irrelevant (answering a different question). Generate synthetic questions from the 

answer, measure similarity to original question. 

\- **Context Precision:** Were the retrieved chunks actually relevant to answering the question? Rank-aware metric: relevant chunks should appear at higher ranks. Score 0-1. 

\- **Context Recall:** Did the retrieval find all the information needed to answer the question? Requires reference answers for evaluation. Compare retrieved context against reference to check coverage. 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 8 

\- **Evaluation Dataset:** Create a test dataset of 20-30 question-answer pairs with reference answers and relevant source chunks. Use this to benchmark different retrieval strategies. 

\- **Strategy Comparison:** Run the evaluation dataset against all retrieval strategies (basic vector, hybrid, hybrid+rerank, parent-child, HyDE). Generate a comparison table showing scores for 

each strategy. 

\- **RAGAS Integration:** (Good-to-Have) Integrate the RAGAS library (Retrieval Augmented Generation Assessment) for standardized evaluation metrics. 

**3.7 Pipeline Visualization & Transparency** 

Make the RAG pipeline fully transparent to the user: 

\- **Pipeline Steps Display:** For every query, show the complete pipeline: Original Query \-\> Query Transformation \-\> Retrieved Chunks (with scores) \-\> Re-Ranked Chunks (with new 

scores) \-\> Context Assembly \-\> LLM Prompt \-\> Generated Answer. 

\- **Chunk Inspector:** Click any retrieved chunk to see: source document, page number, chunk text, embedding similarity score, BM25 score, re-rank score, strategy used. 

\- **Context Window View:** Show exactly what context was sent to the LLM (the final assembled prompt with system message \+ context chunks \+ user question). 

\- **Token Accounting:** Show token counts: input tokens (context \+ query), output tokens (answer), total cost. Break down context tokens by source chunk. 

\- **Retrieval Debugger:** When an answer is wrong or incomplete, users can inspect why: were the right chunks retrieved? Were they ranked high enough? Was the context window too small? 

**3.8 Backend API (FastAPI)** 

 **Method Endpoint Description** 

 POST /api/documents/upload Upload & ingest document (all chunk strategies)  GET /api/documents List ingested documents with metadata 

 DELETE /api/documents/{id} Delete document and all its chunks 

 POST /api/query Query with configurable strategy & filters  POST /api/query/compare A/B test: same query, two strategies 

 GET /api/query/{id}/pipeline Get full pipeline trace for a query 

 GET /api/query/{id}/chunks Get retrieved chunks with all scores 

 POST /api/evaluate Run evaluation on a question-answer pair  POST /api/evaluate/batch Run evaluation on entire test dataset 

 GET /api/evaluate/results Get evaluation results and comparisons  GET /api/strategies List available retrieval strategies 

 GET /api/chunks/search Direct chunk search (debug endpoint) 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 9 

 GET /api/stats Retrieval stats, token usage, query latency  GET /api/health Health check 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 10 

 **4\. Recommended Tech Stack** 

**Backend (FastAPI \+ LangChain)** 

\- **Framework:** FastAPI with async support 

\- **RAG Framework:** LangChain (langchain, langchain-community, langchain-openai or langchain-google-genai) \- **LCEL:** LangChain Expression Language for composable, declarative chain construction \- **Vector Database:** ChromaDB (recommended for simplicity) or Qdrant or FAISS 

\- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2) for local embeddings, or OpenAI text-embedding-3-small 

\- **BM25:** rank\_bm25 library for keyword search 

\- **Re-Ranker:** sentence-transformers CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2) or Cohere Rerank API (free tier) 

\- **LLM:** Google Gemini (free tier), Groq (free tier), or OpenAI GPT 

\- **Document Loaders:** LangChain loaders: PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader 

\- **Text Splitters:** LangChain: RecursiveCharacterTextSplitter, SemanticChunker (experimental), ParentDocumentRetriever 

\- **Evaluation:** Custom evaluation pipeline, or RAGAS library (bonus) 

\- **Database:** SQLite for query history, evaluation results, document metadata 

**Frontend (Next.js)** 

\- **Framework:** Next.js 14+ with TypeScript 

\- **Styling:** Tailwind CSS \+ shadcn/ui components 

\- **Charts:** Recharts for evaluation metric visualization 

\- **Diff View:** react-diff-viewer for A/B comparison of answers 

\- **Code Display:** react-syntax-highlighter for showing LLM prompts and pipeline details 

\- **State:** Zustand or React Context 

**LangChain Version** 

LangChain evolves rapidly. Use langchain \>= 0.2 with the new modular package structure (langchain-core, langchain-community, langchain-openai, etc.). Do NOT use deprecated imports. Check the latest LangChain documentation for current import paths. 

 **5\. Expected Project Structure** 

`advanced-rag-platform/` 

`|` 

`|-- backend/` 

`| |-- app/` 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 11 

`| | |-- main.py # FastAPI entry` 

`| | |-- config.py # Settings` 

`| | |-- routers/` 

`| | | |-- documents.py # Document upload & management` 

`| | | |-- query.py # Query & comparison endpoints` 

`| | | |-- evaluation.py # Evaluation endpoints` 

`| | |-- services/` 

`| | | |-- ingestion.py # Document loading & chunking` 

`| | | |-- chunking_strategies.py # All 4 chunking implementations` 

`| | | |-- retrieval.py # Hybrid search, retrievers` 

`| | | |-- reranker.py # Cross-encoder re-ranking` 

`| | | |-- query_transform.py # Multi-query, decompose, HyDE` 

`| | | |-- rag_chain.py # LangChain LCEL RAG chains` 

`| | | |-- evaluator.py # RAG evaluation pipeline` 

`| | | |-- pipeline_tracer.py # Pipeline step logging` 

`| | |-- models/` 

`| | | |-- database.py # SQLAlchemy models` 

`| | | |-- schemas.py # Pydantic models` 

`| | |-- data/` 

`| | | |-- eval_dataset.json # Evaluation Q&A pairs` 

`| |-- requirements.txt` 

`| |-- Dockerfile` 

`|` 

`|-- frontend/` 

`| |-- src/` 

`| | |-- app/` 

`| | | |-- page.tsx # Main query page` 

`| | | |-- documents/page.tsx # Document management` 

`| | | |-- compare/page.tsx # A/B comparison` 

`| | | |-- evaluate/page.tsx # Evaluation dashboard` 

`| | |-- components/` 

`| | | |-- QueryPanel.tsx # Query input + strategy selector` 

`| | | |-- AnswerDisplay.tsx # Answer + source citations` 

`| | | |-- PipelineVisualizer.tsx # Full pipeline steps view` 

`| | | |-- ChunkInspector.tsx # Chunk detail + scores` 

`| | | |-- ComparisonView.tsx # A/B strategy comparison` 

`| | | |-- EvalDashboard.tsx # Evaluation metrics charts` 

`| | | |-- MetadataFilters.tsx # Filter controls` 

`| | | |-- DocumentList.tsx # Uploaded docs management` 

`| | | |-- StrategySelector.tsx # Retrieval strategy picker` 

`| |-- package.json` 

`|` 

`|-- sample_documents/ # Test documents (include)` 

`|-- eval_dataset/ # Evaluation Q&A pairs` 

`|-- docker-compose.yml` 

`|-- .env.example` 

`|-- .gitignore` 

`|-- README.md` 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 12 

 **6\. Retrieval Strategies (Must Implement All)** 

You must implement all of the following retrieval strategies. Users select which strategy to use per query, or run A/B comparisons between two strategies. 

 **Strategy LangChain Component How It Works** 

 Basic Vector Chroma.as\_retriever() Standard cosine similarity search on embeddings  Hybrid (BM25+Vec) EnsembleRetriever BM25 \+ vector search merged via RRF or weighted  Hybrid \+ Rerank EnsembleRetriever \+ Reranker Hybrid retrieval \-\> cross-encoder re-ranking top-k  Parent-Child ParentDocumentRetriever Search small chunks, return parent (larger) chunks  Multi-Query MultiQueryRetriever LLM generates 3-5 query variants, merge results  HyDE Custom chain LLM generates hypothetical answer, embed & search  Decomposition Custom chain Break complex query into sub-queries, search each 

Each strategy should be selectable in the UI via a dropdown/radio button. The default should be 'Hybrid \+ Rerank' as it generally performs best. 

 **7\. UI Screens & Layout** 

**Screen 1: Query Interface (Main)** 

\- Top: Query input with strategy selector dropdown and 'Ask' button 

\- Below query: Metadata filter panel (collapsible) \- source, page range, section, tags, chunk strategy \- Left panel: Answer display with markdown rendering, source citations with chunk references \- Right panel: Retrieved chunks list with: text preview, source, page, similarity score, BM25 score, re-rank score, rank badges (\#1, \#2, etc.) 

\- Bottom: Pipeline visualization showing each step as a flowchart node with timing 

**Screen 2: Pipeline Visualizer (Expandable Panel)** 

\- Step 1: Original Query \-\> (if transformed) Step 2: Transformed Queries 

\- Step 3: Initial Retrieval (show top-20 with scores) \-\> Step 4: Re-Ranking (show score changes) \- Step 5: Final Context Assembly (show token count) \-\> Step 6: LLM Generation (show full prompt) \- Each step is expandable to see full details. Collapsible by default for clean UI. 

\- Color-coded: green for good scores, yellow for medium, red for low relevance 

**Screen 3: A/B Strategy Comparison** 

\- Query input at top, two strategy selectors (Strategy A, Strategy B) 

\- 'Compare' button runs the same query with both strategies 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 13 

\- Side-by-side results: Answer A vs Answer B 

\- Below: chunk comparison \- which chunks each strategy retrieved, overlap highlighted 

\- Metrics comparison table: latency, token usage, faithfulness, relevancy scores 

**Screen 4: Evaluation Dashboard** 

\- Run evaluation button: processes all Q\&A pairs in the eval dataset 

\- Bar chart comparing all strategies across 4 metrics (faithfulness, relevancy, precision, recall) \- Per-question results table: question, expected answer, actual answer, scores, pass/fail 

\- Strategy leaderboard: ranked list of strategies by average score 

\- Failure analysis: list of questions where the system scored below threshold, with the pipeline trace for debugging 

**Screen 5: Document Management** 

\- Upload documents with tag assignment and chunking preview 

\- List of ingested documents with: name, pages, chunks created (per strategy), upload date \- Click document to see all its chunks, grouped by strategy, with chunk text previews 

\- Delete document (removes all chunks and vectors) 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 14 

 **8\. Feature Breakdown & Priority** 

**Must-Have (Core \- Required)** 

**1\.** LangChain-based document ingestion with at least 3 chunking strategies (recursive, parent-child, section-based) 

**2\.** Vector store (ChromaDB) with metadata storage per chunk (source, page, strategy, section) **3\.** Hybrid search combining semantic (vector) \+ keyword (BM25) with EnsembleRetriever **4\.** Reciprocal Rank Fusion (RRF) for merging hybrid search results 

**5\.** Cross-encoder re-ranking of top-20 results down to top-5 (sentence-transformers or Cohere) **6\.** Multi-Query retrieval: LLM generates query variants, merge results (MultiQueryRetriever) **7\.** HyDE implementation: generate hypothetical answer, embed, search 

**8\.** Metadata filtering: filter by source, page range, chunk strategy, user tags 

**9\.** RAG chain built with LangChain LCEL (not legacy chain classes) 

**10\.** Pipeline transparency: show retrieved chunks, scores, re-rank positions for every query **11\.** A/B strategy comparison: same query, two strategies, side-by-side results 

**12\.** Basic RAG evaluation: faithfulness and answer relevancy scoring (LLM-as-judge) 

**13\.** Evaluation dataset with at least 15 question-answer pairs included in repo 

**14\.** Next.js frontend with query UI, chunk inspector, strategy selector, and pipeline visualizer **15\.** Streaming LLM responses via SSE 

**Good-to-Have (Intermediate)** 

**1\.** Semantic chunking (split based on embedding similarity thresholds) 

**2\.** Query decomposition for complex multi-part questions 

**3\.** Step-back prompting for specific queries 

**4\.** Context precision and context recall evaluation metrics 

**5\.** Evaluation dashboard with charts comparing all strategies across all metrics 

**6\.** Failure analysis: identify and explain queries where RAG failed 

**7\.** Token accounting: show input/output tokens, cost breakdown per query 

**8\.** Configurable hybrid search weights (semantic vs. BM25 ratio) 

**9\.** Chunk overlap visualization: show how chunks overlap in the source document 

**10\.** Query history with ability to re-run and compare results over time 

**Bonus (Advanced)** 

**1\.** RAGAS library integration for standardized evaluation metrics 

**2\.** Contextual compression: use LLM to extract only relevant sentences from retrieved chunks before sending to the main LLM 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 15 

**3\.** Multi-vector retrieval: create summary \+ questions for each chunk, retrieve by multiple representations **4\.** Auto-strategy selection: based on query type, automatically pick the best retrieval strategy **5\.** LangSmith integration for tracing (or custom tracing with similar UI) 

**6\.** Embeddings comparison: try different embedding models, compare retrieval quality 

**7\.** Docker \+ docker-compose deployment 

**8\.** Evaluation CI pipeline: run eval dataset on every code change, fail if metrics drop 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 16 

 **9\. Evaluation Criteria** 

 **Criteria Weight What We Look For** 

 RAG Quality 30% Retrieval accuracy, answer quality, hybrid search working, re-ranking improves results  LangChain Usage 20% Proper LCEL, correct retrievers/splitters, not reinventing what LangChain provides  Evaluation Pipeline 15% Evaluation metrics implemented, dataset provided, strategy comparison works  Code Quality 15% Clean FastAPI \+ LangChain integration, modular services, typed Pydantic models  UI/UX 10% Pipeline visualizer, chunk inspector, A/B comparison, evaluation charts  Documentation 5% Architecture diagram, strategy explanations, eval results in README  Bonus Features 5% RAGAS, contextual compression, auto-strategy, LangSmith, Docker 

**What Impresses Us** 

\- Re-ranking that measurably improves retrieval precision (show before/after scores) 

\- Evaluation results showing clear winner among strategies with quantitative evidence 

\- Pipeline visualization that makes the RAG process fully transparent and debuggable 

\- Proper use of LangChain LCEL with composable, readable chain definitions 

\- Parent-child chunking that demonstrably retrieves better context than fixed-size chunks 

\- HyDE that noticeably improves retrieval for abstract or conceptual queries 

\- A well-curated evaluation dataset with diverse question types and difficulty levels 

**Common Mistakes to Avoid** 

\- Using deprecated LangChain APIs (LLMChain, SequentialChain, etc.) instead of LCEL 

\- Not actually implementing hybrid search (just using vector search and calling it 'hybrid') 

\- Re-ranking that does not change the order of results (misconfigured cross-encoder) 

\- No evaluation dataset: cannot prove any strategy is better without quantitative comparison \- Parent-child retriever that returns child chunks instead of parent chunks to the LLM 

\- Ignoring metadata during ingestion: chunks without source/page metadata cannot be filtered \- Black-box RAG with no pipeline transparency: users cannot debug why answers are wrong 

 **10\. Submission Guidelines** 

**What to Submit** 

**1\.** GitHub Repository \- Full commit history. Include sample\_documents/ (3-5 test docs) and eval\_dataset/ (15+ Q\&A pairs). 

**2\.** README.md \- Architecture diagram showing the full RAG pipeline, explanation of each strategy, evaluation Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 17 

results table, screenshots of pipeline visualizer and A/B comparison, setup instructions. 

**3\.** .env.example \- All required API keys. 

**4\.** Demo Video (Recommended) \- 5-7 min: ingest document, query with different strategies, show pipeline visualization, run A/B comparison, show evaluation dashboard with metrics. 

**Submission Deadline** 

Submit within 4 days of receiving this assignment. 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 18 

 **11\. Getting Started Guide** 

**Day 1: LangChain Ingestion & Chunking** 

\- Initialize FastAPI project, install langchain, chromadb, sentence-transformers, rank-bm25 \- Build document loaders for PDF, TXT, DOCX, Markdown using LangChain loaders 

\- Implement RecursiveCharacterTextSplitter with metadata enrichment 

\- Implement parent-child chunking using LangChain's ParentDocumentRetriever 

\- Implement section-based chunking (split on headers) 

\- Set up ChromaDB with metadata filtering support 

\- Test ingestion with sample documents, verify chunks in ChromaDB 

**Day 2: Retrieval Strategies & Re-Ranking** 

\- Implement basic vector retriever (ChromaDB as\_retriever) 

\- Implement BM25 retriever using rank\_bm25 

\- Build hybrid search with LangChain EnsembleRetriever (vector \+ BM25) 

\- Implement cross-encoder re-ranking (sentence-transformers or Cohere Rerank) 

\- Implement MultiQueryRetriever for query expansion 

\- Implement HyDE chain (LLM \-\> hypothetical answer \-\> embed \-\> search) 

\- Build the main RAG chain using LCEL with configurable strategy selection 

\- Add metadata filtering to all retrievers 

\- Build pipeline tracer that logs every step with timing and scores 

**Day 3: Evaluation & Frontend** 

\- Build RAG evaluation pipeline (faithfulness, answer relevancy using LLM-as-judge) 

\- Create evaluation dataset (15+ Q\&A pairs with reference answers) 

\- Build batch evaluation endpoint that tests all strategies 

\- Initialize Next.js project with Tailwind \+ shadcn/ui 

\- Build query interface with strategy selector and metadata filters 

\- Build pipeline visualizer component (step-by-step flow) 

\- Build chunk inspector (click chunk to see scores and details) 

\- Build A/B comparison page 

**Day 4: Evaluation Dashboard, Polish & Submit** 

\- Build evaluation dashboard with charts (Recharts) 

\- Connect all frontend to backend, test full query flow with each strategy 

\- Run full evaluation, include results in README 

\- Add streaming responses (SSE) 

Confidential \- For Internal Use Only  
Assignment \#7: Advanced RAG with LangChain | Excellence Technologies Page 19 

\- Test with diverse queries: factual, conceptual, multi-hop, specific (IDs/numbers) 

\- Write README with architecture diagram and evaluation results table 

\- Record demo video and submit 

 **12\. Helpful Resources** 

**LangChain** 

\- LangChain Docs: python.langchain.com/docs 

\- LCEL Guide: python.langchain.com/docs/concepts/lcel 

\- Retrievers: python.langchain.com/docs/how\_to/\#retrievers 

\- ParentDocumentRetriever: search 'LangChain ParentDocumentRetriever' 

\- EnsembleRetriever: search 'LangChain EnsembleRetriever hybrid search' 

\- MultiQueryRetriever: search 'LangChain MultiQueryRetriever' 

**Advanced RAG Concepts** 

\- HyDE Paper: search 'Precise Zero-Shot Dense Retrieval without Relevance Labels' 

\- Cross-Encoder Re-Ranking: sbert.net/examples/applications/cross-encoder/ 

\- Reciprocal Rank Fusion: search 'RRF Reciprocal Rank Fusion information retrieval' 

\- RAG evaluation: search 'RAGAS RAG evaluation metrics' 

**Libraries** 

\- ChromaDB: docs.trychroma.com 

\- sentence-transformers: sbert.net 

\- rank\_bm25: pypi.org/project/rank-bm25/ 

\- Cohere Rerank: docs.cohere.com/docs/reranking (free tier) 

\- RAGAS: docs.ragas.io (evaluation library) 

 **13\. Questions?** 

This is the most technically deep assignment so far. It is not about building a pretty UI \- it is about building a RAG system that genuinely works well, and proving it with evaluation metrics. The evaluation results in your README are as important as the code itself. If you have questions about requirements, reach out. 

Good luck\! Build the RAG system you wish you had for Assignment \#1. 

\--- End of Assignment \#7 \--- 

Confidential \- For Internal Use Only