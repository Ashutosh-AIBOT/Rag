# 🔍 Advanced Enterprise RAG Studio

A production-grade, state-of-the-art Retrieval-Augmented Generation (RAG) platform built with **FastAPI**, **LangChain (LCEL)**, and **Next.js (App Router)** with **Tailwind CSS**.

---

## 🏗️ System Architecture

The platform implements a highly modular, decoupled RAG pipeline where every execution step (query transformation, retrieval, re-ranking, and response generation) is recorded as a structured **Pipeline Trace** for full transparency.

```mermaid
graph TD
    User([User/Client]) -->|1. Query Request| API[FastAPI Gateway]
    
    subgraph Ingestion Pipeline [Multi-Strategy Ingestion]
        Loader[PDF/TXT Loader] --> Splitters{Splitters}
        Splitters -->|Recursive Splitting| Rec[Recursive Chunks]
        Splitters -->|Semantic Distance| Sem[Semantic Chunks]
        Splitters -->|Parent-Child Relation| PC[Parent-Child Map]
        Splitters -->|Heading Hierarchy| Sec[Section Chunks]
        
        Rec & Sem & PC & Sec --> Store[ChromaDB Vector Store]
        Rec & Sem & PC & Sec --> BM25[BM25 Sparse Index]
    end

    subgraph Retrieval Orchestrator [Advanced Retrieval]
        API --> SearchServ[Search Service]
        SearchServ -->|Strategy: Hybrid| Hybrid[Ensemble Vector + BM25]
        SearchServ -->|Strategy: Multi-Query| MQ[Multi-Query Expansion]
        SearchServ -->|Strategy: HyDE| HyDE[Hypothetical Document]
        SearchServ -->|Strategy: Step-Back| SB[Step-Back Prompting]
        SearchServ -->|Strategy: Decomposition| Dec[Decomposition Chain]
        
        Hybrid & MQ & HyDE & SB & Dec --> Rerank[Cross-Encoder Re-ranking]
    end

    subgraph LLM Synthesis [LCEL Chain]
        Rerank -->|Top-K Context| LCEL[LangChain LCEL Chain]
        LCEL -->|Streaming SSE| User
    end

    subgraph Evaluation Engine
        LCEL -->|Faithfulness| Faith[LLM Faithfulness Judge]
        LCEL -->|Relevancy| Relev[LLM Relevancy Judge]
        LCEL -->|Precision| Prec[Context Precision Judge]
        LCEL -->|Recall| Reca[Context Recall Judge]
    end
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- Anaconda / Miniconda (recommended)

### 2. Environment Variables Configuration
Configure a `.env` file in the `backend/` directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_api_key

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=rag_documents
DATABASE_URL=./rag_database.db
```

### 3. Run the Backend Server
```bash
cd backend
conda activate rag
/home/creator/miniforge3/envs/rag/bin/python -m uvicorn app.main:app --reload --port 8000
```

### 4. Run the Frontend Dev Studio
```bash
cd frontend
npm run dev
```
Access the dashboard at `http://localhost:3000`.

---

## 🛠️ Feature Walkthrough

### 1. Ingestion & Multi-Strategy Chunking
Upload PDFs or plain text. During upload, the documents are concurrently split into four distinct representations:
- **Recursive Character**: Standard splitter capturing raw document flow.
- **Semantic Splitter**: Uses embedding cosine-similarity boundaries to split text on shift of meaning.
- **Parent-Child**: Index small child chunks for high-precision retrieval but return larger parent contexts.
- **Section Splitting**: Splits text based on headings and paragraph hierarchy.

### 2. Hybrid & Ensemble Retrieval
Combines dense retriever (Chroma DB similarity search) and sparse retriever (BM25 token-matching index) using **Weighted Reciprocal Rank Fusion (RRF)**. The results are re-ranked using a transformer **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to select the top-5 most relevant chunks.

### 3. LLM-as-a-Judge Evaluation Dashboard
An evaluation pipeline measuring four key metrics:
- **Faithfulness**: Is the answer derived *only* from the context?
- **Answer Relevancy**: Does the answer directly address the user's question?
- **Context Precision**: Did the retriever rank relevant chunks higher?
- **Context Recall**: Does the retrieved context contain all facts required to answer?
