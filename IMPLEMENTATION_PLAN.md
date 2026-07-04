06.Rag/
├── docker-compose.yml           # Top-level Docker Compose orchestrator
├── README.md                    # System architecture & benchmark documentation
├── .env.example                 # Environment variables template
├── .gitignore                   # Version control exclusions
│
├── backend/                     # FastAPI Backend Application
│   ├── Dockerfile               # Backend container configuration
│   ├── requirements.txt         # Declared python dependencies
│   ├── rag_database.db          # Active SQLite persistent store
│   │
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint & middleware routing
│   │   ├── config.py            # Pydantic Settings management
│   │   │
│   │   ├── core/                # Core lifecycle and database managers
│   │   │   ├── lifespan.py      # App lifespan setup (LLM/embeddings singletons)
│   │   │   ├── startup.py       # Auto-migration schema validator
│   │   │   └── logging.py       # Central logger configurations
│   │   │
│   │   ├── database/            # Database query methods
│   │   │   └── database.py      # SQLite CRUD handlers
│   │   │
│   │   ├── models/              # Pydantic models for validation
│   │   │   └── schemas.py       # API schemas
│   │   │
│   │   ├── embeddings/          # Embedding model loaders
│   │   │   └── sentence_transformer.py
│   │   │
│   │   ├── llm/                 # LLM provider configurations
│   │   │   ├── base.py          # Unified LLM wrappers
│   │   │   └── manager.py       # LLM provider fallback routing
│   │   │
│   │   ├── vectorstore/         # Vector store connections
│   │   │   └── chroma.py        # ChromaDB setup and bindings
│   │   │
│   │   ├── routers/             # API Router Endpoints
│   │   │   ├── documents.py     # Ingestion & document stats endpoints
│   │   │   ├── query.py         # Strategy execution & comparison
│   │   │   ├── evaluation.py    # Batch evaluation pipeline
│   │   │   ├── stats.py         # Latency & direct search debugging
│   │   │   ├── jobs.py          # Background worker tasks status checker
│   │   │   ├── rerank.py        # Re-ranking logic endpoint
│   │   │   └── health.py        # Service health check
│   │   │
│   │   ├── services/            # LCEL chains and chunkers
│   │   │   ├── ingestion.py          # Loading, splitting, and storing pipeline
│   │   │   ├── chunking_strategies.py # Parallel strategy splitter
│   │   │   ├── loaders.py            # File loader mappings
│   │   │   ├── semantic_chunker.py   # Embedding similarity splitter
│   │   │   ├── section_chunker.py    # Section boundary splits
│   │   │   ├── parent_child_chunker.py # Small/large chunk context generator
│   │   │   ├── retrieval.py          # Vector query & parent resolution
│   │   │   ├── bm25_retriever.py     # persistant keyword indexing
│   │   │   ├── hybrid_retriever.py   # Ensemble & RRF fusion
│   │   │   ├── reranker.py           # MiniLM Cross-Encoder re-ranking
│   │   │   ├── query_transform.py    # Multi-Query, Decomposition, Step-Back
│   │   │   ├── rag_chain.py          # Main generation chains
│   │   │   ├── evaluator.py          # RAGAS metrics & custom evaluators
│   │   │   ├── scoring.py            # Fact recall & precision scoring
│   │   │   └── validators.py         # File validator utilities
│   │   │
│   │   └── data/
│   │       └── eval_dataset.json     # Primary dataset location (25 Q&A pairs)
│   │
│   └── scripts/
│       └── run_eval_ci.py       # Local CI execution benchmark script
│
├── frontend/                    # Next.js Frontend Application
│   ├── Dockerfile               # Frontend container configuration
│   ├── package.json             # Next.js dependencies
│   ├── tsconfig.json            # TypeScript configurations
│   ├── next.config.ts           # Next.js configurations
│   │
│   └── src/
│       ├── components/          # Reusable UI controls
│       │   ├── Navbar.tsx            # Navigation bar
│       │   ├── QueryPanel.tsx        # Query search panel
│       │   ├── AnswerDisplay.tsx     # Citation rendering engine
│       │   ├── PipelineVisualizer.tsx # Flow diagram trace
│       │   ├── ChunkInspector.tsx    # Chunk overlap inspector
│       │   └── MetadataFilters.tsx   # Searchable filters
│       │
│       └── app/                 # Next.js App Router directories
│           ├── layout.tsx       # Root layout file
│           ├── page.tsx         # Main chat window
│           ├── globals.css      # Core tailwind styles
│           ├── compare/         # Side-by-side strategy comparison panel
│           │   └── page.tsx
│           ├── documents/       # Document management page
│           │   └── page.tsx
│           └── evaluate/        # Evaluation charts & metrics dashboard
│               └── page.tsx
│
├── sample_documents/            # Ingestion verification source files
│   └── NIPS-2017-attention-is-all-you-need-Paper.pdf
│
├── eval_dataset/                # Evaluation benchmark files
│   └── eval_dataset.json
│
└── Testing-data/                # Raw test source repository
    └── NIPS-2017-attention-is-all-you-need-Paper.pdf
