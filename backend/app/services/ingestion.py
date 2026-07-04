from pathlib import Path
from langchain_core.runnables import RunnableLambda
from app.core.logging import get_logger
from app.services.loaders import load_chain
from app.services.chunking_strategies import all_strategies_chain
from app.database.database import get_document_by_filename, update_document_chunk_count, update_document_status, insert_parent_document, update_document_pages
from app.vectorstore.chroma import add_documents_to_chroma
from app.models.schemas import IngestionResult

logger = get_logger(__name__)


def generate_summary_and_questions(text: str) -> tuple[str, list[str]]:
    try:
        from app.llm import get_llm_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template(
            "Generate a one-sentence summary and three hypothetical questions that this text answers. Format the output exactly as follows:\n"
            "Summary: [One sentence summary]\n"
            "Questions:\n"
            "1. [Question 1]\n"
            "2. [Question 2]\n"
            "3. [Question 3]\n\n"
            "Text: {text}"
        )
        chain = prompt | get_llm_chain() | StrOutputParser()
        res = chain.invoke({"text": text})
        
        summary = ""
        questions = []
        lines = res.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("summary:"):
                summary = line[8:].strip()
            elif line.lower().startswith("1.") or line.lower().startswith("2.") or line.lower().startswith("3."):
                q = line[2:].strip()
                if q:
                    questions.append(q)
        return summary, questions
    except Exception as e:
        logger.error(f"Failed to generate summary/questions: {e}")
        return "", []


def _validate(data: dict) -> dict:
    file_path = data["file_path"]
    doc_id = data["doc_id"]
    existing = get_document_by_filename(Path(file_path).name)
    if existing:
        data["doc_id"] = existing["id"]
        data["duplicate"] = True
        return data
    data["duplicate"] = False
    return data


def _store(data: dict) -> IngestionResult:
    chunks_result = data["chunks"]
    doc_id = data["doc_id"]
    file_path = data["file_path"]
    filename = Path(file_path).name

    all_chunks = []
    for strategy_name, chunks in chunks_result.items():
        if strategy_name == "parent_mapping":
            continue
        if isinstance(chunks, dict):
            if "child_chunks" in chunks:
                all_chunks.extend(chunks["child_chunks"])
            continue
        all_chunks.extend(chunks)

    import datetime
    upload_date = datetime.date.today().isoformat()

    texts = [chunk.page_content for chunk in all_chunks]
    metadatas = [
        {
            "source": filename,
            "page": chunk.metadata.get("page", 0),
            "strategy": chunk.metadata.get("strategy", "recursive"),
            "doc_id": doc_id,
            "section": chunk.metadata.get("section", ""),
            "parent_id": chunk.metadata.get("parent_id", ""),
            "upload_date": upload_date,
        }
        for chunk in all_chunks
    ]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(all_chunks))]

    # Generate multi-vector summaries/questions for first 15 chunks concurrently
    extra_texts = []
    extra_metadatas = []
    extra_ids = []
    
    from concurrent.futures import ThreadPoolExecutor
    chunks_to_process = all_chunks[:15]
    
    def process_one_chunk(idx_chunk):
        idx, chunk = idx_chunk
        chunk_id = f"{doc_id}_chunk_{idx}"
        summary, questions = generate_summary_and_questions(chunk.page_content)
        res = []
        if summary:
            res.append((
                summary,
                {
                    "source": filename,
                    "parent_chunk_id": chunk_id,
                    "type": "summary",
                    "doc_id": doc_id,
                    "strategy": "multi-vector",
                    "upload_date": upload_date,
                },
                f"{chunk_id}_summary"
            ))
        for q_idx, q in enumerate(questions):
            res.append((
                q,
                {
                    "source": filename,
                    "parent_chunk_id": chunk_id,
                    "type": "question",
                    "doc_id": doc_id,
                    "strategy": "multi-vector",
                    "upload_date": upload_date,
                },
                f"{chunk_id}_q_{q_idx}"
            ))
        return res

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_one_chunk, enumerate(chunks_to_process)))
        for r_list in results:
            for txt, meta, cid in r_list:
                extra_texts.append(txt)
                extra_metadatas.append(meta)
                extra_ids.append(cid)
        logger.info(f"Generated {len(extra_texts)} multi-vector summary/question chunks")
    except Exception as e:
        logger.error(f"Failed concurrent multi-vector generation: {e}")

    texts.extend(extra_texts)
    metadatas.extend(extra_metadatas)
    ids.extend(extra_ids)

    add_documents_to_chroma(data["chroma_store"], texts, metadatas, ids)

    try:
        from app.vectorstore.chroma import initialize_chroma_gemini
        gemini_store = initialize_chroma_gemini()
        if gemini_store:
            add_documents_to_chroma(gemini_store, texts, metadatas, ids)
            logger.info("Stored document chunks in Gemini collection")
    except Exception as e:
        logger.error(f"Failed to store in Gemini collection: {e}")

    parent_mapping = []
    if "parent_child" in chunks_result and isinstance(chunks_result["parent_child"], dict):
        parent_mapping = chunks_result["parent_child"].get("parent_mapping", [])

    inserted_parent_ids = set()
    for mapping in parent_mapping:
        p_id = mapping["parent_id"]
        if p_id in inserted_parent_ids:
            continue
        inserted_parent_ids.add(p_id)
        insert_parent_document(
            parent_id=p_id,
            document_id=doc_id,
            parent_content=mapping["parent_content"],
            chunk_index=0,
        )

    update_document_chunk_count(doc_id, len(all_chunks))
    update_document_pages(doc_id, data.get("total_pages", 0))
    update_document_status(doc_id, "completed")

    try:
        from app.services.bm25_retriever import get_bm25_retriever
        bm25_retriever = get_bm25_retriever()
        current_docs = list(bm25_retriever.documents)
        current_docs.extend(all_chunks)
        bm25_retriever._build_index(current_docs)
    except Exception as e:
        logger.error(f"Failed to update BM25 index during ingestion: {e}")

    logger.info(f"Ingestion complete: {filename} - {len(all_chunks)} chunks")
    return IngestionResult(
        doc_id=doc_id,
        filename=filename,
        chunks=len(all_chunks),
        duplicate=data["duplicate"],
    )


def _load_and_split(data: dict) -> dict:
    documents = load_chain.invoke(data["file_path"])
    chunks = all_strategies_chain.invoke(documents)
    data["chunks"] = chunks
    data["total_pages"] = len(documents)
    return data


def get_ingestion_chain(chroma_store):
    def _set_chroma(data: dict) -> dict:
        data["chroma_store"] = chroma_store
        return data

    return (
        RunnableLambda(_validate)
        | RunnableLambda(_set_chroma)
        | RunnableLambda(_load_and_split)
        | RunnableLambda(_store)
    )
