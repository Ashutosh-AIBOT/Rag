from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logging import get_logger
from app.services.semantic_chunker import semantic_chunking_chain
from app.services.parent_child_chunker import parent_child_chain
from app.services.section_chunker import section_chunking_chain

logger = get_logger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _recursive_split(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["strategy"] = "recursive"
    logger.info(f"Recursive chunking: {len(chunks)} chunks created")
    return chunks


def _all_strategies(documents: list[Document]) -> dict:
    recursive_chunks = _recursive_split(documents)
    semantic_chunks = semantic_chunking_chain.invoke(documents)
    parent_child_result = parent_child_chain.invoke(documents)
    section_chunks = section_chunking_chain.invoke(documents)

    return {
        "recursive": recursive_chunks,
        "semantic": semantic_chunks,
        "parent_child": parent_child_result["child_chunks"],
        "parent_mapping": parent_child_result["parent_mapping"],
        "section": section_chunks,
    }


recursive_chunking_chain = RunnableLambda(_recursive_split)
all_strategies_chain = RunnableLambda(_all_strategies)
