from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel
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


def _semantic_split(documents: list[Document]) -> list[Document]:
    return semantic_chunking_chain.invoke(documents)


def _parent_child_split(documents: list[Document]) -> dict:
    return parent_child_chain.invoke(documents)


def _section_split(documents: list[Document]) -> list[Document]:
    return section_chunking_chain.invoke(documents)


def _all_strategies(documents: list[Document]) -> dict:
    recursive_chunks = _recursive_split(documents)
    semantic_chunks = _semantic_split(documents)
    parent_child_result = _parent_child_split(documents)
    section_chunks = _section_split(documents)

    return {
        "recursive": recursive_chunks,
        "semantic": semantic_chunks,
        "parent_child": parent_child_result["child_chunks"],
        "parent_mapping": parent_child_result["parent_mapping"],
        "section": section_chunks,
    }


recursive_chunking_chain = RunnableLambda(_recursive_split)
semantic_chunking_chain_standalone = RunnableLambda(_semantic_split)
parent_child_chain_standalone = RunnableLambda(_parent_child_split)
section_chunking_chain_standalone = RunnableLambda(_section_split)
all_strategies_chain = RunnableParallel(
    recursive=RunnableLambda(_recursive_split),
    semantic=RunnableLambda(_semantic_split),
    parent_child=RunnableLambda(_parent_child_split),
    section=RunnableLambda(_section_split),
)
