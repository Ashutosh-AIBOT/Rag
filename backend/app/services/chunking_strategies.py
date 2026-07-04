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


all_strategies_chain = RunnableParallel(
    recursive=RunnableLambda(_recursive_split),
    semantic=semantic_chunking_chain,
    parent_child=parent_child_chain,
    section=section_chunking_chain,
)

