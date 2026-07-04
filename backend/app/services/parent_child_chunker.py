import uuid
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logging import get_logger

logger = get_logger(__name__)

PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200
CHUNK_OVERLAP = 20


def _parent_child_split(documents: list[Document]) -> dict:
    parent_chunks = []
    child_chunks = []
    parent_mapping = []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc in documents:
        parents = parent_splitter.split_documents([doc])
        for parent in parents:
            parent_id = str(uuid.uuid4())
            parent.metadata["strategy"] = "parent-child"
            parent.metadata["parent_id"] = parent_id
            parent.metadata["is_parent"] = True
            parent_chunks.append(parent)

            children = child_splitter.split_documents([parent])
            for i, child in enumerate(children):
                child_id = f"{parent_id}_child_{i}"
                child.metadata["strategy"] = "parent-child"
                child.metadata["parent_id"] = parent_id
                child.metadata["child_index"] = i
                child.metadata["is_parent"] = False
                child_chunks.append(child)

                parent_mapping.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "parent_content": parent.page_content,
                })

    logger.info(f"Parent-child: {len(parent_chunks)} parents, {len(child_chunks)} children")
    return {
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
        "parent_mapping": parent_mapping,
    }


parent_child_chain = RunnableLambda(_parent_child_split)
