import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.embeddings.sentence_transformer import get_embedding_model
from app.core.logging import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.5


def _get_embeddings(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.embed_documents(texts)


def _semantic_split(documents: list[Document]) -> list[Document]:
    chunks = []
    for doc in documents:
        sentences = [s.strip() for s in doc.page_content.split(". ") if s.strip()]
        if len(sentences) <= 1:
            chunks.append(Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "strategy": "semantic"}
            ))
            continue

        embeddings = _get_embeddings(sentences)
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            if similarity < SIMILARITY_THRESHOLD:
                chunk_text = ". ".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "strategy": "semantic"}
                ))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunk_text = ". ".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={**doc.metadata, "strategy": "semantic"}
            ))

    logger.info(f"Semantic chunking: {len(chunks)} chunks created")
    return chunks


semantic_chunking_chain = RunnableLambda(_semantic_split)
