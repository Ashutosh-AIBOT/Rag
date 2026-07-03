import re
from typing import List
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logging import get_logger

logger = get_logger(__name__)

def split_recursive(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Standard Recursive Character Splitting.
    Splits text on paragraphs, sentences, and words recursively to keep paragraphs/sentences intact.
    """
    logger.info(f"Splitting with Recursive Character Splitter (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    
    # Label each chunk with the strategy used
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = "recursive"
        chunk.metadata["chunk_index"] = i
        
    logger.info(f"Generated {len(chunks)} recursive chunks.")
    return chunks

def split_parent_child(documents: List[Document], parent_size: int = 1000, child_size: int = 200, overlap: int = 20) -> List[Document]:
    """
    Parent-Child Chunking strategy.
    Creates small child chunks for high retrieval precision,
    but keeps the larger parent chunk content in metadata to send full context to the LLM.
    """
    logger.info(f"Splitting with Parent-Child Splitter (parent={parent_size}, child={child_size})...")
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=overlap, add_start_index=True)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=overlap, add_start_index=True)
    
    final_chunks = []
    parent_id_counter = 0
    
    for doc in documents:
        parent_docs = parent_splitter.split_documents([doc])
        for p_doc in parent_docs:
            parent_id = f"parent_{parent_id_counter}"
            parent_id_counter += 1
            
            # Split the parent doc content into child chunks
            child_docs = child_splitter.split_documents([p_doc])
            for c_idx, c_doc in enumerate(child_docs):
                c_doc.metadata["parent_id"] = parent_id
                c_doc.metadata["parent_text"] = p_doc.page_content
                c_doc.metadata["chunk_strategy"] = "parent-child"
                c_doc.metadata["chunk_index"] = c_idx
                final_chunks.append(c_doc)
                
    logger.info(f"Generated {len(final_chunks)} parent-child chunks.")
    return final_chunks

def split_section_based(documents: List[Document]) -> List[Document]:
    """
    Section-based chunking.
    Splits text by common Markdown heading syntax (e.g. lines starting with '#') or custom headers.
    """
    logger.info("Splitting with Section-Based Splitter...")
    final_chunks = []
    
    for doc in documents:
        lines = doc.page_content.split("\n")
        current_section_title = "Introduction"
        current_section_content = []
        section_index = 0
        
        for line in lines:
            # If line is a Markdown header, split and save previous section
            if line.strip().startswith("#"):
                if current_section_content:
                    content_str = "\n".join(current_section_content).strip()
                    if content_str:
                        sec_doc = Document(
                            page_content=content_str,
                            metadata={
                                **doc.metadata,
                                "chunk_strategy": "section",
                                "section_title": current_section_title,
                                "chunk_index": section_index
                            }
                        )
                        final_chunks.append(sec_doc)
                        section_index += 1
                
                # Update title to new header
                current_section_title = line.strip("#").strip()
                current_section_content = []
            else:
                current_section_content.append(line)
                
        # Append the final section
        if current_section_content:
            content_str = "\n".join(current_section_content).strip()
            if content_str:
                sec_doc = Document(
                    page_content=content_str,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "section",
                        "section_title": current_section_title,
                        "chunk_index": section_index
                    }
                )
                final_chunks.append(sec_doc)
                
    logger.info(f"Generated {len(final_chunks)} section-based chunks.")
    return final_chunks

def split_semantic(documents: List[Document], embeddings, threshold: float = 0.8) -> List[Document]:
    """
    Semantic Chunking based on cosine similarity of sentence embeddings.
    Splits the document when the similarity between consecutive sentences drops below threshold.
    """
    if embeddings is None:
        logger.warning("No embeddings model provided for semantic chunking. Falling back to recursive splitting.")
        return split_recursive(documents)

    logger.info("Splitting with Semantic Splitter...")
    final_chunks = []

    for doc in documents:
        # 1. Split document into sentences using basic punctuation regex
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        # 2. Embed each sentence
        try:
            sentence_embeddings = embeddings.embed_documents(sentences)
        except Exception as e:
            logger.error(f"Failed to embed sentences for semantic chunking: {e}")
            final_chunks.extend(split_recursive([doc]))
            continue

        # 3. Calculate similarity between consecutive sentences
        def cosine_similarity(v1, v2) -> float:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))

        chunks = []
        current_chunk_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity(sentence_embeddings[i-1], sentence_embeddings[i])
            if sim < threshold:
                # Low similarity starts a new chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        # 4. Create Document objects
        for idx, chunk_text in enumerate(chunks):
            final_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_strategy": "semantic",
                    "chunk_index": idx
                }
            ))

    logger.info(f"Generated {len(final_chunks)} semantic chunks.")
    return final_chunks

