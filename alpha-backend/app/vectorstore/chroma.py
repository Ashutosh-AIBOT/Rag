import os
from langchain_community.vectorstores import Chroma
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def get_chroma_db(embeddings) -> Chroma:
    """
    Simple, beginner-friendly helper to connect to or initialize ChromaDB.
    Uses the embeddings model passed from the global app state.
    """
    logger.info(f"Connecting to ChromaDB at: {settings.CHROMA_DB_PATH} using collection: {settings.CHROMA_COLLECTION_NAME}")
    
    # Ensure database folder exists
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
    
    # Initialize the Chroma vector store
    db = Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )
    return db
