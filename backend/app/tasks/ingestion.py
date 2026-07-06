from app.core.celery_app import celery_app
from app.core.logging import get_logger
from app.database.database import update_document_status
from app.embeddings.sentence_transformer import load_embedding_model
from app.vectorstore.chroma import initialize_chroma
from app.services.ingestion import get_ingestion_chain

logger = get_logger(__name__)


@celery_app.task(name="app.tasks.ingestion.process_document_task")
def process_document_task(file_path: str, doc_id: str):
    try:
        logger.info(f"Celery processing document {doc_id} starting...")
        update_document_status(doc_id, "processing")
        
        # Initialize chroma store inside the worker process
        embeddings = load_embedding_model()
        chroma_store = initialize_chroma(embeddings)
        
        chain = get_ingestion_chain(chroma_store)
        chain.invoke({"file_path": file_path, "doc_id": doc_id})
        logger.info(f"Celery processing document {doc_id} completed successfully")
    except Exception as e:
        update_document_status(doc_id, "failed")
        logger.error(f"Celery processing document {doc_id} failed: {e}")
        raise
