from fastapi import HTTPException, status


class DocumentNotFoundException(HTTPException):
    def __init__(self, doc_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {doc_id}",
        )


class IngestionFailedException(HTTPException):
    def __init__(self, detail: str = "Document ingestion failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


class QueryFailedException(HTTPException):
    def __init__(self, detail: str = "Query processing failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


class NoLLMProviderException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No LLM provider available. All providers failed.",
        )
