import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.llm import get_llm
from app.services.retrieval import search_chunks
from app.core.logging import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc, _ in docs)


def get_rag_chain(vectorstore, document_ids: list[str] | None = None):
    llm = get_llm()

    def retrieve(query: str) -> list:
        filter_dict = None
        if document_ids:
            if len(document_ids) == 1:
                filter_dict = {"doc_id": document_ids[0]}
            else:
                filter_dict = {"doc_id": {"$in": document_ids}}
        return search_chunks(vectorstore, query, k=5, filter_dict=filter_dict)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retrieve(x["question"])),
            question=lambda x: x["question"],
        )
        | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain ready")
    return chain
