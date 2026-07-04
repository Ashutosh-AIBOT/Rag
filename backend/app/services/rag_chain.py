from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm_chain
from app.core.logging import get_logger

logger = get_logger(__name__)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the context provided.

Context:
{context}

Question:
{question}

Answer:"""


def build_rag_chain():
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: x["context"],
            question=lambda x: x["question"],
        )
        | prompt
        | get_llm_chain()
        | StrOutputParser()
    )
    return chain
