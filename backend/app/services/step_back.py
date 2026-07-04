from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.llm import get_llm_chain
from app.core.logging import get_logger

logger = get_logger(__name__)

STEP_BACK_PROMPT_TEMPLATE = """You are a helpful assistant that generates broader, more general questions.

Given the following specific question, generate a broader, more general question that would help find relevant context.

Specific Question: {question}

Broader Question:"""


def build_step_back_chain():
    prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(question=lambda x: x["question"])
        | prompt
        | get_llm_chain()
        | StrOutputParser()
    )
    return chain
