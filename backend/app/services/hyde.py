from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.llm import get_llm_chain
from app.core.logging import get_logger

logger = get_logger(__name__)

HYDE_PROMPT_TEMPLATE = """Write a detailed, informative answer to the following question. 
This answer will be used to find similar documents in a vector database.

Question: {question}

Detailed Answer:"""


def build_hyde_chain():
    prompt = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(question=lambda x: x["question"])
        | prompt
        | get_llm_chain()
        | StrOutputParser()
    )
    return chain
