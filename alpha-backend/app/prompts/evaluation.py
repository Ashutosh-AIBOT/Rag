FAITHFULNESS_PROMPT = """You are a grader assessing whether an LLM answer is faithful to/supported by the provided context.
Given the context and the answer, output a score between 0.0 and 1.0 (where 1.0 means perfectly faithful and supported, and 0.0 means completely unsupported or contradicted).
Provide a brief reason on the first line, and the score as a single float on the second line. Do not output anything else.

Context:
{context}

Answer:
{answer}

Reason:
Score:"""

RELEVANCY_PROMPT = """You are a grader assessing whether an LLM answer is relevant to the user query.
Given the query and the answer, output a score between 0.0 and 1.0 (where 1.0 means perfectly relevant, and 0.0 means completely irrelevant).
Provide a brief reason on the first line, and the score as a single float on the second line. Do not output anything else.

Query:
{question}

Answer:
{answer}

Reason:
Score:"""

CONTEXT_PRECISION_PROMPT = """You are a grader assessing whether a retrieved context chunk is relevant to the question.
Given the question and the context chunk, output a score between 0.0 and 1.0 (where 1.0 means highly relevant and useful to answer the question, and 0.0 means completely irrelevant).
Provide a brief reason on the first line, and the score as a single float on the second line. Do not output anything else.

Question:
{question}

Context Chunk:
{context_chunk}

Reason:
Score:"""

CONTEXT_RECALL_PROMPT = """You are a grader assessing whether the retrieved context contains all the necessary information to reconstruct the reference answer.
Given the question, the reference answer, and the retrieved context, output a score between 0.0 and 1.0 (where 1.0 means the context contains all facts mentioned in the reference answer, and 0.0 means the context contains none of the facts).
Provide a brief reason on the first line, and the score as a single float on the second line. Do not output anything else.

Question:
{question}

Reference Answer:
{reference}

Retrieved Context:
{context}

Reason:
Score:"""
