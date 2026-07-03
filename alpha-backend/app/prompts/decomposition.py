DECOMPOSITION_PROMPT = """Decompose the following complex user query into 2 to 3 simpler, distinct sub-questions that need to be answered to address the main query.
Return the sub-questions separated by newlines. Do not add any introductory or concluding text. Do not number the questions.

Complex Query: {question}
"""
