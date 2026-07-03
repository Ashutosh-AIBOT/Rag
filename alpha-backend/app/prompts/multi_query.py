MULTI_QUERY_PROMPT = """You are an AI language model assistant.
Your task is to generate 3 alternative versions of the user query to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user query, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Provide these alternative queries separated by newlines. Do not add any introductory or concluding text. Do not number the queries.

Original query: {question}
"""
