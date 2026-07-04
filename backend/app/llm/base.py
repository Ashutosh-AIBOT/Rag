from langchain_openai import ChatOpenAI
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_llm(self):
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.3,
            max_tokens=4096,
            timeout=10,
        )
