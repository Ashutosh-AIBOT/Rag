from enum import Enum

from pydantic import BaseModel


class ProviderName(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    NVIDIA = "nvidia"


class LLMConfig(BaseModel):
    provider: ProviderName
    model_name: str
    api_key_env: str
    temperature: float = 0.3
    max_tokens: int = 4096


class LLMResponse(BaseModel):
    content: str
    provider: ProviderName
    model: str
    usage: dict | None = None
