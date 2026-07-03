from enum import Enum

from pydantic import BaseModel


class ProviderName(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    NVIDIA = "nvidia"


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: ProviderName
    model_name: str
    api_key_env: str
    temperature: float = 0.3
    max_tokens: int = 4096


class LLMResponse(BaseModel):
    """Standard response from any LLM provider."""
    content: str
    provider: ProviderName
    model: str
    usage: dict | None = None
