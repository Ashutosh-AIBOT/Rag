from pydantic import BaseModel


class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str
    model_name: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 4096


def get_llm_configs():
    from app.config import settings

    return [
        LLMConfig(
            provider="nvidia",
            model_name="meta/llama-3.1-70b-instruct",
            api_key=settings.NVIDIA_API_KEY,
        ),
        LLMConfig(
            provider="groq",
            model_name="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
        ),
        LLMConfig(
            provider="gemini",
            model_name="gemini-2.0-flash",
            api_key=settings.GOOGLE_API_KEY,
        ),
    ]
