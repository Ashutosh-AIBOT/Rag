from langchain_core.language_models import BaseChatModel
from app.llm.models import LLMConfig


def create_llm(config: LLMConfig) -> BaseChatModel:
    if config.provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model=config.model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {config.provider}")