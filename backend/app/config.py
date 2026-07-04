from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()
print("[stage00 | config | 002-A] OK: .env loaded")


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    EMBEDDING_MODEL_NAME: str = ""
    CHROMA_PERSIST_DIRECTORY: str = ""
    CHROMA_COLLECTION_NAME: str = ""
    DATABASE_URL: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
print("[stage00 | config | 002-B] OK: Settings loaded")

keys = [
    ("GOOGLE_API_KEY", settings.GOOGLE_API_KEY),
    ("GROQ_API_KEY", settings.GROQ_API_KEY),
    ("NVIDIA_API_KEY", settings.NVIDIA_API_KEY),
    ("EMBEDDING_MODEL_NAME", settings.EMBEDDING_MODEL_NAME),
    ("CHROMA_PERSIST_DIRECTORY", settings.CHROMA_PERSIST_DIRECTORY),
    ("CHROMA_COLLECTION_NAME", settings.CHROMA_COLLECTION_NAME),
    ("DATABASE_URL", settings.DATABASE_URL),
]
for letter, (name, val) in zip("CDEFGHI", keys):
    status = "OK" if val else "WARN"
    msg = "present" if val else "missing"
    print(f"[stage00 | config | 002-{letter}] {status}: {name} {msg}")
