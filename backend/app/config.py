from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()
print("[stage00 | config | 002] OK: .env loaded")


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
print("[stage00 | config | 003] OK: Settings loaded")

for i, (name, val) in enumerate(
    [("GOOGLE_API_KEY", settings.GOOGLE_API_KEY),
     ("GROQ_API_KEY", settings.GROQ_API_KEY),
     ("NVIDIA_API_KEY", settings.NVIDIA_API_KEY)], 4
):
    status = "OK" if val else "WARN"
    msg = "present" if val else "missing"
    print(f"[stage00 | config | {i:03d}] {status}: {name} {msg}")