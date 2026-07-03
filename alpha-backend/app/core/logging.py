import logging
import sys
from app.config import settings

def setup_logging() -> None:
    """
    Initializes and configures the application-wide logging system.
    Configures format and level from application settings.
    """
    # Clear existing handlers to avoid duplication
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Set up console handler writing to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL)

    # Set log formatting
    formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(settings.LOG_LEVEL)
    root_logger.addHandler(console_handler)

    # Suppress verbose third-party libraries logging unless DEBUG is active
    if not settings.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    root_logger.info("Logging system configured successfully.")

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance for the given module name.
    """
    return logging.getLogger(name)
