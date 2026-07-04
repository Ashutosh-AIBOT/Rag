from pathlib import Path

from app.core.logging import get_logger
from app.database.database import get_document_by_filename

logger = get_logger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def validate_file_type(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")
    logger.info(f"File type valid: {ext}")


def validate_file_size(file_path: str) -> None:
    file_size = Path(file_path).stat().st_size
    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / 1024 / 1024
        raise ValueError(f"File too large: {size_mb:.1f}MB. Max: 50MB")
    logger.info(f"File size valid: {file_size / 1024:.1f}KB")


def check_duplicate(filename: str) -> str | None:
    existing = get_document_by_filename(filename)
    if existing:
        logger.warning(f"Duplicate detected: {filename}")
        return existing["id"]
    logger.info("No duplicate found")
    return None


def validate_all(file_path: str) -> str | None:
    filename = Path(file_path).name
    validate_file_type(filename)
    validate_file_size(file_path)
    return check_duplicate(filename)
