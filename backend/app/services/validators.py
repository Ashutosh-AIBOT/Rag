from pathlib import Path

from app.core.logging import get_logger
from app.database.database import get_document_by_filename

logger = get_logger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def validate_file_type(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f"[stage01 | validators | 008-A] FAIL: Unsupported file type - {ext}")
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")
    print(f"[stage01 | validators | 008-A] OK: File type valid - {ext}")


def validate_file_size(file_path: str) -> None:
    file_size = Path(file_path).stat().st_size
    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / 1024 / 1024
        print(f"[stage01 | validators | 008-B] FAIL: File too large - {size_mb:.1f}MB")
        raise ValueError(f"File too large: {size_mb:.1f}MB. Max: 50MB")
    print(f"[stage01 | validators | 008-B] OK: File size valid - {file_size / 1024:.1f}KB")


def check_duplicate(filename: str) -> str | None:
    existing = get_document_by_filename(filename)
    if existing:
        print(f"[stage01 | validators | 008-C] WARN: Duplicate detected - {filename}")
        return existing["id"]
    print(f"[stage01 | validators | 008-C] OK: No duplicate found")
    return None


def validate_all(file_path: str) -> str | None:
    filename = Path(file_path).name
    validate_file_type(filename)
    validate_file_size(file_path)
    return check_duplicate(filename)