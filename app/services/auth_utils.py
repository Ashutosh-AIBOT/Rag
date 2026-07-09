"""
Authentication utilities: JWT, password hashing, OTP generation.
"""
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import get_settings

logger = logging.getLogger("auth_utils")
settings = get_settings()

SECRET_KEY = settings.jwt_secret_key

# ── Revoked token store (thread-safe, TTL-based) ──
_revoked_tokens: set = set()
_revoked_lock = threading.Lock()

def _persist_secret(key: str) -> str:
    """Load a previously-generated secret from disk, or generate and save a new one."""
    secret_path = os.path.join(os.path.dirname(settings.sqlite_db_path) or ".", ".jwt_secret.json")
    try:
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                data = json.load(f)
                if data.get("secret"):
                    return data["secret"]
    except Exception:
        pass
    # Generate new secret and persist it
    try:
        os.makedirs(os.path.dirname(secret_path) or ".", exist_ok=True)
        with open(secret_path, "w") as f:
            json.dump({"secret": key}, f)
        os.chmod(secret_path, 0o600)
    except Exception as e:
        logger.warning("Could not persist JWT secret to %s: %s", secret_path, e)
    return key

# Always use a strong secret — randomize if the default is detected, then persist
if SECRET_KEY == "rag-platform-secret-key-change-in-production" or not SECRET_KEY:
    generated = secrets.token_hex(32)
    SECRET_KEY = _persist_secret(generated)
    if SECRET_KEY == generated:
        logger.warning("JWT secret auto-generated and persisted. Set JWT_SECRET_KEY env var for full control.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def hash_password(password: str) -> str:
    """Secure password hashing using PBKDF2 with SHA-256."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return f"pbkdf2_sha256:100000:{salt}:{dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash (supporting both legacy SHA-256 and new PBKDF2 format)."""
    try:
        if hashed.startswith("pbkdf2_sha256:"):
            _, iterations, salt, h = hashed.split(":", 3)
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
            return dk.hex() == h
        else:
            # Legacy SHA-256 format for backward compatibility
            salt, h = hashed.split(":", 1)
            return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest() == h
    except Exception as e:
        logger.warning("Password verify failed: %s", e)
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token. Requires python-jose."""
    logger.info("Access token created for sub=%s", data.get("sub"))
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError(
            "python-jose is required for secure JWT authentication. "
            "Install with: pip install 'python-jose[cryptography]'"
        )
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "jti": secrets.token_hex(8)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a signed JWT token. Requires python-jose."""
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError(
            "python-jose is required for secure JWT authentication. "
            "Install with: pip install 'python-jose[cryptography]'"
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Check if token was revoked (logout)
        jti = payload.get("jti")
        if jti:
            with _revoked_lock:
                if jti in _revoked_tokens:
                    return None
        return payload
    except Exception:
        logger.warning("Token decode failed: invalid or expired")
        return None


def revoke_token(token: str) -> bool:
    """Add a token's jti to the revoked set. Returns True if successful."""
    try:
        from jose import jwt
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        jti = payload.get("jti")
        if jti:
            with _revoked_lock:
                _revoked_tokens.add(jti)
            return True
    except Exception:
        pass
    return False


def generate_otp() -> str:
    """Generate a 6-digit OTP code."""
    return f"{secrets.randbelow(1000000):06d}"


def generate_reset_token() -> str:
    """Generate a password reset token."""
    return secrets.token_urlsafe(32)
