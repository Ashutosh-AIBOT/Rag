"""
Authentication API router: register, login, OTP, forgot password, change password.
All user data persisted to SQLite via SQLAlchemy User model.
"""
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.database import get_db, gen_id, User
from app.services.auth_utils import (
    hash_password, verify_password, create_access_token,
    decode_token, generate_otp, generate_reset_token, revoke_token,
)

logger = logging.getLogger("auth_service")

# ── Simple in-memory rate limiter ──
_rate_limits: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()
MAX_REQUESTS_PER_MINUTE = 20

def _check_rate_limit(key: str) -> bool:
    now = time.time()
    with _rate_lock:
        timestamps = _rate_limits[key]
        _rate_limits[key] = [t for t in timestamps if t > now - 60]
        if len(_rate_limits[key]) >= MAX_REQUESTS_PER_MINUTE:
            return False
        _rate_limits[key].append(now)
        return True

def send_otp_email(email: str, otp: str):
    logger.info("Mock SMTP: Sending OTP to %s (length=%d)", email, len(otp))

def send_reset_email(email: str, token: str):
    logger.info("Mock SMTP: Sending password reset token to %s (length=%d)", email, len(token))
router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


# ── Pydantic models ──

class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str = ""

class UserLogin(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str


# ── In-memory OTP / reset token stores (thread-safe, ephemeral) ──
_pending_otps: dict = {}
_reset_tokens: dict = {}
_store_lock = threading.Lock()


def _get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that extracts and validates the JWT, returning the User ORM object."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.email == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ── Endpoints ──

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: UserRegister, db: Session = Depends(get_db)):
    if not _check_rate_limit("register"):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    if not body.email or "@" not in body.email:
        raise HTTPException(status_code=422, detail="Invalid email format")
    if not body.password or len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_count = db.query(User).count()
    user = User(
        id=gen_id(),
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        is_active=True,
        is_admin=(user_count == 0),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("User registered: %s (admin=%s)", body.email, user.is_admin)
    return {"message": "Registration successful. Please verify your email."}


@router.post("/login")
def login(body: UserLogin, db: Session = Depends(get_db)):
    if not _check_rate_limit("login"):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    token = create_access_token({"sub": user.email, "user_id": user.id})
    logger.info("User logged in: %s", body.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        },
    }


@router.get("/me")
def get_me(current_user: User = Depends(_get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_admin": current_user.is_admin,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


@router.post("/verify-otp")
def verify_otp(body: VerifyOTPRequest):
    with _store_lock:
        stored = _pending_otps.get(body.email)
    if not stored or stored["otp"] != body.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    expires_at = datetime.fromisoformat(stored["expires"])
    if datetime.now(timezone.utc) > expires_at:
        with _store_lock:
            _pending_otps.pop(body.email, None)
        raise HTTPException(status_code=400, detail="OTP expired")
    with _store_lock:
        _pending_otps.pop(body.email, None)
    logger.info("OTP verified for: %s", body.email)
    return {"message": "Email verified successfully"}


@router.post("/forgot-password")
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user:
        return {"message": "If the email exists, a reset link has been sent"}
    otp = generate_otp()
    with _store_lock:
        _pending_otps[body.email] = {
            "otp": otp,
            "expires": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        }
    send_otp_email(body.email, otp)
    
    token = generate_reset_token()
    with _store_lock:
        _reset_tokens[body.email] = {
            "token": token,
            "expires": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        }
    send_reset_email(body.email, token)
    
    logger.info("Password reset OTP and Token sent to: %s", body.email)
    return {
        "message": "If the email exists, a reset link has been sent",
    }


@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    with _store_lock:
        items = list(_reset_tokens.items())
    for email, data in items:
        if data["token"] == body.token:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.hashed_password = hash_password(body.new_password)
                db.commit()
            with _store_lock:
                _reset_tokens.pop(email, None)
            logger.info("Password reset completed for: %s", email)
            return {"message": "Password reset successful"}
    raise HTTPException(status_code=400, detail="Invalid or expired reset token")


@router.post("/change-password")
def change_password(body: ChangePasswordRequest, db: Session = Depends(get_db), current_user: User = Depends(_get_current_user)):
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.hashed_password = hash_password(body.new_password)
    db.commit()
    logger.info("Password changed for: %s", current_user.email)
    return {"message": "Password changed successfully"}


@router.post("/logout")
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if credentials:
        revoke_token(credentials.credentials)
    return {"message": "Logged out successfully"}
