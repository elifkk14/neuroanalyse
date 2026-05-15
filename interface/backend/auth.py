from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import AuditLog, User, UserSession, get_db, get_setting

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)

SESSION_TIMEOUT_DEFAULT = 30  # minutes
MAX_FAILED_ATTEMPTS = 5


# ── Password utilities ────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def validate_password_strength(plain: str) -> str | None:
    """Returns an error message string or None if valid."""
    if len(plain) < 8:
        return "password_too_short"
    if not any(c.isupper() for c in plain):
        return "password_needs_upper"
    if not any(c.isdigit() for c in plain):
        return "password_needs_digit"
    return None


# ── Token utilities ───────────────────────────────────────────────────────────

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(
    db: Session,
    user: User,
    ip_address: str | None,
    timeout_minutes: int = SESSION_TIMEOUT_DEFAULT,
) -> str:
    token = secrets.token_urlsafe(48)
    h = _token_hash(token)
    expires = datetime.utcnow() + timedelta(minutes=timeout_minutes)
    sess = UserSession(
        user_id=user.id,
        token_hash=h,
        expires_at=expires,
        ip_address=ip_address,
    )
    db.add(sess)
    db.commit()
    return token


def invalidate_session(db: Session, token: str) -> None:
    h = _token_hash(token)
    sess = db.query(UserSession).filter(UserSession.token_hash == h).first()
    if sess:
        sess.is_valid = False
        db.commit()


def _get_session_timeout(db: Session) -> int:
    val = get_setting(db, "session_timeout_minutes")
    try:
        return int(val) if val else SESSION_TIMEOUT_DEFAULT
    except (TypeError, ValueError):
        return SESSION_TIMEOUT_DEFAULT


# ── FastAPI dependencies ──────────────────────────────────────────────────────

def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("na_session")


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not_authenticated")

    h = _token_hash(token)
    sess = (
        db.query(UserSession)
        .filter(UserSession.token_hash == h, UserSession.is_valid == True)
        .first()
    )
    if not sess:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="session_invalid")

    if datetime.utcnow() > sess.expires_at:
        sess.is_valid = False
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="session_expired")

    # Slide the expiry window
    timeout = _get_session_timeout(db)
    sess.last_active_at = datetime.utcnow()
    sess.expires_at = datetime.utcnow() + timedelta(minutes=timeout)
    db.commit()

    user = db.query(User).filter(User.id == sess.user_id).first()
    if not user or not user.is_active or user.is_locked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_unavailable")

    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin_required")
    return user


# ── Audit logging ─────────────────────────────────────────────────────────────

def audit(
    db: Session,
    action: str,
    details: str | None = None,
    user_id: int | None = None,
    ip_address: str | None = None,
) -> None:
    db.add(AuditLog(
        user_id=user_id,
        action=action,
        details=details,
        ip_address=ip_address,
    ))
    db.commit()


def get_client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None
