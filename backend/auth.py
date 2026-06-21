"""
JWT-based authentication helpers for SleepSense AI.

Tokens are signed with HS256 and expire after ACCESS_TOKEN_EXPIRE_MINUTES.
The SECRET_KEY must be set in the environment for production deployments;
a random fallback is used in development so the server never fails to start.
"""

import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from backend import db

_raw_secret = os.environ.get("JWT_SECRET_KEY")
if not _raw_secret:
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        raise RuntimeError(
            "JWT_SECRET_KEY env var must be set in production. "
            "Generate one with: openssl rand -hex 32"
        )
    _raw_secret = secrets.token_hex(32)  # dev-only fallback

SECRET_KEY = _raw_secret
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24h default
ACCESS_TOKEN_EXPIRE_SECONDS = ACCESS_TOKEN_EXPIRE_MINUTES * 60

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Token helpers ─────────────────────────────────────────────────────────────

def create_access_token(user_id: str, role: str) -> str:
    from backend import session as _session
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "role": role, "exp": expire, "jti": jti}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    from backend import session as _session
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    jti = payload.get("jti")
    if jti and _session.is_token_revoked(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


# ── FastAPI dependency: current user ──────────────────────────────────────────

def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """
    Dependency that resolves the authenticated user from the Bearer token.
    Raises 401 if the token is missing or invalid.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    """Like get_current_user but returns None instead of raising when unauthenticated."""
    if not token:
        return None
    try:
        return get_current_user(token)
    except HTTPException:
        return None


def require_provider(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "provider":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Provider access required")
    return user
