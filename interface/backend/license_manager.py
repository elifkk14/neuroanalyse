"""
License Manager
===============
Handles license validation for standalone hospital workstations.

Architecture:
  - License key stored in system_settings table
  - Validation: local check + optional remote ping
  - Offline grace: 30 days from last successful online validation
  - Types: demo (30d, 10 analyses, watermark), standard (12m), enterprise (multi-user)
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from database import get_setting, set_setting

# Well-known demo license key (shipped with the app for evaluation)
_DEMO_KEY = "DEMO-NEURO-2024-EVAL"
_DEMO_KEY_HASH = hashlib.sha256(_DEMO_KEY.encode()).hexdigest()

# Offline grace period
OFFLINE_GRACE_DAYS = 30

# License server URL (placeholder; only called if internet is available)
LICENSE_SERVER_URL = "https://license.neuroanalyse.io/v1/validate"


class LicenseStatus:
    ACTIVE   = "active"
    DEMO     = "demo"
    EXPIRED  = "expired"
    INVALID  = "invalid"
    OFFLINE  = "offline"   # within grace period


def get_license_info(db: Session) -> dict:
    key         = get_setting(db, "license_key") or ""
    lic_type    = get_setting(db, "license_type") or "demo"
    expires_raw = get_setting(db, "license_expires") or ""
    last_valid  = get_setting(db, "license_last_validated") or ""
    analyses_used = int(get_setting(db, "demo_analyses_used") or "0")

    now = datetime.utcnow()

    # Demo key bypass
    if key.upper() == _DEMO_KEY:
        expires_at = _parse_dt(expires_raw)
        expired = expires_at is not None and now > expires_at
        return {
            "status":          LicenseStatus.EXPIRED if expired else LicenseStatus.DEMO,
            "type":            "demo",
            "key_masked":      _mask_key(key),
            "expires_at":      expires_raw,
            "analyses_used":   analyses_used,
            "analyses_limit":  100,
            "is_demo":         True,
            "institution_name": get_setting(db, "institution_name") or "Demo Institution",
            "days_remaining":  _days_remaining(expires_at),
            "warn_expiry":     _warn_expiry(expires_at),
        }

    if not key:
        return _inactive_info(db)

    # Check local expiry
    expires_at = _parse_dt(expires_raw)
    if expires_at and now > expires_at:
        return {
            "status":  LicenseStatus.EXPIRED,
            "type":    lic_type,
            "key_masked": _mask_key(key),
            "expires_at": expires_raw,
            "is_demo": False,
            "institution_name": get_setting(db, "institution_name") or "",
            "days_remaining": 0,
            "warn_expiry": False,
        }

    # Check offline grace
    last_dt = _parse_dt(last_valid)
    grace_ok = last_dt is not None and (now - last_dt).days < OFFLINE_GRACE_DAYS
    online_status = _try_remote_validate(key)

    if online_status == "active":
        set_setting(db, "license_last_validated", now.isoformat())
        status = LicenseStatus.ACTIVE
    elif grace_ok:
        status = LicenseStatus.OFFLINE
    else:
        status = LicenseStatus.INVALID

    return {
        "status":          status,
        "type":            lic_type,
        "key_masked":      _mask_key(key),
        "expires_at":      expires_raw,
        "is_demo":         False,
        "institution_name": get_setting(db, "institution_name") or "",
        "days_remaining":  _days_remaining(expires_at),
        "warn_expiry":     _warn_expiry(expires_at),
        "offline_grace_remaining": (
            OFFLINE_GRACE_DAYS - (now - last_dt).days if last_dt else None
        ),
    }


def activate_license(db: Session, key: str, institution_name: str) -> dict:
    key = key.strip().upper()
    now = datetime.utcnow()

    # Demo key
    if key == _DEMO_KEY:
        expires = (now + timedelta(days=30)).isoformat()
        set_setting(db, "license_key",             key)
        set_setting(db, "license_type",            "demo")
        set_setting(db, "license_expires",         expires)
        set_setting(db, "license_last_validated",  now.isoformat())
        set_setting(db, "demo_analyses_used",      "0")
        set_setting(db, "institution_name",        institution_name)
        return {"ok": True, "type": "demo", "expires_at": expires}

    # Attempt remote activation
    result = _try_remote_activate(key, institution_name)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "activation_failed")}

    set_setting(db, "license_key",            key)
    set_setting(db, "license_type",           result.get("type", "standard"))
    set_setting(db, "license_expires",        result.get("expires_at", ""))
    set_setting(db, "license_last_validated", now.isoformat())
    set_setting(db, "institution_name",       institution_name)
    return {"ok": True, "type": result.get("type"), "expires_at": result.get("expires_at")}


def check_can_analyze(db: Session) -> tuple[bool, str | None]:
    """Returns (allowed, error_code_or_None)."""
    info = get_license_info(db)
    status = info["status"]

    if status in (LicenseStatus.EXPIRED, LicenseStatus.INVALID):
        return False, "license_expired"

    if status == LicenseStatus.DEMO:
        used  = info.get("analyses_used", 0)
        limit = info.get("analyses_limit", 100)
        if used >= limit:
            return False, "demo_limit_reached"
        # Increment counter
        set_setting(db, "demo_analyses_used", str(used + 1))

    return True, None


def init_default_license(db: Session) -> None:
    """Called at startup if no license is configured."""
    existing = get_setting(db, "license_key")
    if not existing:
        now = datetime.utcnow()
        expires = (now + timedelta(days=30)).isoformat()
        set_setting(db, "license_key",            _DEMO_KEY)
        set_setting(db, "license_type",           "demo")
        set_setting(db, "license_expires",        expires)
        set_setting(db, "license_last_validated", now.isoformat())
        set_setting(db, "demo_analyses_used",     "0")
        set_setting(db, "institution_name",       "Demo Kurumu")


# ── Remote calls (best-effort, never raise) ───────────────────────────────────

def _try_remote_validate(key: str) -> str:
    try:
        resp = httpx.post(
            LICENSE_SERVER_URL,
            json={"key": key, "action": "validate"},
            timeout=3.0,
        )
        if resp.status_code == 200:
            return resp.json().get("status", "invalid")
    except Exception:
        pass
    return "offline"


def _try_remote_activate(key: str, institution: str) -> dict:
    try:
        resp = httpx.post(
            LICENSE_SERVER_URL,
            json={"key": key, "action": "activate", "institution": institution},
            timeout=5.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"ok": False, "error": "server_unreachable"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return key[:4] + "-****-****-" + key[-4:]


def _days_remaining(expires_at: datetime | None) -> int | None:
    if not expires_at:
        return None
    delta = (expires_at - datetime.utcnow()).days
    return max(0, delta)


def _warn_expiry(expires_at: datetime | None) -> bool:
    if not expires_at:
        return False
    return (expires_at - datetime.utcnow()).days <= 30


def _inactive_info(db: Session) -> dict:
    return {
        "status":   LicenseStatus.INVALID,
        "type":     "none",
        "key_masked": "",
        "expires_at": "",
        "is_demo":  False,
        "institution_name": get_setting(db, "institution_name") or "",
        "days_remaining": None,
        "warn_expiry": False,
    }
