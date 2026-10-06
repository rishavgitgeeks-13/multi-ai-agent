"""
Simple user auth for Streamlit login / signup.

Rules (company access):
  - User id must be a @gitgeeks.com email (local part alone is auto-appended).
  - Signup creates a *pending* request — only admin can approve.
  - Login is allowed only for approved users (plus bootstrap APP_USERNAME admin).

Storage priority:
  1. MongoDB `users` collection (when MONGODB_URI is available)
  2. Local JSON file at data/users.json (fallback)

Passwords are stored as PBKDF2-SHA256 hashes (never plaintext).
Env APP_USERNAME / APP_PASSWORD remain a bootstrap admin login.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
_PBKDF2_ITERATIONS = 120_000

# Company email domain — signup/login identity must use this.
ALLOWED_EMAIL_DOMAIN = "gitgeeks.com"
_LOCAL_PART_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,62}[a-zA-Z0-9]$|^[a-zA-Z0-9]{2,64}$")
_EMAIL_RE = re.compile(
    rf"^[a-zA-Z0-9][a-zA-Z0-9._-]{{0,62}}[a-zA-Z0-9]@{re.escape(ALLOWED_EMAIL_DOMAIN)}$",
    re.I,
)

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"


def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    ).hex()
    return digest, salt


def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    digest, _ = _hash_password(password, salt=salt)
    return secrets.compare_digest(digest, password_hash)


def _admin_credentials() -> Tuple[str, str]:
    try:
        from config.settings import settings

        return str(settings.APP_USERNAME), str(settings.APP_PASSWORD)
    except Exception:
        return (
            os.getenv("APP_USERNAME", "admin"),
            os.getenv("APP_PASSWORD", "admin@123/"),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_user_id(raw: str) -> Tuple[bool, str, str]:
    """
    Normalise signup/login identity to a @gitgeeks.com email.

    Accepts:
      - full email: name@gitgeeks.com
      - local part: name  →  name@gitgeeks.com

    Returns (ok, email_or_error, message).
    """
    value = (raw or "").strip().lower()
    if not value:
        return False, "", "Enter your GitGeeks work email."

    if "@" in value:
        if not value.endswith(f"@{ALLOWED_EMAIL_DOMAIN}"):
            return (
                False,
                "",
                f"Only @{ALLOWED_EMAIL_DOMAIN} emails can sign up or log in.",
            )
        if not _EMAIL_RE.match(value):
            return False, "", "Enter a valid work email (example: name@gitgeeks.com)."
        return True, value, ""

    # Local part only — append company domain
    if not _LOCAL_PART_RE.match(value):
        return (
            False,
            "",
            "User id must be 2–64 characters (letters, numbers, . _ -).",
        )
    email = f"{value}@{ALLOWED_EMAIL_DOMAIN}"
    if not _EMAIL_RE.match(email):
        return False, "", "Enter a valid work email local part."
    return True, email, ""


def _load_file_users() -> Dict[str, Dict[str, Any]]:
    if not _USERS_FILE.exists():
        return {}
    try:
        data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("Failed to read users file: %s", exc)
    return {}


def _save_file_users(users: Dict[str, Dict[str, Any]]) -> None:
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USERS_FILE.write_text(
        json.dumps(users, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _mongo_collection():
    try:
        from memory.mongodb import MongoDBClient

        client = MongoDBClient()
        col = client.db.users
        col.create_index("username", unique=True)
        return col
    except Exception as exc:
        logger.info("Mongo user store unavailable, using file fallback: %s", exc)
        return None


def mongo_users_available() -> bool:
    """True when signup/login can persist across machines (Streamlit Cloud)."""
    return _mongo_collection() is not None


def _lookup_keys(username: str) -> List[str]:
    """Match both name@gitgeeks.com and legacy local-part-only usernames."""
    raw = (username or "").strip().lower()
    if not raw:
        return []
    keys = [raw]
    if "@" in raw:
        local = raw.split("@", 1)[0]
        if local and local not in keys:
            keys.append(local)
        _, email, _ = normalize_user_id(raw)
        if email and email not in keys:
            keys.insert(0, email)
    else:
        _, email, _ = normalize_user_id(raw)
        if email and email not in keys:
            keys.insert(0, email)
    # Unique, email-first
    seen: List[str] = []
    for k in keys:
        if k and k not in seen:
            seen.append(k)
    return seen


def _get_user(username: str) -> Optional[Dict[str, Any]]:
    keys = _lookup_keys(username)
    if not keys:
        return None
    col = _mongo_collection()
    if col is not None:
        try:
            for key in keys:
                doc = col.find_one(
                    {"$or": [{"username": key}, {"email": key}]},
                    {"_id": 0},
                )
                if doc:
                    return doc
        except Exception as exc:
            logger.warning("Mongo get_user failed: %s", exc)

    users = _load_file_users()
    for key in keys:
        if key in users:
            return users[key]
        for doc in users.values():
            if not isinstance(doc, dict):
                continue
            if str(doc.get("username") or "").strip().lower() in keys:
                return doc
            if str(doc.get("email") or "").strip().lower() in keys:
                return doc
    return None


def _upsert_user(doc: Dict[str, Any]) -> Tuple[bool, str]:
    key = str(doc.get("username") or "").strip().lower()
    if not key:
        return False, "Missing username."

    # Always store canonical email fields when possible
    payload = dict(doc)
    payload["username"] = key
    if "@" in key and not payload.get("email"):
        payload["email"] = key

    col = _mongo_collection()
    if col is not None:
        try:
            col.update_one({"username": key}, {"$set": payload}, upsert=True)
            # Best-effort local mirror (helps laptop admin; Cloud disk is ephemeral)
            try:
                users = _load_file_users()
                users[key] = payload
                _save_file_users(users)
            except Exception:
                pass
            return True, ""
        except Exception as exc:
            logger.error("Mongo upsert failed: %s", exc)
            return False, "Could not save account to MongoDB. Check MONGODB_URI."

    # Local-only fallback (development without Mongo)
    users = _load_file_users()
    users[key] = payload
    try:
        _save_file_users(users)
        return True, ""
    except Exception as exc:
        logger.error("Could not persist user: %s", exc)
        return False, "Could not save account. Please try again."


def _user_status(user: Dict[str, Any]) -> str:
    """Legacy users without status are treated as approved."""
    status = str(user.get("status") or "").strip().lower()
    if not status:
        return STATUS_APPROVED
    return status


def signup(username: str, password: str, confirm_password: str = "") -> Tuple[bool, str]:
    """
    Submit a signup *request* for a @gitgeeks.com identity.

    Account stays pending until an admin approves it.
    """
    ok, email, err = normalize_user_id(username)
    if not ok:
        return False, err

    password = password or ""
    confirm_password = confirm_password if confirm_password != "" else password

    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."

    admin_user, _ = _admin_credentials()
    if email == admin_user.strip().lower() or email.split("@")[0] == admin_user.strip().lower():
        return False, "This username is reserved. Choose another."

    existing = _get_user(email)
    if existing:
        st = _user_status(existing)
        if st == STATUS_PENDING:
            return (
                False,
                "A signup request for this email is already pending admin approval.",
            )
        if st == STATUS_APPROVED:
            return False, "Account already exists. Please log in instead."
        if st == STATUS_REJECTED:
            return (
                False,
                "This signup was rejected. Contact admin if you need access.",
            )
        return False, "Username already exists. Please log in instead."

    # Cloud has no durable disk — never create accounts only in users.json there.
    if _mongo_collection() is None:
        return (
            False,
            "User directory is not connected (MongoDB). "
            "Ask admin to set MONGODB_URI in Streamlit secrets so signups persist.",
        )

    password_hash, salt = _hash_password(password)
    local = email.split("@")[0]
    doc = {
        "username": email,
        "email": email,
        "display_name": local,
        "password_hash": password_hash,
        "salt": salt,
        "status": STATUS_PENDING,
        "role": "user",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    saved, save_err = _upsert_user(doc)
    if not saved:
        return False, save_err or "Could not save signup request."

    return (
        True,
        "Signup request submitted. An admin must approve your @"
        f"{ALLOWED_EMAIL_DOMAIN} account before you can log in.",
    )


def is_admin(username: str) -> bool:
    """True only for the bootstrap admin account (APP_USERNAME)."""
    admin_user, _ = _admin_credentials()
    raw = (username or "").strip().lower()
    if not raw:
        return False
    admin_l = admin_user.strip().lower()
    if raw == admin_l:
        return True
    # Allow admin@gitgeeks.com if APP_USERNAME is the local part (or vice versa)
    if "@" not in admin_l and raw == f"{admin_l}@{ALLOWED_EMAIL_DOMAIN}":
        return True
    if "@" in admin_l and raw == admin_l.split("@")[0]:
        return True
    return False


def login(username: str, password: str) -> Tuple[bool, str]:
    """
    Authenticate a user.

    Accepts:
      - bootstrap admin from APP_USERNAME / APP_PASSWORD (always)
      - approved @gitgeeks.com users from MongoDB / local file
    """
    username_raw = (username or "").strip()
    password = password or ""
    if not username_raw or not password:
        return False, "Enter work email and password."

    admin_user, admin_pass = _admin_credentials()
    admin_l = admin_user.strip().lower()
    candidate = username_raw.lower()
    # Bootstrap admin may still use plain APP_USERNAME (not necessarily email)
    if candidate == admin_l or (
        "@" not in admin_l and candidate == f"{admin_l}@{ALLOWED_EMAIL_DOMAIN}"
    ):
        if password == admin_pass:
            return True, admin_user.strip()

    ok, email, err = normalize_user_id(username_raw)
    if not ok:
        # Non-email bootstrap already handled; otherwise domain error
        return False, err or "Invalid username or password."

    user = _get_user(email)
    if not user:
        return False, "Invalid username or password."

    if not _verify_password(password, user.get("password_hash", ""), user.get("salt", "")):
        return False, "Invalid username or password."

    status = _user_status(user)
    if status == STATUS_PENDING:
        return (
            False,
            "Your signup is pending admin approval. You cannot log in yet.",
        )
    if status == STATUS_REJECTED:
        return (
            False,
            "Your signup was rejected. Contact admin if you need access.",
        )
    if status != STATUS_APPROVED:
        return False, "Account is not active. Contact admin."

    return True, user.get("email") or user.get("username") or email


def list_users(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List users (file + mongo merge). Optionally filter by status."""
    by_key: Dict[str, Dict[str, Any]] = {}

    for key, doc in _load_file_users().items():
        if isinstance(doc, dict):
            row = dict(doc)
            row["username"] = row.get("username") or key
            by_key[str(row["username"]).lower()] = row

    col = _mongo_collection()
    if col is not None:
        try:
            for doc in col.find({}, {"_id": 0}):
                if not isinstance(doc, dict):
                    continue
                key = str(doc.get("username") or "").lower()
                if key:
                    by_key[key] = dict(doc)
        except Exception as exc:
            logger.warning("Mongo list_users failed: %s", exc)

    rows = list(by_key.values())
    for row in rows:
        row["status"] = _user_status(row)

    if status:
        want = status.strip().lower()
        rows = [r for r in rows if r.get("status") == want]

    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return rows


def list_pending_signups() -> List[Dict[str, Any]]:
    return list_users(status=STATUS_PENDING)


def set_user_status(username: str, status: str, actor: str = "") -> Tuple[bool, str]:
    """Approve or reject a signup request."""
    status = (status or "").strip().lower()
    if status not in (STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING):
        return False, "Invalid status."

    ok, email, err = normalize_user_id(username)
    # Pending records should already be emails; also allow raw key lookup
    key = email if ok else (username or "").strip().lower()
    if not key:
        return False, err or "Missing user id."

    user = _get_user(key)
    if not user:
        return False, "User not found."

    user = dict(user)
    user["status"] = status
    user["updated_at"] = _now_iso()
    if actor:
        user["reviewed_by"] = actor.strip().lower()
        user["reviewed_at"] = _now_iso()

    saved, save_err = _upsert_user(user)
    if not saved:
        return False, save_err or "Could not update user."

    if status == STATUS_APPROVED:
        return True, f"Approved {key}. They can log in now."
    if status == STATUS_REJECTED:
        return True, f"Rejected {key}."
    return True, f"Updated {key} → {status}."


def approve_user(username: str, actor: str = "") -> Tuple[bool, str]:
    return set_user_status(username, STATUS_APPROVED, actor=actor)


def reject_user(username: str, actor: str = "") -> Tuple[bool, str]:
    return set_user_status(username, STATUS_REJECTED, actor=actor)


def allowed_email_domain() -> str:
    return ALLOWED_EMAIL_DOMAIN


def person_label(username_or_email: str) -> str:
    """
    Human-friendly person name for admin monitoring.

    Prefer stored display_name; else derive from email local part
    (rishav.patel@gitgeeks.com → Rishav Patel).
    """
    key = (username_or_email or "").strip().lower()
    if not key:
        return "Unknown"

    user = _get_user(key)
    stored = ""
    if user:
        stored = str(user.get("display_name") or "").strip()
        if not stored:
            stored = str(user.get("email") or user.get("username") or "").strip()

    raw = stored or key
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    parts = [p for p in raw.replace("_", " ").replace(".", " ").split() if p]
    if not parts:
        return key
    return " ".join(p[:1].upper() + p[1:] for p in parts)
