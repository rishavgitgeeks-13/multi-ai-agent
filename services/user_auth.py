"""
Simple user auth for Streamlit login / signup.

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
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,40}$")
_PBKDF2_ITERATIONS = 120_000


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
            os.getenv("APP_PASSWORD", "admin123"),
        )


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


def _get_user(username: str) -> Optional[Dict[str, Any]]:
    key = username.strip().lower()
    col = _mongo_collection()
    if col is not None:
        try:
            doc = col.find_one({"username": key}, {"_id": 0})
            if doc:
                return doc
        except Exception as exc:
            logger.warning("Mongo get_user failed: %s", exc)

    users = _load_file_users()
    return users.get(key)


def signup(username: str, password: str, confirm_password: str = "") -> Tuple[bool, str]:
    """Create a new user account. Returns (ok, message)."""
    username = (username or "").strip()
    password = password or ""
    confirm_password = confirm_password if confirm_password != "" else password

    if not _USERNAME_RE.match(username):
        return False, "Username must be 3-40 characters (letters, numbers, . _ -)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."

    admin_user, _ = _admin_credentials()
    if username.lower() == admin_user.lower():
        return False, "This username is reserved. Choose another."

    if _get_user(username):
        return False, "Username already exists. Please log in instead."

    password_hash, salt = _hash_password(password)
    doc = {
        "username": username.lower(),
        "display_name": username,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    col = _mongo_collection()
    if col is not None:
        try:
            col.insert_one(dict(doc))
            return True, "Account created. You can log in now."
        except Exception as exc:
            # Duplicate key or transient — fall through to file
            logger.warning("Mongo signup failed, trying file: %s", exc)

    users = _load_file_users()
    if username.lower() in users:
        return False, "Username already exists. Please log in instead."
    users[username.lower()] = doc
    try:
        _save_file_users(users)
    except Exception as exc:
        logger.error("Could not persist signup: %s", exc)
        return False, "Could not save account. Please try again."
    return True, "Account created. You can log in now."


def is_admin(username: str) -> bool:
    """True only for the bootstrap admin account (APP_USERNAME)."""
    admin_user, _ = _admin_credentials()
    return bool(username) and username.strip().lower() == admin_user.strip().lower()


def login(username: str, password: str) -> Tuple[bool, str]:
    """
    Authenticate a user.

    Accepts:
      - bootstrap admin from APP_USERNAME / APP_PASSWORD
      - registered users from MongoDB / local file
    """
    username = (username or "").strip()
    password = password or ""
    if not username or not password:
        return False, "Enter username and password."

    admin_user, admin_pass = _admin_credentials()
    if username == admin_user and password == admin_pass:
        return True, admin_user

    user = _get_user(username)
    if not user:
        return False, "Invalid username or password."

    if _verify_password(password, user.get("password_hash", ""), user.get("salt", "")):
        return True, user.get("display_name") or username

    return False, "Invalid username or password."
