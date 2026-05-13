"""SQLite-backed user store for MediAid AI authentication.

All passwords are hashed with bcrypt. Connections are opened per-call so the
module is safe to use from Streamlit's worker threads.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import bcrypt

DB_PATH = Path(__file__).parent / "users.db"

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "Admin@2025"
DEFAULT_ADMIN_EMAIL = "admin@mediaid.local"
DEFAULT_ADMIN_NAME = "MediAid Admin"

RESET_TOKEN_TTL = timedelta(hours=1)


@contextmanager
def _connect():
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def init_db() -> None:
    """Create the users table and seed the default admin account if missing."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT,
                email TEXT UNIQUE,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT 'local',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                reset_token TEXT,
                reset_token_expiry TIMESTAMP
            )
            """
        )
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)
        ).fetchone()
        if not existing:
            conn.execute(
                """
                INSERT INTO users (full_name, email, username, password_hash, provider)
                VALUES (?, ?, ?, ?, 'local')
                """,
                (
                    DEFAULT_ADMIN_NAME,
                    DEFAULT_ADMIN_EMAIL,
                    DEFAULT_ADMIN_USERNAME,
                    _hash_password(DEFAULT_ADMIN_PASSWORD),
                ),
            )


def create_user(
    full_name: str,
    email: str,
    username: str,
    password: str,
    provider: str = "local",
) -> tuple[bool, str]:
    """Insert a new user. Returns (ok, message)."""
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."
    if not email:
        return False, "Email is required."
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO users (full_name, email, username, password_hash, provider)
                VALUES (?, ?, ?, ?, ?)
                """,
                (full_name, email, username, _hash_password(password), provider),
            )
        return True, "Account created successfully."
    except sqlite3.IntegrityError as exc:
        msg = str(exc).lower()
        if "users.username" in msg:
            return False, "That username is already taken."
        if "users.email" in msg:
            return False, "An account with that email already exists."
        return False, "Could not create account (duplicate value)."


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict]:
    if not username:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
        ).fetchone()
    return _row_to_dict(row)


def get_user_by_email(email: str) -> Optional[dict]:
    if not email:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    return _row_to_dict(row)


def verify_password(username: str, password: str) -> Optional[dict]:
    """Return the user dict if credentials are valid, else None."""
    user = get_user_by_username(username)
    if not user:
        return None
    if user.get("provider") != "local":
        return None
    if _check_password(password, user["password_hash"]):
        return user
    return None


def create_reset_token(email: str) -> Optional[str]:
    """Generate a reset token for the given email. Returns the token, or None
    if no user exists with that email."""
    user = get_user_by_email(email)
    if not user:
        return None
    token = secrets.token_urlsafe(32)
    expiry = datetime.utcnow() + RESET_TOKEN_TTL
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?",
            (token, expiry, user["id"]),
        )
    return token


def verify_reset_token(token: str, new_password: Optional[str] = None) -> tuple[bool, str]:
    """Validate a reset token. If new_password is provided, also rotate the
    password and clear the token. Returns (ok, message)."""
    if not token:
        return False, "Reset token is required."
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, reset_token_expiry FROM users WHERE reset_token = ?",
            (token,),
        ).fetchone()
        if not row:
            return False, "Invalid reset token."
        expiry = row["reset_token_expiry"]
        if isinstance(expiry, str):
            try:
                expiry = datetime.fromisoformat(expiry)
            except ValueError:
                expiry = None
        if not expiry or expiry < datetime.utcnow():
            return False, "Reset token has expired."
        if new_password is None:
            return True, "Token is valid."
        conn.execute(
            """
            UPDATE users
               SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL
             WHERE id = ?
            """,
            (_hash_password(new_password), row["id"]),
        )
    return True, "Password updated."


# Initialise the database on import so the default admin always exists.
init_db()
