"""SQLite-backed, thread-safe API key store.

Follows the same singleton pattern as JobStore: a reentrant lock guards all
SQLite access and a class-level ``default()`` classmethod provides a
module-level singleton for convenience.
"""
from __future__ import annotations

import hmac
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any


class KeyStore:
    """Thread-safe, SQLite-backed API key store.

    Each key is stored as an HMAC-SHA256 digest.  The plaintext key is
    returned *only* at creation time (like GitHub's SSH-key setup).

    The store is file-backed by default (``ruitong-keys.db`` in the CWD).
    Pass ``:memory:`` explicitly for test isolation.
    """

    _default_instance: KeyStore | None = None

    @classmethod
    def default(cls) -> KeyStore:
        """Return the module-level singleton (in-memory when no path set)."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def __init__(self, db_path: str = "") -> None:
        path = db_path or "ruitong-keys.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Lifecycle ────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA busy_timeout=5000;
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id      TEXT PRIMARY KEY,
                    key_hash    TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    prefix      TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    last_used_at TEXT,
                    is_active   INTEGER NOT NULL DEFAULT 1
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_hash
                    ON api_keys(key_hash);
            """)
            self._conn.commit()

    # ── CRUD ─────────────────────────────────────────────────────────

    def create_key(self, name: str) -> tuple[str, str]:
        """Generate a new API key and return *(key_id, plaintext_key)*.

        The *plaintext_key* must be saved immediately — it is never
        recoverable after this call.

        Args:
            name: Human-readable name for the key.

        Returns:
            A tuple of (key_id, plaintext_key) where *plaintext_key* starts
            with ``rt_``.
        """
        key_id = secrets.token_hex(16)  # 32-char hex
        plaintext = f"rt_{secrets.token_hex(24)}"  # rt_ + 48 hex chars
        key_hash = hmac.new(
            plaintext.encode("utf-8"),
            b"",
            "sha256",
        ).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO api_keys (key_id, key_hash, name, prefix, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key_id, key_hash, name, plaintext[:11], now),
            )
            self._conn.commit()
        return key_id, plaintext

    def authenticate(self, provided_key: str) -> str | None:
        """Return the key_id if *provided_key* is valid and active, else ``None``."""
        candidate_hash = hmac.new(
            provided_key.encode("utf-8"),
            b"",
            "sha256",
        ).hexdigest()
        with self._lock:
            row = self._conn.execute(
                """SELECT key_id FROM api_keys
                   WHERE key_hash = ? AND is_active = 1""",
                (candidate_hash,),
            ).fetchone()
        if row is None:
            return None
        self.update_last_used(row["key_id"])
        return row["key_id"]

    def list_keys(self) -> list[dict[str, Any]]:
        """Return metadata for every key (never returns hashes)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_id, name, prefix, created_at, last_used_at, is_active "
                "FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_key(self, key_id: str) -> bool:
        """Deactivate a key.  Returns ``True`` if a row was updated."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_id = ? AND is_active = 1",
                (key_id,),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def update_last_used(self, key_id: str) -> None:
        """Mark the given key as recently used — debounced to ~5 min."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """UPDATE api_keys SET last_used_at = ?
                   WHERE key_id = ?
                     AND (last_used_at IS NULL
                          OR last_used_at < datetime('now', '-5 minutes'))""",
                (now, key_id),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()