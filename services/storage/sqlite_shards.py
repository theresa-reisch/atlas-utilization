"""
SQLite-backed shard storage for invariant-mass arrays.

This module stores arrays in a small number of files to avoid inode/file-count
explosion from per-combination .npy outputs.

Each row holds one chunk of invariant masses and, optionally, the matching
per-entry event weights (MC). Rows without weights read back as weight 1, so
data and old shards keep working.
"""

from __future__ import annotations

import io
import os
import sqlite3
import zlib
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# A value to store: either a bare mass array (unweighted) or (masses, weights).
ArrayOrPair = Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]


class SqliteArrayShardWriter:
    """Append-only writer for array shards keyed by signature."""

    def __init__(self, db_path: str, table_name: str = "array_chunks"):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        # WAL needs shared-memory (-shm) locking that some network/parallel
        # filesystems (e.g. BeeGFS) don't support: readers of a WAL db can hang
        # there. Set SQLITE_JOURNAL_MODE=DELETE on such clusters.
        journal_mode = os.environ.get("SQLITE_JOURNAL_MODE", "WAL")
        self.conn.execute(f"PRAGMA journal_mode={journal_mode};")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature TEXT NOT NULL,
                n_entries INTEGER NOT NULL,
                payload BLOB NOT NULL,
                weights BLOB
            )
            """
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signature ON {table_name}(signature)"
        )

    def append_array(
        self, signature: str, arr: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> None:
        """Append one numpy array chunk (and optional weights) under a signature."""
        row = _make_row(signature, arr, weights)
        if row is None:
            return
        self.conn.execute(
            f"INSERT INTO {self.table_name}(signature, n_entries, payload, weights) VALUES (?, ?, ?, ?)",
            row,
        )

    def append_many(self, signature_to_array: Dict[str, ArrayOrPair]) -> int:
        """Append many arrays in a single transaction.

        Values are either a mass array or a ``(masses, weights)`` tuple.
        """
        rows = []
        for signature, value in signature_to_array.items():
            arr, weights = value if isinstance(value, tuple) else (value, None)
            row = _make_row(signature, arr, weights)
            if row is not None:
                rows.append(row)
        if not rows:
            return 0
        self.conn.executemany(
            f"INSERT INTO {self.table_name}(signature, n_entries, payload, weights) VALUES (?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def _make_row(signature: str, arr: np.ndarray, weights: Optional[np.ndarray]):
    if arr.size == 0:
        return None
    w_blob = None
    if weights is not None:
        if len(weights) != len(arr):
            raise ValueError(
                f"{signature}: {len(arr)} masses but {len(weights)} weights"
            )
        w_blob = _serialize_array(np.asarray(weights, dtype=np.float64))
    return (signature, int(arr.size), _serialize_array(arr), w_blob)


def _has_weights_column(conn: sqlite3.Connection, table_name: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(c[1] == "weights" for c in cols)


def _decode(payload: bytes, w_blob: Optional[bytes]) -> Tuple[np.ndarray, np.ndarray]:
    arr = _deserialize_array(payload)
    if w_blob is None:
        return arr, np.ones(len(arr), dtype=np.float64)
    return arr, _deserialize_array(w_blob)


def list_signatures(db_path: str, table_name: str = "array_chunks") -> List[str]:
    """Return all distinct signatures in a shard DB."""
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT signature FROM {table_name} ORDER BY signature"
        ).fetchall()
    return [r[0] for r in rows]


def iter_arrays_for_signature(
    db_path: str,
    signature: str,
    table_name: str = "array_chunks",
) -> Iterator[np.ndarray]:
    """Yield all mass chunks for one signature (weights dropped)."""
    for arr, _w in iter_weighted_arrays_for_signature(db_path, signature, table_name):
        yield arr


def iter_weighted_arrays_for_signature(
    db_path: str,
    signature: str,
    table_name: str = "array_chunks",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(masses, weights)`` for every chunk of one signature.

    Chunks stored without weights come back with weight 1.
    """
    if not os.path.exists(db_path):
        return
    with sqlite3.connect(db_path) as conn:
        w_col = "weights" if _has_weights_column(conn, table_name) else "NULL"
        rows = conn.execute(
            f"SELECT payload, {w_col} FROM {table_name} WHERE signature = ?",
            (signature,),
        ).fetchall()
    for payload, w_blob in rows:
        yield _decode(payload, w_blob)


def iter_all_chunks(
    db_path: str, table_name: str = "array_chunks"
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (signature, chunk_array) for every row."""
    if not os.path.exists(db_path):
        return
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT signature, payload FROM {table_name} ORDER BY id"
        ).fetchall()
    for signature, payload in rows:
        yield signature, _deserialize_array(payload)


def get_total_entries(db_path: str, table_name: str = "array_chunks") -> int:
    """Return total entry count from metadata column."""
    if not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT COALESCE(SUM(n_entries), 0) FROM {table_name}").fetchone()
    return int(row[0] if row else 0)


def _serialize_array(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=1)


def _deserialize_array(payload: bytes) -> np.ndarray:
    raw = zlib.decompress(payload)
    buffer = io.BytesIO(raw)
    return np.load(buffer, allow_pickle=False)
