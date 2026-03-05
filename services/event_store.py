
"""
services/event_store.py
========================
SQLite-backed event persistence layer for crowd behaviour alerts.

Responsibility
--------------
Save every abnormal behaviour event produced by the analysis pipeline into
a local SQLite database so that:

* The dashboard backend can query recent events without re-reading the
  pipeline's in-memory history.
* Events survive server restarts.
* Analysts can run offline queries against the database.

Only events whose label is **not** ``normal`` or ``insufficient_data`` are
persisted.  This keeps the database focused on actionable signals.

Schema
------
::

    CREATE TABLE events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp    TEXT    NOT NULL,          -- ISO-8601 UTC
        stream_id    TEXT    NOT NULL,
        frame_index  INTEGER NOT NULL,
        behavior     TEXT    NOT NULL,
        confidence   REAL    NOT NULL,
        people_count INTEGER NOT NULL,
        density      REAL    NOT NULL
    )

Usage
-----
    from services.event_store import EventStore

    store = EventStore()            # default: crowd_events.db
    store.init_db()

    store.save_event(
        stream_id="abc123",
        frame_index=500,
        behavior="panic",
        confidence=0.87,
        people_count=34,
        density=2.7e-5,
    )

    events = store.get_recent_events(limit=20)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

logger = logging.getLogger("crowd_analysis.services.event_store")

# Labels that represent actionable alerts worth persisting
_ABNORMAL_BEHAVIORS = frozenset({
    "running", "panic", "violence", "suspicion", "crowd_surge",
})

# Default database path (relative to the project root)
_DEFAULT_DB_PATH = Path("crowd_events.db")


class EventStore:
    """
    Thread-safe SQLite event store.

    Parameters
    ----------
    db_path : str | Path
        Filesystem path to the SQLite database file.
        The file is created automatically on first ``init_db()`` call.
    auto_init : bool
        If True (default) call ``init_db()`` in ``__init__`` so the table
        exists before any writes are attempted.
    """

    def __init__(
        self,
        db_path:   str | Path = _DEFAULT_DB_PATH,
        auto_init: bool       = True,
    ) -> None:
        self._db_path = Path(db_path)
        self._lock    = threading.Lock()

        if auto_init:
            self.init_db()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """
        Create the ``events`` table if it does not already exist.

        Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        try:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp    TEXT    NOT NULL,
                        stream_id    TEXT    NOT NULL,
                        frame_index  INTEGER NOT NULL,
                        behavior     TEXT    NOT NULL,
                        confidence   REAL    NOT NULL,
                        people_count INTEGER NOT NULL,
                        density      REAL    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_stream_id
                    ON events (stream_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_events_timestamp
                    ON events (timestamp)
                """)
            logger.info(
                "EventStore initialised — db_path='%s'", self._db_path
            )
        except sqlite3.Error as exc:
            logger.error("EventStore.init_db failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_event(
        self,
        stream_id:    str,
        frame_index:  int,
        behavior:     str,
        confidence:   float,
        people_count: int,
        density:      float,
        timestamp:    Optional[str] = None,
    ) -> Optional[int]:
        """
        Persist one crowd behaviour event.

        Only behaviours in ``_ABNORMAL_BEHAVIORS`` are written.  Normal /
        insufficient-data frames are silently ignored to keep the database
        focused on actionable alerts.

        Parameters
        ----------
        stream_id    : stream identifier (from ``StreamRegistry``).
        frame_index  : 0-based frame counter from the pipeline.
        behavior     : behaviour label string (e.g. ``"panic"``).
        confidence   : classifier confidence in [0, 1].
        people_count : number of confirmed tracks in the frame.
        density      : crowd density (persons / px²) from ``CrowdMetricsAnalyzer``.
        timestamp    : optional ISO-8601 UTC timestamp; auto-generated if None.

        Returns
        -------
        int | None — row ``id`` of the inserted record, or None if skipped.
        """
        if behavior not in _ABNORMAL_BEHAVIORS:
            logger.debug(
                "save_event: skipping behavior='%s' (not abnormal).", behavior
            )
            return None

        ts = timestamp or _utc_now()

        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO events
                        (timestamp, stream_id, frame_index, behavior,
                         confidence, people_count, density)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts, stream_id, frame_index, behavior,
                     round(float(confidence), 6),
                     int(people_count),
                     round(float(density), 10)),
                )
                row_id = cursor.lastrowid
                logger.info(
                    "Event saved — id=%d  stream=%s  frame=%d  behavior=%s  conf=%.2f",
                    row_id, stream_id, frame_index, behavior, confidence,
                )
                return row_id
        except sqlite3.Error as exc:
            logger.error(
                "EventStore.save_event failed: %s", exc, exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_recent_events(
        self,
        limit:     int           = 50,
        stream_id: Optional[str] = None,
        behavior:  Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Fetch the most recently stored events.

        Parameters
        ----------
        limit     : maximum number of rows to return.
        stream_id : filter to a specific stream (optional).
        behavior  : filter to a specific behaviour label (optional).

        Returns
        -------
        list of dicts, newest-first.
        Each dict has keys:
        ``id``, ``timestamp``, ``stream_id``, ``frame_index``,
        ``behavior``, ``confidence``, ``people_count``, ``density``.
        """
        sql    = "SELECT * FROM events"
        params: list = []
        wheres: list = []

        if stream_id is not None:
            wheres.append("stream_id = ?")
            params.append(stream_id)
        if behavior is not None:
            wheres.append("behavior = ?")
            params.append(behavior)

        if wheres:
            sql += " WHERE " + " AND ".join(wheres)

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error(
                "EventStore.get_recent_events failed: %s", exc, exc_info=True
            )
            return []

    def get_event_counts_by_behavior(
        self,
        stream_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Return a count of events grouped by behaviour label.

        Parameters
        ----------
        stream_id : filter to a specific stream (optional).

        Returns
        -------
        dict mapping behavior label → event count.
        """
        sql    = "SELECT behavior, COUNT(*) AS cnt FROM events"
        params: list = []

        if stream_id is not None:
            sql += " WHERE stream_id = ?"
            params.append(stream_id)

        sql += " GROUP BY behavior ORDER BY cnt DESC"

        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return {row[0]: row[1] for row in rows}
        except sqlite3.Error as exc:
            logger.error(
                "EventStore.get_event_counts_by_behavior failed: %s",
                exc, exc_info=True,
            )
            return {}

    def get_total_event_count(self, stream_id: Optional[str] = None) -> int:
        """Return the total number of stored events, optionally per stream."""
        sql    = "SELECT COUNT(*) FROM events"
        params: list = []

        if stream_id is not None:
            sql += " WHERE stream_id = ?"
            params.append(stream_id)

        try:
            with self._connect() as conn:
                return int(conn.execute(sql, params).fetchone()[0])
        except sqlite3.Error:
            return 0

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_old_events(self, keep_latest: int = 10_000) -> int:
        """
        Delete old events keeping only the ``keep_latest`` most recent rows.

        Returns the number of rows deleted.
        """
        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM events
                    WHERE id NOT IN (
                        SELECT id FROM events
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    """,
                    (keep_latest,),
                )
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(
                        "EventStore.purge_old_events: deleted %d rows.", deleted
                    )
                return deleted
        except sqlite3.Error as exc:
            logger.error(
                "EventStore.purge_old_events failed: %s", exc, exc_info=True
            )
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager that yields an auto-committing SQLite connection.

        ``check_same_thread=False`` is required because the store is shared
        across FastAPI request handlers (different OS threads) while the
        write lock (``self._lock``) serialises concurrent writes at the
        Python level.
        """
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=10,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def __repr__(self) -> str:
        return f"EventStore(db_path='{self._db_path}')"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()