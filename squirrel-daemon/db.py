from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
from typing import Iterable, List, Dict, Any, Optional


@dataclass
class Click:
    id: int
    created_at: float
    pan: float
    tilt: float
    x_px: float
    y_px: float
    img_w: float
    img_h: float


class ClickStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        base = Path(__file__).parent.parent
        self.path = (db_path or (base / "clicks.db")).resolve()
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    pan REAL NOT NULL,
                    tilt REAL NOT NULL,
                    x_px REAL NOT NULL,
                    y_px REAL NOT NULL,
                    img_w REAL NOT NULL,
                    img_h REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clicks_created_at ON clicks(created_at DESC)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS aim_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    start_pan REAL NOT NULL,
                    start_tilt REAL NOT NULL,
                    dx REAL NOT NULL,
                    dy REAL NOT NULL,
                    dpan REAL NOT NULL,
                    dtilt REAL NOT NULL,
                    result_pan REAL NOT NULL,
                    result_tilt REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_aim_created_at ON aim_logs(created_at DESC)")

    def record(self, pan: float, tilt: float, x_px: float, y_px: float, img_w: float, img_h: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO clicks(created_at, pan, tilt, x_px, y_px, img_w, img_h) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), pan, tilt, x_px, y_px, img_w, img_h),
            )
            return int(cur.lastrowid)

    def list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(1, min(1000, int(limit)))
        offset = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, pan, tilt, x_px, y_px, img_w, img_h FROM clicks ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM clicks")
            return cur.rowcount if cur.rowcount is not None else 0

    # --- Aim logs ---
    def record_aim(self, *, start_pan: float, start_tilt: float, dx: float, dy: float, dpan: float, dtilt: float, result_pan: float, result_tilt: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO aim_logs(created_at, start_pan, start_tilt, dx, dy, dpan, dtilt, result_pan, result_tilt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), start_pan, start_tilt, dx, dy, dpan, dtilt, result_pan, result_tilt),
            )
            return int(cur.lastrowid)

    def list_aims(self, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(1, min(5000, int(limit)))
        offset = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, start_pan, start_tilt, dx, dy, dpan, dtilt, result_pan, result_tilt FROM aim_logs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]
