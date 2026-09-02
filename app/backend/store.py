"""SQLite persistence for reviewed applications.

One row per `/review` call: the submitted application, the model
probability, the decision, the full SHAP explanation, and the
adverse-action notice (when the decision was a denial). Nested structures
are stored as JSON text -- this is a demo audit log, not an analytics store,
and stdlib `sqlite3` keeps the dependency surface small.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    probability    REAL NOT NULL,
    decision       TEXT NOT NULL,
    application    TEXT NOT NULL,   -- JSON
    explanation    TEXT NOT NULL,   -- JSON
    adverse_action TEXT             -- JSON or NULL
);
"""


class ReviewStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def save(
        self,
        *,
        model_version: str,
        probability: float,
        decision: str,
        application: dict[str, Any],
        explanation: dict[str, Any],
        adverse_action: dict[str, Any] | None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reviews (created_at, model_version, probability,
                                     decision, application, explanation,
                                     adverse_action)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    model_version,
                    float(probability),
                    decision,
                    json.dumps(application),
                    json.dumps(explanation),
                    json.dumps(adverse_action) if adverse_action is not None else None,
                ),
            )
            return int(cur.lastrowid)

    def get(self, review_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reviews WHERE id = ?", (review_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def list(
        self, limit: int = 50, offset: int = 0, decision: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM reviews"
        params: list[Any] = []
        if decision is not None:
            query += " WHERE decision = ?"
            params.append(decision)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "model_version": row["model_version"],
        "probability": row["probability"],
        "decision": row["decision"],
        "application": json.loads(row["application"]),
        "explanation": json.loads(row["explanation"]),
        "adverse_action": (
            json.loads(row["adverse_action"])
            if row["adverse_action"] is not None
            else None
        ),
    }
