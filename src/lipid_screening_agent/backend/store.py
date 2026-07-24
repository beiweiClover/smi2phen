"""Single SQLite business store for V3 sessions, messages, runs, and inputs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class V3Store:
    """Small store sharing one database file with the deterministic WorkflowStore."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS v3_sessions (
                    thread_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS v3_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES v3_sessions(thread_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS v3_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    disease_json TEXT NOT NULL,
                    workflow_created INTEGER NOT NULL DEFAULT 0,
                    plan_previewed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES v3_sessions(thread_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS v3_inputs (
                    run_id TEXT NOT NULL,
                    input_key TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, input_key),
                    FOREIGN KEY(run_id) REFERENCES v3_runs(run_id) ON DELETE CASCADE
                );
                """
            )
            run_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(v3_runs)").fetchall()
            }
            if "plan_previewed" not in run_columns:
                connection.execute(
                    "ALTER TABLE v3_runs ADD COLUMN plan_previewed INTEGER NOT NULL DEFAULT 0"
                )

    def create_session(self, thread_id: str) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO v3_sessions(thread_id, created_at, updated_at) VALUES (?, ?, ?)",
                (thread_id, now, now),
            )
        return self.session(thread_id)

    def session(self, thread_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM v3_sessions WHERE thread_id=?", (thread_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown session {thread_id!r}")
        return dict(row)

    def sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    session.*,
                    run.disease_json,
                    run.workflow_created,
                    run.plan_previewed,
                    (
                        SELECT COUNT(*) FROM v3_messages AS message
                        WHERE message.thread_id=session.thread_id
                    ) AS message_count
                FROM v3_sessions AS session
                LEFT JOIN v3_runs AS run ON run.run_id=session.run_id
                ORDER BY session.updated_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            encoded_disease = value.pop("disease_json", None)
            value["disease"] = json.loads(encoded_disease) if encoded_disease else None
            value["workflow_created"] = bool(value.get("workflow_created"))
            value["plan_previewed"] = bool(value.get("plan_previewed"))
            values.append(value)
        return values

    def append_message(self, thread_id: str, message: dict[str, Any]) -> None:
        now = _now()
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM v3_sessions WHERE thread_id=?", (thread_id,)
            ).fetchone() is None:
                raise KeyError(f"unknown session {thread_id!r}")
            connection.execute(
                "INSERT INTO v3_messages(thread_id, message_json, created_at) VALUES (?, ?, ?)",
                (thread_id, _json(message), now),
            )
            connection.execute(
                "UPDATE v3_sessions SET updated_at=? WHERE thread_id=?", (now, thread_id)
            )

    def messages(self, thread_id: str, *, limit: int = 24) -> list[dict[str, Any]]:
        return [
            record["message"]
            for record in self.message_records(thread_id, limit=limit)
        ]

    def message_records(
        self, thread_id: str, *, limit: int = 2000
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, message_json, created_at FROM (
                    SELECT id, message_json, created_at FROM v3_messages
                    WHERE thread_id=? ORDER BY id DESC LIMIT ?
                ) ORDER BY id
                """,
                (thread_id, max(1, min(limit, 5000))),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "message": json.loads(row["message_json"]),
            }
            for row in rows
        ]

    def create_run(self, run_id: str, thread_id: str, disease: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            session = connection.execute(
                "SELECT run_id FROM v3_sessions WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown session {thread_id!r}")
            if session["run_id"] is not None:
                raise ValueError("this V3 session already has a run")
            connection.execute(
                """
                INSERT INTO v3_runs(run_id, thread_id, disease_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, thread_id, _json(disease), now, now),
            )
            connection.execute(
                "UPDATE v3_sessions SET run_id=?, updated_at=? WHERE thread_id=?",
                (run_id, now, thread_id),
            )
        return self.run(run_id)

    def run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM v3_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id!r}")
        value = dict(row)
        value["disease"] = json.loads(value.pop("disease_json"))
        value["workflow_created"] = bool(value["workflow_created"])
        value["plan_previewed"] = bool(value["plan_previewed"])
        return value

    def run_for_thread(self, thread_id: str) -> dict[str, Any] | None:
        session = self.session(thread_id)
        return None if session["run_id"] is None else self.run(str(session["run_id"]))

    def mark_workflow_created(self, run_id: str) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE v3_runs
                SET workflow_created=1, plan_previewed=1, updated_at=?
                WHERE run_id=?
                """,
                (now, run_id),
            )
            connection.execute(
                "UPDATE v3_sessions SET updated_at=? WHERE run_id=?",
                (now, run_id),
            )

    def mark_plan_previewed(self, run_id: str) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE v3_runs SET plan_previewed=1, updated_at=? WHERE run_id=?",
                (now, run_id),
            )
            connection.execute(
                "UPDATE v3_sessions SET updated_at=? WHERE run_id=?",
                (now, run_id),
            )

    def inputs_locked(self, run_id: str) -> bool:
        run = self.run(run_id)
        if not run["workflow_created"]:
            return False
        with self._connect() as connection:
            workflow = connection.execute(
                "SELECT status, started_at FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if workflow is None:
            return True
        return bool(workflow["started_at"]) or workflow["status"] not in {
            "pending",
            "ready",
        }

    def put_input(
        self,
        *,
        run_id: str,
        input_key: str,
        original_name: str,
        stored_path: str,
        size_bytes: int,
        sha256: str,
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO v3_inputs(
                    run_id, input_key, original_name, stored_path, size_bytes, sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, input_key) DO UPDATE SET
                    original_name=excluded.original_name,
                    stored_path=excluded.stored_path,
                    size_bytes=excluded.size_bytes,
                    sha256=excluded.sha256,
                    created_at=excluded.created_at
                """,
                (run_id, input_key, original_name, stored_path, size_bytes, sha256, now),
            )
            workflow = connection.execute(
                "SELECT status, started_at FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if workflow is not None:
                if workflow["started_at"] or workflow["status"] not in {"pending", "ready"}:
                    raise ValueError("inputs are locked after workflow execution starts")
                connection.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
            connection.execute(
                """
                UPDATE v3_runs
                SET workflow_created=0, plan_previewed=0, updated_at=?
                WHERE run_id=?
                """,
                (now, run_id),
            )
            connection.execute(
                "UPDATE v3_sessions SET updated_at=? WHERE run_id=?",
                (now, run_id),
            )
        return self.input(run_id, input_key)

    def input(self, run_id: str, input_key: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM v3_inputs WHERE run_id=? AND input_key=?", (run_id, input_key)
            ).fetchone()
        if row is None:
            raise KeyError(f"missing input {input_key!r}")
        return dict(row)

    def inputs(self, run_id: str) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM v3_inputs WHERE run_id=? ORDER BY input_key", (run_id,)
            ).fetchall()
        return {str(row["input_key"]): dict(row) for row in rows}


__all__ = ["V3Store"]
