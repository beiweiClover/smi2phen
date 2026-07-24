"""Concurrency-safe SQLite persistence for deterministic workflows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import (
    DependencyRef,
    NodeRecord,
    PlannedNode,
    WorkflowPlan,
    WorkflowStatus,
    isoformat,
    utc_now,
)
from .state_machine import validate_run_transition, validate_transition


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode(value: str | None, default: Any) -> Any:
    return default if value is None else json.loads(value)


class WorkflowStore:
    """SQLite state store with transactional transitions and append-only events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    workflow_version TEXT NOT NULL,
                    mode TEXT,
                    evidence_mode TEXT,
                    config_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    input_state_json TEXT NOT NULL,
                    hardware_fingerprint TEXT,
                    input_scale_json TEXT NOT NULL DEFAULT '{}',
                    plan_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    run_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    progress REAL,
                    created_at TEXT NOT NULL,
                    queued_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    heartbeat_at TEXT,
                    worker_id TEXT,
                    queue_name TEXT,
                    error_json TEXT,
                    artifacts_json TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    resource_class TEXT NOT NULL DEFAULT 'cpu',
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    cache_key TEXT,
                    version INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, node_id, task_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS dependencies (
                    run_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    depends_on_node_id TEXT NOT NULL,
                    depends_on_task_id TEXT NOT NULL,
                    PRIMARY KEY (
                        run_id, node_id, task_id, depends_on_node_id, depends_on_task_id
                    ),
                    FOREIGN KEY (run_id, node_id, task_id)
                        REFERENCES nodes(run_id, node_id, task_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS fanouts (
                    run_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    consumer_node_id TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    items_json TEXT,
                    resolved INTEGER NOT NULL,
                    PRIMARY KEY (run_id, target_node_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    node_id TEXT,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(run_id, status);
                CREATE TABLE IF NOT EXISTS cache_entries (
                    node_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    PRIMARY KEY (node_id, task_id, cache_key)
                );
                CREATE TABLE IF NOT EXISTS node_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    hardware_fingerprint TEXT,
                    input_scale_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_history_node
                    ON node_history(node_id, hardware_fingerprint, status);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(nodes)").fetchall()
            }
            if "worker_id" not in columns:
                connection.execute("ALTER TABLE nodes ADD COLUMN worker_id TEXT")
            if "queue_name" not in columns:
                connection.execute("ALTER TABLE nodes ADD COLUMN queue_name TEXT")

    def create_run(
        self,
        *,
        run_id: str,
        run_dir: str,
        workflow_id: str,
        workflow_version: str,
        config_hash: str,
        config: Mapping[str, Any],
        input_state: Mapping[str, Any],
        hardware_fingerprint: str | None,
        input_scale: Mapping[str, Any],
    ) -> None:
        now = isoformat(utc_now())
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO runs(
                    run_id, run_dir, status, workflow_id, workflow_version,
                    config_hash, config_json, input_state_json, hardware_fingerprint,
                    input_scale_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_dir,
                    WorkflowStatus.PENDING.value,
                    workflow_id,
                    workflow_version,
                    config_hash,
                    _json(config),
                    _json(input_state),
                    hardware_fingerprint,
                    _json(input_scale),
                    now,
                    now,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="run_created",
                new_status=WorkflowStatus.PENDING,
                payload={"run_dir": run_dir},
            )

    def save_plan(self, plan: WorkflowPlan) -> None:
        now = isoformat(utc_now())
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT COUNT(*) AS count FROM nodes WHERE run_id=?", (plan.run_id,)
            ).fetchone()["count"]
            if existing:
                raise ValueError(f"run {plan.run_id!r} already has a persisted plan")
            for node in plan.nodes:
                self._insert_node(connection, plan.run_id, node, now)
                for dependency in node.dependencies:
                    connection.execute(
                        """INSERT INTO dependencies(
                            run_id, node_id, task_id, depends_on_node_id, depends_on_task_id
                        ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            plan.run_id,
                            node.node_id,
                            node.task_id,
                            dependency.node_id,
                            dependency.task_id,
                        ),
                    )
            for fanout in plan.fanouts:
                connection.execute(
                    """INSERT INTO fanouts(
                        run_id, source_node_id, target_node_id, consumer_node_id,
                        item_key, items_json, resolved
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        plan.run_id,
                        fanout.source_node_id,
                        fanout.target_node_id,
                        fanout.consumer_node_id,
                        fanout.item_key,
                        None if fanout.items is None else _json(list(fanout.items)),
                        int(fanout.resolved),
                    ),
                )
            connection.execute(
                """UPDATE runs SET mode=?, evidence_mode=?, plan_json=?, status=?,
                    updated_at=?, version=version+1 WHERE run_id=?""",
                (
                    plan.mode,
                    plan.evidence_mode,
                    _json(plan.to_dict()),
                    WorkflowStatus.READY.value,
                    now,
                    plan.run_id,
                ),
            )
            self._event(
                connection,
                run_id=plan.run_id,
                event_type="plan_created",
                old_status=WorkflowStatus.PENDING,
                new_status=WorkflowStatus.READY,
                payload={"mode": plan.mode, "node_count": len(plan.nodes)},
            )

    def _insert_node(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        node: PlannedNode,
        now: str,
    ) -> None:
        connection.execute(
            """INSERT INTO nodes(
                run_id, node_id, task_id, stage, status, created_at,
                finished_at, error_json, resource_class, parameters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                node.node_id,
                node.task_id,
                node.stage,
                node.initial_status.value,
                now,
                now
                if node.initial_status in {WorkflowStatus.SUCCEEDED, WorkflowStatus.SKIPPED}
                else None,
                None,
                node.resource_class,
                _json(node.parameters),
            ),
        )
        self._event(
            connection,
            run_id=run_id,
            node_id=node.node_id,
            task_id=node.task_id,
            event_type="node_created",
            new_status=node.initial_status,
            payload={"skip_reason": node.skip_reason} if node.skip_reason else {},
        )

    def materialize_fanout(
        self,
        run_id: str,
        target_node_id: str,
        items: Sequence[str],
        *,
        stage: str,
        resource_class: str,
    ) -> list[str]:
        normalized = tuple(str(item) for item in items)
        if len(normalized) != len(set(normalized)):
            raise ValueError("fan-out items must be unique")
        now = isoformat(utc_now())
        with self.transaction() as connection:
            fanout = connection.execute(
                "SELECT * FROM fanouts WHERE run_id=? AND target_node_id=?",
                (run_id, target_node_id),
            ).fetchone()
            if fanout is None:
                raise KeyError(f"unknown fan-out {target_node_id!r}")
            if fanout["resolved"]:
                existing = tuple(_decode(fanout["items_json"], []))
                if existing != normalized:
                    raise ValueError("fan-out has already been materialized with different items")
                return list(existing)
            for item in normalized:
                node = PlannedNode(
                    node_id=target_node_id,
                    task_id=item,
                    stage=stage,
                    dependencies=(DependencyRef(fanout["source_node_id"]),),
                    resource_class=resource_class,
                    parameters={fanout["item_key"]: item},
                )
                self._insert_node(connection, run_id, node, now)
                connection.execute(
                    """INSERT INTO dependencies VALUES (?, ?, ?, ?, ?)""",
                    (run_id, target_node_id, item, fanout["source_node_id"], "main"),
                )
            connection.execute(
                "UPDATE fanouts SET items_json=?, resolved=1 WHERE run_id=? AND target_node_id=?",
                (_json(list(normalized)), run_id, target_node_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="fanout_materialized",
                payload={"target_node_id": target_node_id, "items": list(normalized)},
            )
        return list(normalized)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id!r}")
        result = dict(row)
        for key in ("config_json", "input_state_json", "input_scale_json", "plan_json"):
            result[key.removesuffix("_json")] = _decode(
                result.pop(key), None if key == "plan_json" else {}
            )
        result["cancel_requested"] = bool(result["cancel_requested"])
        return result

    def get_node(self, run_id: str, node_id: str, task_id: str = "main") -> NodeRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE run_id=? AND node_id=? AND task_id=?",
                (run_id, node_id, task_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown node task {node_id}:{task_id}")
            dependencies = self._dependencies(connection, run_id, node_id, task_id)
        return self._node_record(row, dependencies)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT run_id FROM runs ORDER BY created_at").fetchall()
        return [self.get_run(row["run_id"]) for row in rows]

    def list_nodes(self, run_id: str) -> list[NodeRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM nodes WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
            return [
                self._node_record(
                    row,
                    self._dependencies(connection, run_id, row["node_id"], row["task_id"]),
                )
                for row in rows
            ]

    def _dependencies(
        self, connection: sqlite3.Connection, run_id: str, node_id: str, task_id: str
    ) -> tuple[DependencyRef, ...]:
        rows = connection.execute(
            """SELECT depends_on_node_id, depends_on_task_id FROM dependencies
               WHERE run_id=? AND node_id=? AND task_id=?
               ORDER BY depends_on_node_id, depends_on_task_id""",
            (run_id, node_id, task_id),
        ).fetchall()
        return tuple(DependencyRef(row[0], row[1]) for row in rows)

    @staticmethod
    def _node_record(row: sqlite3.Row, dependencies: tuple[DependencyRef, ...]) -> NodeRecord:
        return NodeRecord(
            run_id=row["run_id"],
            node_id=row["node_id"],
            task_id=row["task_id"],
            stage=row["stage"],
            status=WorkflowStatus(row["status"]),
            attempt=row["attempt"],
            progress=row["progress"],
            created_at=row["created_at"],
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            heartbeat_at=row["heartbeat_at"],
            worker_id=row["worker_id"],
            queue=row["queue_name"],
            error=_decode(row["error_json"], None),
            artifacts=tuple(_decode(row["artifacts_json"], [])),
            metrics=_decode(row["metrics_json"], {}),
            warnings=tuple(_decode(row["warnings_json"], [])),
            dependencies=dependencies,
            resource_class=row["resource_class"],
            parameters=_decode(row["parameters_json"], {}),
            cache_key=row["cache_key"],
            version=row["version"],
        )

    def transition_node(
        self,
        run_id: str,
        node_id: str,
        task_id: str,
        target: WorkflowStatus | str,
        *,
        event_type: str = "node_status_changed",
        payload: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        artifacts: Sequence[Mapping[str, Any]] | None = None,
        metrics: Mapping[str, Any] | None = None,
        warnings: Sequence[str] | None = None,
        cache_key: str | None = None,
        invalidation: bool = False,
        expected_version: int | None = None,
        worker_id: str | None = None,
        queue: str | None = None,
    ) -> NodeRecord:
        destination = WorkflowStatus(target)
        now = isoformat(utc_now())
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE run_id=? AND node_id=? AND task_id=?",
                (run_id, node_id, task_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown node task {node_id}:{task_id}")
            if expected_version is not None and row["version"] != expected_version:
                raise RuntimeError("concurrent node update detected")
            source, destination = validate_transition(
                row["status"], destination, invalidation=invalidation
            )
            fields: dict[str, Any] = {
                "status": destination.value,
                "version": row["version"] + 1,
            }
            if destination is WorkflowStatus.QUEUED:
                fields["queued_at"] = now
            if destination is WorkflowStatus.RUNNING:
                fields.update(
                    started_at=now,
                    heartbeat_at=now,
                    finished_at=None,
                    error_json=None,
                    progress=0.0,
                    attempt=row["attempt"] + 1,
                    worker_id=worker_id,
                    queue_name=queue,
                )
            if destination in {
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.BLOCKED,
                WorkflowStatus.CACHED,
                WorkflowStatus.SKIPPED,
                WorkflowStatus.CANCELLED,
            }:
                fields["finished_at"] = now
            if destination is WorkflowStatus.PENDING:
                fields.update(
                    queued_at=None,
                    started_at=None,
                    finished_at=None,
                    heartbeat_at=None,
                    progress=None,
                    error_json=None,
                    artifacts_json="[]",
                    metrics_json="{}",
                    warnings_json="[]",
                    cache_key=None,
                    worker_id=None,
                    queue_name=None,
                )
            if error is not None or destination is WorkflowStatus.FAILED:
                fields["error_json"] = _json(error) if error is not None else row["error_json"]
            if artifacts is not None:
                fields["artifacts_json"] = _json(list(artifacts))
            if metrics is not None:
                fields["metrics_json"] = _json(metrics)
            if warnings is not None:
                fields["warnings_json"] = _json(list(warnings))
            if cache_key is not None:
                fields["cache_key"] = cache_key
            assignments = ", ".join(f"{key}=?" for key in fields)
            connection.execute(
                f"UPDATE nodes SET {assignments} WHERE run_id=? AND node_id=? AND task_id=?",
                (*fields.values(), run_id, node_id, task_id),
            )
            self._event(
                connection,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                event_type=event_type,
                old_status=source,
                new_status=destination,
                payload=payload or {},
            )
            updated = connection.execute(
                "SELECT * FROM nodes WHERE run_id=? AND node_id=? AND task_id=?",
                (run_id, node_id, task_id),
            ).fetchone()
            dependencies = self._dependencies(connection, run_id, node_id, task_id)
        return self._node_record(updated, dependencies)

    def claim_node(
        self,
        run_id: str,
        node_id: str,
        task_id: str,
        *,
        attempt: int,
        worker_id: str,
        queue: str,
    ) -> NodeRecord:
        """Atomically reject duplicates/stale attempts and claim one queued task."""

        now = isoformat(utc_now())
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE run_id=? AND node_id=? AND task_id=?",
                (run_id, node_id, task_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown node task {node_id}:{task_id}")
            if row["status"] != WorkflowStatus.QUEUED.value:
                raise ValueError(f"duplicate_or_inactive_job:{row['status']}")
            expected_attempt = int(row["attempt"]) + 1
            if attempt != expected_attempt:
                raise ValueError(
                    f"stale_attempt:expected={expected_attempt}:received={attempt}"
                )
            connection.execute(
                """UPDATE nodes SET status=?, attempt=?, started_at=?, heartbeat_at=?,
                   finished_at=NULL, error_json=NULL, progress=0.0, worker_id=?, queue_name=?,
                   version=version+1 WHERE run_id=? AND node_id=? AND task_id=?""",
                (
                    WorkflowStatus.RUNNING.value,
                    attempt,
                    now,
                    now,
                    worker_id,
                    queue,
                    run_id,
                    node_id,
                    task_id,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                event_type="queue_job_claimed",
                old_status=WorkflowStatus.QUEUED,
                new_status=WorkflowStatus.RUNNING,
                payload={"attempt": attempt, "worker_id": worker_id, "queue": queue},
            )
            updated = connection.execute(
                "SELECT * FROM nodes WHERE run_id=? AND node_id=? AND task_id=?",
                (run_id, node_id, task_id),
            ).fetchone()
            dependencies = self._dependencies(connection, run_id, node_id, task_id)
        return self._node_record(updated, dependencies)

    def heartbeat(
        self, run_id: str, node_id: str, task_id: str, *, progress: float | None = None
    ) -> None:
        if progress is not None and not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        now = isoformat(utc_now())
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM nodes WHERE run_id=? AND node_id=? AND task_id=?",
                (run_id, node_id, task_id),
            ).fetchone()
            if row is None or row["status"] != WorkflowStatus.RUNNING.value:
                raise ValueError("heartbeats are accepted only for running nodes")
            connection.execute(
                """UPDATE nodes SET heartbeat_at=?, progress=COALESCE(?, progress),
                   version=version+1 WHERE run_id=? AND node_id=? AND task_id=?""",
                (now, progress, run_id, node_id, task_id),
            )

    def set_run_status(
        self,
        run_id: str,
        status: WorkflowStatus | str,
        *,
        event_type: str = "run_status_changed",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        destination = WorkflowStatus(status)
        now = isoformat(utc_now())
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown run {run_id!r}")
            source = WorkflowStatus(row["status"])
            if source == destination:
                return
            validate_run_transition(source, destination)
            fields: dict[str, Any] = {"status": destination.value, "updated_at": now}
            if destination is WorkflowStatus.RUNNING and row["started_at"] is None:
                fields["started_at"] = now
            if destination in {
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.BLOCKED,
                WorkflowStatus.CANCELLED,
            }:
                fields["finished_at"] = now
            else:
                fields["finished_at"] = None
            assignments = ", ".join(f"{key}=?" for key in fields)
            connection.execute(
                f"UPDATE runs SET {assignments}, version=version+1 WHERE run_id=?",
                (*fields.values(), run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type=event_type,
                old_status=source,
                new_status=destination,
                payload=payload or {},
            )

    def request_cancel(self, run_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE runs SET cancel_requested=1, updated_at=?, version=version+1 WHERE run_id=?",
                (isoformat(utc_now()), run_id),
            )
            self._event(connection, run_id=run_id, event_type="cancel_requested")

    def fanout_state(self, run_id: str, target_node_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM fanouts WHERE run_id=? AND target_node_id=?",
                (run_id, target_node_id),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["resolved"] = bool(value["resolved"])
        value["items"] = _decode(value.pop("items_json"), None)
        return value

    def downstream(self, run_id: str, roots: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
        discovered = set(roots)
        frontier = list(roots)
        with self._connect() as connection:
            while frontier:
                node_id, task_id = frontier.pop()
                rows = connection.execute(
                    """SELECT node_id, task_id FROM dependencies
                       WHERE run_id=? AND depends_on_node_id=?
                         AND (depends_on_task_id=? OR depends_on_task_id='*')""",
                    (run_id, node_id, task_id),
                ).fetchall()
                for row in rows:
                    key = (row["node_id"], row["task_id"])
                    if key not in discovered:
                        discovered.add(key)
                        frontier.append(key)
                fanout_consumers = connection.execute(
                    """SELECT consumer_node_id FROM fanouts
                       WHERE run_id=? AND target_node_id=?""",
                    (run_id, node_id),
                ).fetchall()
                for fanout in fanout_consumers:
                    key = (fanout["consumer_node_id"], "main")
                    if key not in discovered:
                        discovered.add(key)
                        frontier.append(key)
        return discovered

    def cache_get(self, node_id: str, task_id: str, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM cache_entries
                   WHERE node_id=? AND task_id=? AND cache_key=?""",
                (node_id, task_id, cache_key),
            ).fetchone()
        if row is None:
            return None
        return {
            "artifacts": _decode(row["artifacts_json"], []),
            "metrics": _decode(row["metrics_json"], {}),
            "source_run_id": row["source_run_id"],
            "created_at": row["created_at"],
        }

    def cache_put(
        self,
        *,
        node_id: str,
        task_id: str,
        cache_key: str,
        artifacts: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
        source_run_id: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO cache_entries(
                    node_id, task_id, cache_key, artifacts_json, metrics_json,
                    created_at, source_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    task_id,
                    cache_key,
                    _json(list(artifacts)),
                    _json(metrics),
                    isoformat(utc_now()),
                    source_run_id,
                ),
            )

    def add_history(
        self,
        *,
        node_id: str,
        task_id: str,
        duration_seconds: float,
        hardware_fingerprint: str | None,
        input_scale: Mapping[str, Any],
        metrics: Mapping[str, Any],
        status: WorkflowStatus,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO node_history(
                    node_id, task_id, duration_seconds, hardware_fingerprint,
                    input_scale_json, metrics_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    task_id,
                    duration_seconds,
                    hardware_fingerprint,
                    _json(input_scale),
                    _json(metrics),
                    status.value,
                    isoformat(utc_now()),
                ),
            )

    def history(
        self, node_id: str, hardware_fingerprint: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM node_history WHERE node_id=? AND status IN ('succeeded','cached')"
        values: list[Any] = [node_id]
        if hardware_fingerprint is not None:
            query += " AND hardware_fingerprint=?"
            values.append(hardware_fingerprint)
        query += " ORDER BY history_id DESC LIMIT 100"
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [
            {
                **dict(row),
                "input_scale": _decode(row["input_scale_json"], {}),
                "metrics": _decode(row["metrics_json"], {}),
            }
            for row in rows
        ]

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY event_id", (run_id,)
            ).fetchall()
        return [{**dict(row), "payload": _decode(row["payload_json"], {})} for row in rows]

    def _event(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        event_type: str,
        node_id: str | None = None,
        task_id: str | None = None,
        old_status: WorkflowStatus | None = None,
        new_status: WorkflowStatus | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO events(
                run_id, node_id, task_id, event_type, old_status,
                new_status, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                node_id,
                task_id,
                event_type,
                None if old_status is None else old_status.value,
                None if new_status is None else new_status.value,
                _json(payload or {}),
                isoformat(utc_now()),
            ),
        )
