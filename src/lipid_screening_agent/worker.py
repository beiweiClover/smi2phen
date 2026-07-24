"""Redis workflow consumer with attempt and lease validation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import redis

from lipid_screening_agent.orchestrator import (
    KNOWN_QUEUES,
    REDIS_PROTOCOL_VERSION,
    LocalExecutor,
    QueueExecutor,
    QueueJob,
    RunnerOutcome,
    RunnerRequest,
    WorkflowService,
    WorkflowStatus,
    WorkflowStore,
    default_runner_registry,
    project_source_digest,
)
from lipid_screening_agent.runtime import RunContext


class RedisWorkflowWorker:
    def __init__(
        self,
        *,
        redis_url: str,
        state_db: str | Path,
        runs_root: str | Path,
        project_root: str | Path,
        resource_root: str | Path,
        queues: tuple[str, ...] = ("cpu",),
        worker_id: str | None = None,
        namespace: str = "lipid-agent-v3",
        execution_profile: str = "production",
        lease_seconds: float = 300.0,
        heartbeat_ttl_seconds: int = 300,
    ) -> None:
        unknown = set(queues) - KNOWN_QUEUES
        if unknown:
            raise ValueError("unknown queues: " + ", ".join(sorted(unknown)))
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.namespace = namespace
        self.queues = queues
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.execution_profile = execution_profile
        self.lease_seconds = max(1.0, lease_seconds)
        self.heartbeat_ttl_seconds = max(3, heartbeat_ttl_seconds)
        self.project_root = Path(project_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.resource_root = Path(resource_root).resolve()
        self.source_digest = project_source_digest(self.project_root)
        expected_source_digest = os.environ.get("LIPID_AGENT_SOURCE_DIGEST")
        if expected_source_digest in {None, "", "unknown"}:
            expected_source_digest = None
        if expected_source_digest and expected_source_digest != self.source_digest:
            raise RuntimeError(
                "container source digest does not match the baked/current project source: "
                f"expected={expected_source_digest} observed={self.source_digest}"
            )
        self.image_source_digest = expected_source_digest or self.source_digest
        self._active_netinfer_lock: str | None = None
        self.service = WorkflowService(
            store=WorkflowStore(state_db),
            registry=default_runner_registry(),
            executor=_WorkerQueueBoundary(self.client, namespace),
            project_root=self.project_root,
            resource_dir=self.resource_root,
            code_version=self.source_digest,
            recover_on_startup=False,
        )
        self.executor = LocalExecutor(heartbeat_interval_seconds=0.1)
        self.stopping = False

    def _worker_heartbeat(self) -> None:
        payload = {
            "worker_id": self.worker_id,
            "queues": list(self.queues),
            "protocol_version": REDIS_PROTOCOL_VERSION,
            "source_digest": self.source_digest,
            "image_source_digest": self.image_source_digest,
            "execution_profile": self.execution_profile,
            "updated_at": time.time(),
        }
        self.client.setex(
            f"{self.namespace}:worker:{self.worker_id}",
            self.heartbeat_ttl_seconds,
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
        self.client.sadd(f"{self.namespace}:workers", self.worker_id)

    def _cancelled(self, run_id: str, node_id: str, task_id: str) -> bool:
        keys = (
            f"{self.namespace}:cancel:{run_id}:*:*",
            f"{self.namespace}:cancel:{run_id}:{node_id}:*",
            f"{self.namespace}:cancel:{run_id}:{node_id}:{task_id}",
        )
        return any(self.client.exists(key) for key in keys)

    def _ack(self, job_id: str, queue: str, *, result: str) -> None:
        pipeline = self.client.pipeline(transaction=True)
        pipeline.lrem(f"{self.namespace}:processing:{queue}", 0, job_id)
        pipeline.zrem(f"{self.namespace}:leases", job_id)
        pipeline.hdel(f"{self.namespace}:jobs", job_id)
        pipeline.hset(f"{self.namespace}:job_results", job_id, result)
        pipeline.execute()
        if queue == "netinfer":
            self._release_netinfer_lock(job_id)

    def _acquire_netinfer_lock(self, job_id: str) -> bool:
        token = f"{self.worker_id}:{job_id}"
        ttl_seconds = max(60, int(self.lease_seconds * 4))
        acquired = self.client.set(
            f"{self.namespace}:lock:netinfer-global",
            token,
            nx=True,
            ex=ttl_seconds,
        )
        if acquired:
            self._active_netinfer_lock = token
        return bool(acquired)

    def _refresh_netinfer_lock(self) -> None:
        if self._active_netinfer_lock is None:
            return
        ttl_milliseconds = max(60_000, int(self.lease_seconds * 4_000))
        self.client.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end",
            1,
            f"{self.namespace}:lock:netinfer-global",
            self._active_netinfer_lock,
            ttl_milliseconds,
        )

    def _release_netinfer_lock(self, job_id: str) -> None:
        token = f"{self.worker_id}:{job_id}"
        self.client.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end",
            1,
            f"{self.namespace}:lock:netinfer-global",
            token,
        )
        if self._active_netinfer_lock == token:
            self._active_netinfer_lock = None

    def _reject(self, job_id: str, queue: str, payload: dict[str, Any], reason: str) -> None:
        try:
            self.service.record_queue_rejection(
                str(payload.get("run_id", "")),
                str(payload.get("node_id", "")),
                str(payload.get("task_id", "main")),
                reason=reason,
                job_id=job_id,
            )
        except Exception:
            pass
        self.client.hset(
            f"{self.namespace}:dead_letters",
            job_id,
            json.dumps({"reason": reason, "payload": payload}, separators=(",", ":")),
        )
        self._ack(job_id, queue, result=f"rejected:{reason}")

    def recover_expired(self) -> int:
        recovered = 0
        for job_id in self.client.zrangebyscore(
            f"{self.namespace}:leases", min="-inf", max=time.time()
        ):
            raw = self.client.hget(f"{self.namespace}:jobs", job_id)
            if raw is None:
                self.client.zrem(f"{self.namespace}:leases", job_id)
                continue
            payload = json.loads(raw)
            queue = str(payload.get("resource_class", "cpu"))
            try:
                self.service.interrupt_job(
                    str(payload["run_id"]),
                    str(payload["node_id"]),
                    str(payload["task_id"]),
                    attempt=int(payload["attempt"]),
                )
                result = "interrupted"
            except Exception as exc:
                result = f"expired_ignored:{type(exc).__name__}"
            self._ack(job_id, queue, result=result)
            recovered += 1
        return recovered

    def consume_one(self, *, timeout: int = 1) -> bool:
        self._worker_heartbeat()
        self.recover_expired()
        job_id = None
        queue = ""
        deadline = time.monotonic() + max(0, timeout)
        while not job_id:
            for candidate in self.queues:
                job_id = self.client.rpoplpush(
                    f"{self.namespace}:queue:{candidate}",
                    f"{self.namespace}:processing:{candidate}",
                )
                if job_id:
                    queue = candidate
                    break
            if job_id or time.monotonic() >= deadline:
                break
            self._worker_heartbeat()
            time.sleep(0.05)
        if not job_id:
            return False
        if queue == "netinfer" and not self._acquire_netinfer_lock(job_id):
            with self.client.pipeline(transaction=True) as pipeline:
                pipeline.lrem(f"{self.namespace}:processing:{queue}", 0, job_id)
                pipeline.rpush(f"{self.namespace}:queue:{queue}", job_id)
                pipeline.execute()
            time.sleep(0.1)
            return False
        raw = self.client.hget(f"{self.namespace}:jobs", job_id)
        if raw is None:
            self._ack(job_id, queue, result="missing_payload")
            return True
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._reject(job_id, queue, {}, "invalid_json")
            return True
        if payload.get("protocol_version") != REDIS_PROTOCOL_VERSION:
            self._reject(job_id, queue, payload, "protocol_version")
            return True
        if payload.get("resource_class") not in KNOWN_QUEUES or payload.get("resource_class") != queue:
            self._reject(job_id, queue, payload, "unknown_or_mismatched_queue")
            return True
        self.client.zadd(
            f"{self.namespace}:leases", {job_id: time.time() + self.lease_seconds}
        )
        try:
            runner_payload = self.service.claim_job(
                str(payload["run_id"]),
                str(payload["node_id"]),
                str(payload["task_id"]),
                attempt=int(payload["attempt"]),
                worker_id=self.worker_id,
                queue=queue,
            )
        except Exception as exc:
            self._reject(job_id, queue, payload, str(exc))
            return True

        run_id = str(payload["run_id"])
        node_id = str(payload["node_id"])
        task_id = str(payload["task_id"])
        attempt = int(payload["attempt"])

        def heartbeat(progress: float | None = None) -> None:
            self._worker_heartbeat()
            if queue == "netinfer":
                self._refresh_netinfer_lock()
            self.client.zadd(
                f"{self.namespace}:leases", {job_id: time.time() + self.lease_seconds}
            )
            self.service.heartbeat(
                run_id, node_id, task_id, progress=progress, attempt=attempt
            )

        context = RunContext.open_existing(
            run_dir=runner_payload["run_dir"],
            run_id=run_id,
            project_root=self.project_root,
            resource_dir=self.resource_root,
            create_missing_directories=True,
        )
        node = self.service.store.get_node(run_id, node_id, task_id)
        request = RunnerRequest(
            context=context,
            node=node,
            config_path=Path(runner_payload["config_path"]),
            input_artifacts=tuple(runner_payload.get("input_artifacts") or ()),
            code_version=self.source_digest,
            resource_hashes=runner_payload.get("resource_hashes") or {},
            is_cancelled=lambda: self._cancelled(run_id, node_id, task_id),
            heartbeat=heartbeat,
        )
        try:
            outcome = self.executor.execute(
                self.service.registry.get(node_id), request, heartbeat=heartbeat
            )
            self.service.complete_job(
                run_id, node_id, task_id, outcome, attempt=attempt
            )
            self._ack(job_id, queue, result=outcome.status.value)
        except Exception as exc:
            failure = RunnerOutcome(
                status=WorkflowStatus.FAILED,
                error={
                    "category": "execution",
                    "code": "worker_exception",
                    "message": str(exc)[:1000] or type(exc).__name__,
                    "exception_type": type(exc).__name__,
                    "retryable": True,
                    "details": {},
                },
            )
            try:
                self.service.complete_job(
                    run_id, node_id, task_id, failure, attempt=attempt
                )
                self._ack(job_id, queue, result="failed")
            except Exception:
                # Leave processing + lease intact for another worker's expiry recovery.
                pass
        return True

    def run_forever(self) -> None:
        while not self.stopping:
            self.consume_one(timeout=1)

    def healthcheck(self) -> bool:
        raw = self.client.get(f"{self.namespace}:worker:{self.worker_id}")
        return raw is not None and bool(self.client.ping())


class _WorkerQueueBoundary(QueueExecutor):
    """Queue interface used only for downstream scheduling from a worker completion."""

    def __init__(self, client: redis.Redis, namespace: str) -> None:
        self.client = client
        self.namespace = namespace

    def enqueue(self, job: QueueJob) -> str:
        if job.resource_class not in KNOWN_QUEUES:
            raise ValueError(f"unknown workflow queue: {job.resource_class}")
        job_id = f"job-{uuid.uuid4().hex}"
        payload = {
            "protocol_version": REDIS_PROTOCOL_VERSION,
            "job_id": job_id,
            "run_id": job.run_id,
            "node_id": job.node_id,
            "task_id": job.task_id,
            "attempt": job.attempt,
            "resource_class": job.resource_class,
            "payload": job.payload,
        }
        with self.client.pipeline(transaction=True) as pipeline:
            pipeline.hset(
                f"{self.namespace}:jobs",
                job_id,
                json.dumps(payload, separators=(",", ":")),
            )
            pipeline.rpush(f"{self.namespace}:queue:{job.resource_class}", job_id)
            pipeline.execute()
        return job_id

    def cancel(
        self, run_id: str, node_id: str | None = None, task_id: str | None = None
    ) -> None:
        scope = f"{run_id}:{node_id or '*'}:{task_id or '*'}"
        self.client.set(f"{self.namespace}:cancel:{scope}", "1", ex=7 * 24 * 3600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    project_root = Path(os.environ.get("LIPID_AGENT_PROJECT_ROOT", "/opt/lipid-agent"))
    worker = RedisWorkflowWorker(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        state_db=os.environ.get("LIPID_AGENT_DB", "/state/state.sqlite3"),
        runs_root=os.environ.get("LIPID_AGENT_RUNS_DIR", "/runs"),
        project_root=project_root,
        resource_root=os.environ.get("LIPID_AGENT_RESOURCE_ROOT", "/resources"),
        queues=tuple(
            item.strip()
            for item in os.environ.get("LIPID_AGENT_WORKER_QUEUES", "cpu").split(",")
            if item.strip()
        ),
        worker_id=os.environ.get("LIPID_AGENT_WORKER_ID") or None,
        namespace=os.environ.get("LIPID_AGENT_REDIS_NAMESPACE", "lipid-agent-v3"),
        execution_profile=os.environ.get("LIPID_AGENT_EXECUTION_PROFILE", "production"),
        lease_seconds=float(os.environ.get("LIPID_AGENT_LEASE_SECONDS", "300")),
        heartbeat_ttl_seconds=int(os.environ.get("LIPID_AGENT_WORKER_HEARTBEAT_TTL", "300")),
    )
    if args.healthcheck:
        return 0 if worker.healthcheck() else 1
    signal.signal(signal.SIGTERM, lambda *_args: setattr(worker, "stopping", True))
    signal.signal(signal.SIGINT, lambda *_args: setattr(worker, "stopping", True))
    if args.once:
        worker.consume_one(timeout=1)
        return 0
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
