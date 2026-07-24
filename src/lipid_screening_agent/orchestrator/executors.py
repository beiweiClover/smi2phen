"""Local subprocess/callable execution and a stable external queue interface."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RunnerOutcome, WorkflowStatus
from .registry import RegisteredRunner, RunnerRequest


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def _error(code: str, message: str, *, retryable: bool = True) -> RunnerOutcome:
    return RunnerOutcome(
        status=WorkflowStatus.FAILED,
        error={
            "category": "execution",
            "code": code,
            "message": message,
            "exception_type": "WorkflowExecutorError",
            "retryable": retryable,
            "details": {},
        },
    )


class LocalExecutor:
    """Development executor supporting registered callables and argv-only subprocesses."""

    def __init__(self, *, heartbeat_interval_seconds: float = 1.0) -> None:
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._tokens: dict[tuple[str, str, str], CancellationToken] = {}
        self._lock = threading.Lock()
        self._resource_locks: dict[str, threading.Semaphore] = {
            # Pretrain and every seed share this class, enforcing one training task per GPU in MVP.
            "gpu_training": threading.Semaphore(1)
        }

    def execute(
        self,
        registered: RegisteredRunner,
        request: RunnerRequest,
        *,
        heartbeat: Callable[[float | None], None],
    ) -> RunnerOutcome:
        key = (request.context.run_id, request.node.node_id, request.node.task_id)
        token = CancellationToken()
        with self._lock:
            self._tokens[key] = token
        bound_request = RunnerRequest(
            context=request.context,
            node=request.node,
            config_path=request.config_path,
            input_artifacts=request.input_artifacts,
            code_version=request.code_version,
            resource_hashes=request.resource_hashes,
            is_cancelled=lambda: token.is_cancelled() or request.is_cancelled(),
            heartbeat=heartbeat,
        )
        semaphore = self._resource_locks.get(request.node.resource_class)
        try:
            if semaphore is not None:
                semaphore.acquire()
            if token.is_cancelled():
                return RunnerOutcome(status=WorkflowStatus.CANCELLED)
            if registered.callable is not None:
                return self._run_callable(
                    registered.callable,
                    bound_request,
                    timeout=registered.timeout_seconds,
                    token=token,
                    external_cancel=request.is_cancelled,
                    heartbeat=heartbeat,
                )
            invocation = registered.command_factory(bound_request)  # type: ignore[misc]
            timeout = invocation.timeout_seconds or registered.timeout_seconds
            return self._run_subprocess(
                invocation.argv,
                cwd=invocation.cwd,
                env=invocation.env,
                timeout=timeout,
                token=token,
                external_cancel=request.is_cancelled,
                heartbeat=heartbeat,
            )
        except Exception as exc:
            return _error("executor_exception", str(exc))
        finally:
            if semaphore is not None:
                semaphore.release()
            with self._lock:
                self._tokens.pop(key, None)

    def cancel(self, run_id: str, node_id: str | None = None, task_id: str | None = None) -> None:
        with self._lock:
            for key, token in self._tokens.items():
                if key[0] != run_id:
                    continue
                if node_id is not None and key[1] != node_id:
                    continue
                if task_id is not None and key[2] != task_id:
                    continue
                token.cancel()

    def _run_callable(
        self,
        function: Callable[[RunnerRequest], RunnerOutcome],
        request: RunnerRequest,
        *,
        timeout: float | None,
        token: CancellationToken,
        external_cancel: Callable[[], bool],
        heartbeat: Callable[[float | None], None],
    ) -> RunnerOutcome:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workflow-runner")
        future = pool.submit(function, request)
        started = time.monotonic()
        try:
            while True:
                if token.is_cancelled() or external_cancel():
                    future.cancel()
                    return RunnerOutcome(status=WorkflowStatus.CANCELLED)
                remaining = None if timeout is None else timeout - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    token.cancel()
                    future.cancel()
                    return _error("timeout", f"runner exceeded timeout of {timeout} seconds")
                wait_for = self.heartbeat_interval_seconds
                if remaining is not None:
                    wait_for = min(wait_for, max(remaining, 0.001))
                try:
                    outcome = future.result(timeout=wait_for)
                    if not isinstance(outcome, RunnerOutcome):
                        return _error(
                            "invalid_runner_result",
                            "registered callable did not return RunnerOutcome",
                            retryable=False,
                        )
                    return outcome
                except FutureTimeout:
                    self._best_effort_heartbeat(heartbeat)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_subprocess(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        env: dict[str, str] | Any,
        timeout: float | None,
        token: CancellationToken,
        external_cancel: Callable[[], bool],
        heartbeat: Callable[[float | None], None],
    ) -> RunnerOutcome:
        process_env = None if env is None else {**os.environ, **dict(env)}
        process = subprocess.Popen(
            list(argv),
            cwd=None if cwd is None else str(cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        # The scientific CLIs can emit substantially more than one OS pipe
        # buffer (RDKit warnings are a common example).  Waiting for process
        # exit before calling communicate() deadlocks once either pipe fills.
        # Drain both streams continuously while retaining only bounded tails;
        # the final NodeResult JSON is emitted at the end of stdout.
        stdout_chunks: deque[str] = deque()
        stderr_chunks: deque[str] = deque()

        def drain_stream(stream: Any, chunks: deque[str], limit: int) -> None:
            retained = 0
            try:
                while True:
                    chunk = stream.read(65_536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    retained += len(chunk)
                    while retained > limit and chunks:
                        excess = retained - limit
                        first = chunks[0]
                        if len(first) <= excess:
                            retained -= len(chunks.popleft())
                        else:
                            chunks[0] = first[excess:]
                            retained -= excess
            finally:
                stream.close()

        assert process.stdout is not None
        assert process.stderr is not None
        drainers = (
            threading.Thread(
                target=drain_stream,
                args=(process.stdout, stdout_chunks, 4 * 1024 * 1024),
                name="runner-stdout-drain",
                daemon=True,
            ),
            threading.Thread(
                target=drain_stream,
                args=(process.stderr, stderr_chunks, 64 * 1024),
                name="runner-stderr-drain",
                daemon=True,
            ),
        )
        for drainer in drainers:
            drainer.start()

        def collected_output() -> tuple[str, str]:
            for drainer in drainers:
                drainer.join(timeout=5)
            return "".join(stdout_chunks), "".join(stderr_chunks)

        started = time.monotonic()
        while process.poll() is None:
            if token.is_cancelled() or external_cancel():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                collected_output()
                return RunnerOutcome(status=WorkflowStatus.CANCELLED)
            if timeout is not None and time.monotonic() - started > timeout:
                process.kill()
                process.wait()
                collected_output()
                return _error("timeout", f"runner exceeded timeout of {timeout} seconds")
            self._best_effort_heartbeat(heartbeat)
            time.sleep(min(self.heartbeat_interval_seconds, 0.25))
        stdout, stderr = collected_output()
        if process.returncode != 0:
            # Scientific CLIs deliberately exit non-zero for a failed
            # NodeResult while still emitting the full structured result.
            # Preserve that contract instead of replacing it with an import
            # warning or an opaque subprocess error.
            try:
                payload = json.loads(stdout.strip())
                status = WorkflowStatus(payload["status"])
                artifacts = tuple(
                    {"artifact_id": item} for item in payload.get("outputs", ())
                )
                return RunnerOutcome(
                    status=status,
                    artifacts=artifacts,
                    metrics=payload.get("metrics", {}),
                    error=payload.get("error"),
                    warnings=payload.get("warnings", ()),
                    fanout_items=payload.get("fanout_items", {}),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
            return _error(
                "subprocess_failed",
                (stderr or stdout or f"runner exited with code {process.returncode}")[-2000:],
            )
        try:
            # Formal runner CLIs print ``NodeResult.to_json()``, whose stable
            # default representation is indented across multiple lines.  Keep
            # compatibility with older one-line/status-tail emitters as a
            # fallback, but do not mistake the final closing brace for a JSON
            # document.
            try:
                payload = json.loads(stdout.strip())
            except json.JSONDecodeError:
                line = next(line for line in reversed(stdout.splitlines()) if line.strip())
                payload = json.loads(line)
            status = WorkflowStatus(payload["status"])
            artifacts = tuple({"artifact_id": item} for item in payload.get("outputs", ()))
            return RunnerOutcome(
                status=status,
                artifacts=artifacts,
                metrics=payload.get("metrics", {}),
                error=payload.get("error"),
                warnings=payload.get("warnings", ()),
                fanout_items=payload.get("fanout_items", {}),
            )
        except Exception as exc:
            return _error(
                "invalid_subprocess_result", f"cannot parse NodeResult JSON: {exc}", retryable=False
            )

    @staticmethod
    def _best_effort_heartbeat(
        heartbeat: Callable[[float | None], None],
    ) -> None:
        """Retry later rather than abandon a scientific process on transient contention."""

        try:
            heartbeat(None)
        except Exception:
            return


@dataclass(frozen=True, slots=True)
class QueueJob:
    run_id: str
    node_id: str
    task_id: str
    attempt: int
    resource_class: str
    payload: dict[str, Any]


class QueueExecutor(ABC):
    """Stable Redis/RQ adapter boundary; queue implementations own worker delivery."""

    @abstractmethod
    def enqueue(self, job: QueueJob) -> str:
        """Enqueue one immutable job and return the external queue job ID."""

    @abstractmethod
    def cancel(self, run_id: str, node_id: str | None = None, task_id: str | None = None) -> None:
        """Request cancellation without directly mutating workflow state."""


class InMemoryQueueExecutor(QueueExecutor):
    """Small deterministic queue adapter useful for contract tests and Stage 10 wiring."""

    def __init__(self) -> None:
        self.jobs: list[tuple[str, QueueJob]] = []
        self.cancelled: list[tuple[str, str | None, str | None]] = []

    def enqueue(self, job: QueueJob) -> str:
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs.append((job_id, job))
        return job_id

    def cancel(self, run_id: str, node_id: str | None = None, task_id: str | None = None) -> None:
        self.cancelled.append((run_id, node_id, task_id))


__all__ = [
    "CancellationToken",
    "InMemoryQueueExecutor",
    "LocalExecutor",
    "QueueExecutor",
    "QueueJob",
]
