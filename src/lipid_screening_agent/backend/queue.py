"""Minimal Redis implementation of the stable workflow queue boundary."""

from __future__ import annotations

import json
import uuid

from lipid_screening_agent.orchestrator import (
    KNOWN_QUEUES,
    REDIS_PROTOCOL_VERSION,
    QueueExecutor,
    QueueJob,
)


class RedisQueueExecutor(QueueExecutor):
    def __init__(self, url: str, *, namespace: str = "lipid-agent-v3") -> None:
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)
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

    def cancel(self, run_id: str, node_id: str | None = None, task_id: str | None = None) -> None:
        scope = f"{run_id}:{node_id or '*'}:{task_id or '*'}"
        self.client.set(f"{self.namespace}:cancel:{scope}", "1", ex=7 * 24 * 3600)


__all__ = ["RedisQueueExecutor"]
