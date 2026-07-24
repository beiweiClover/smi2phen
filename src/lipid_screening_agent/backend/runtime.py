"""Composition root for the minimal V3 backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lipid_screening_agent.config import load_workflow_config, parse_workflow_config
from lipid_screening_agent.orchestrator import (
    QueueExecutor,
    WorkflowService,
    WorkflowStore,
    default_runner_registry,
    project_source_digest,
)

from .queue import RedisQueueExecutor
from .service import ScreeningService
from .store import V3Store
from .tool_loop import ChatModel, OpenAICompatibleModel, ToolCallingAgent, ToolDispatcher
from .uploads import UploadService


@dataclass(frozen=True, slots=True)
class Runtime:
    service: ScreeningService
    uploads: UploadService
    agent: ToolCallingAgent | None
    store: V3Store
    model_configured: bool
    model_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-flash"
    model_timeout: float = 90.0
    model_max_retries: int = 2

    def agent_for_api_key(self, api_key: str | None) -> ToolCallingAgent | None:
        """Build a request-scoped agent without persisting the supplied secret."""

        key = (api_key or "").strip()
        if not key:
            return self.agent
        model = OpenAICompatibleModel(
            api_key=key,
            base_url=self.model_base_url,
            model=self.model_name,
            timeout=self.model_timeout,
            max_retries=self.model_max_retries,
        )
        return ToolCallingAgent(
            model=model,
            store=self.store,
            dispatcher=ToolDispatcher(self.service),
        )


def build_runtime(
    *,
    environ: Mapping[str, str] | None = None,
    executor: QueueExecutor | None = None,
    model: ChatModel | None = None,
) -> Runtime:
    values = os.environ if environ is None else environ
    project_root = Path(
        values.get("LIPID_AGENT_PROJECT_ROOT", Path(__file__).resolve().parents[3])
    ).resolve()
    state_dir = Path(values.get("LIPID_AGENT_STATE_DIR", project_root / ".data")).resolve()
    runs_root = Path(values.get("LIPID_AGENT_RUNS_DIR", state_dir / "runs")).resolve()
    resource_root = Path(
        values.get("LIPID_AGENT_RESOURCE_ROOT", state_dir / "resources")
    ).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    resource_root.mkdir(parents=True, exist_ok=True)
    database = Path(values.get("LIPID_AGENT_DB", state_dir / "state.sqlite3")).resolve()

    store = V3Store(database)
    workflow_store = WorkflowStore(database)
    queue = executor or RedisQueueExecutor(
        values.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
        namespace=values.get("LIPID_AGENT_REDIS_NAMESPACE", "lipid-agent-v3"),
    )
    workflow = WorkflowService(
        store=workflow_store,
        registry=default_runner_registry(),
        executor=queue,
        project_root=project_root,
        resource_dir=resource_root,
        code_version=project_source_digest(project_root),
        recover_on_startup=False,
    )
    workflow_config = load_workflow_config(project_root / "configs" / "workflow.yaml")
    if values.get("LIPID_AGENT_VALIDATION_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        workflow_config = _validation_workflow_config(workflow_config)
    service = ScreeningService(
        store=store,
        workflow=workflow,
        workflow_config=workflow_config,
        runs_root=runs_root,
        project_root=project_root,
        resource_root=resource_root,
    )
    uploads = UploadService(store=store, runs_root=runs_root)

    model_base_url = values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = values.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    model_timeout = float(values.get("DEEPSEEK_TIMEOUT_SECONDS", "90"))
    model_max_retries = int(values.get("DEEPSEEK_MAX_RETRIES", "2"))
    configured_model = model
    api_key = values.get("DEEPSEEK_API_KEY", "").strip()
    if configured_model is None and api_key:
        configured_model = OpenAICompatibleModel(
            api_key=api_key,
            base_url=model_base_url,
            model=model_name,
            timeout=model_timeout,
            max_retries=model_max_retries,
        )
    agent = (
        None
        if configured_model is None
        else ToolCallingAgent(
            model=configured_model,
            store=store,
            dispatcher=ToolDispatcher(service),
        )
    )
    return Runtime(
        service=service,
        uploads=uploads,
        agent=agent,
        store=store,
        model_configured=configured_model is not None,
        model_base_url=model_base_url,
        model_name=model_name,
        model_timeout=model_timeout,
        model_max_retries=model_max_retries,
    )


def _validation_workflow_config(config):
    """Reduce stochastic work while preserving the real scientific node implementations."""

    value = config.to_dict()
    value["workflow"]["id"] = "lipid_screening_v3_validation"
    value["workflow"]["version"] = "3.0-validation"
    value["proximity"]["randomizations"] = 20
    value["kg"]["construction"]["pubchem_fingerprint_workers"] = 1
    value["kg"]["pretrain"].update(
        {
            "epochs": 1,
            "batch_size": 8192,
            "train_print_every_batches": 10,
            "early_stopping_patience": 1,
            "save_embeddings": False,
        }
    )
    value["kg"]["finetune"].update(
        {
            "seeds": [5],
            "epochs": 2,
            "train_print_every_epochs": 1,
            "validate_every_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    value["ranking"]["kg"]["top_n"] = 20
    return parse_workflow_config(value)


__all__ = ["Runtime", "build_runtime"]
