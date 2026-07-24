"""FastAPI transport for V3; scientific work always leaves the API process through the queue."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .runtime import Runtime, build_runtime
from .uploads import EXPRESSION_PAIR_SPEC, UPLOADS, example_path, upload_catalog
from .web_ui import WEB_UI


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionRequest(StrictModel):
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)


class CreateRunRequest(StrictModel):
    disease_name: str = Field(min_length=1, max_length=200)
    disease_slug: str = Field(pattern=r"^[a-z0-9](?:[a-z0-9_]{0,62}[a-z0-9])?$")


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)


class StartRequest(StrictModel):
    confirmed: bool


def create_app(runtime: Runtime | None = None) -> FastAPI:
    active = runtime or build_runtime()
    application = FastAPI(title="smi2phen", version="0.1.0")

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def web_ui() -> str:
        return WEB_UI

    @application.get("/healthz")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_configured": active.model_configured,
            "model": active.model_name,
            "execution": "external_workers",
        }

    @application.get("/input-specs")
    def input_specs() -> dict[str, object]:
        return upload_catalog()

    @application.get("/examples/{example_id}", response_class=FileResponse)
    def download_example(example_id: str) -> FileResponse:
        path = _call(
            example_path,
            project_root=active.service.project_root,
            example_id=example_id,
        )
        return FileResponse(path, filename=path.name)

    @application.post("/sessions", status_code=201)
    def create_session(request: SessionRequest) -> dict[str, Any]:
        return _call(active.service.create_session, request.thread_id)

    @application.get("/sessions")
    def sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"sessions": _call(active.service.sessions, limit=limit)}

    @application.get("/sessions/{thread_id}")
    def session_history(thread_id: str) -> dict[str, Any]:
        return _call(active.service.session_history, thread_id)

    @application.get("/sessions/{thread_id}/export")
    def export_session(thread_id: str) -> Response:
        content, filename = _call(active.service.session_export, thread_id)
        return Response(
            content=content,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @application.post("/sessions/{thread_id}/runs", status_code=201)
    def create_run(thread_id: str, request: CreateRunRequest) -> dict[str, Any]:
        return _call(
            active.service.create_run,
            thread_id=thread_id,
            disease_name=request.disease_name,
            disease_slug=request.disease_slug,
        )

    @application.post("/sessions/{thread_id}/chat")
    def chat(
        thread_id: str,
        request: ChatRequest,
        model_api_key: str | None = Header(default=None, alias="X-Model-API-Key"),
    ) -> dict[str, Any]:
        agent = active.agent_for_api_key(model_api_key)
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="enter an API key in the web page or configure DEEPSEEK_API_KEY",
            )
        try:
            return _call(agent.chat, thread_id=thread_id, user_message=request.message)
        except HTTPException:
            raise
        except Exception as exc:
            detail = str(exc)
            if model_api_key:
                detail = detail.replace(model_api_key, "[redacted]")
            detail = detail.strip()[:400] or "no error detail returned"
            raise HTTPException(
                status_code=502,
                detail=f"model request failed ({type(exc).__name__}): {detail}",
            ) from None

    @application.post("/sessions/{thread_id}/chat/stream")
    def chat_stream(
        thread_id: str,
        request: ChatRequest,
        model_api_key: str | None = Header(default=None, alias="X-Model-API-Key"),
    ) -> StreamingResponse:
        agent = active.agent_for_api_key(model_api_key)
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="enter an API key in the web page or configure DEEPSEEK_API_KEY",
            )

        def generate() -> Iterator[bytes]:
            try:
                for event in agent.chat_events(
                    thread_id=thread_id,
                    user_message=request.message,
                ):
                    yield _ndjson(event)
            except Exception as exc:
                yield _ndjson(
                    {
                        "type": "error",
                        "message": _model_error_detail(exc, model_api_key),
                        "retryable": True,
                    }
                )

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @application.post("/runs/{run_id}/files/{kind}", status_code=201)
    def upload(
        run_id: str,
        kind: str,
        upload: UploadFile = File(...),
        replace: bool = Query(default=False),
        pair_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="upload filename is required")
        saved = _call(
            active.uploads.save,
            run_id=run_id,
            kind=kind,
            original_name=upload.filename,
            stream=upload.file,
            replace=replace,
            pair_id=pair_id,
        )
        requirements = active.service.requirements(run_id)
        receipt = active.service.record_upload_receipt(
            run_id,
            label=str(UPLOADS[kind]["label"]),
            file_names=[str(saved["original_name"])],
            requirements=requirements,
        )
        return {"input": saved, "requirements": requirements, "receipt": receipt}

    @application.post("/runs/{run_id}/expression-pairs/{pair_id}", status_code=201)
    def upload_expression_pair(
        run_id: str,
        pair_id: str,
        tpm: UploadFile = File(...),
        metadata: UploadFile = File(...),
        replace: bool = Query(default=False),
    ) -> dict[str, Any]:
        if not tpm.filename or not metadata.filename:
            raise HTTPException(status_code=400, detail="both TPM and metadata filenames are required")
        saved = _call(
            active.uploads.save_expression_pair,
            run_id=run_id,
            pair_id=pair_id,
            tpm_original_name=tpm.filename,
            tpm_stream=tpm.file,
            metadata_original_name=metadata.filename,
            metadata_stream=metadata.file,
            replace=replace,
        )
        requirements = active.service.requirements(run_id)
        receipt = active.service.record_upload_receipt(
            run_id,
            label=str(EXPRESSION_PAIR_SPEC["label"]),
            file_names=[tpm.filename, metadata.filename],
            requirements=requirements,
        )
        return {
            "expression_pair": saved,
            "requirements": requirements,
            "receipt": receipt,
        }

    @application.get("/runs/{run_id}/plan")
    def plan(run_id: str) -> dict[str, Any]:
        return _call(active.service.preview_plan, run_id)

    @application.post("/runs/{run_id}/start")
    def start(run_id: str, request: StartRequest) -> dict[str, Any]:
        return _call(active.service.start, run_id, confirmed=request.confirmed)

    @application.get("/runs/{run_id}")
    def status(run_id: str) -> dict[str, Any]:
        return _call(active.service.snapshot, run_id)

    @application.get("/runs/{run_id}/results")
    def results(run_id: str) -> dict[str, Any]:
        return _call(active.service.results, run_id)

    @application.get("/runs/{run_id}/artifacts/{artifact_id}", response_class=FileResponse)
    def download_artifact(run_id: str, artifact_id: str) -> FileResponse:
        path, artifact = _call(active.service.artifact_path, run_id, artifact_id)
        filename = path.name
        artifact_type = artifact.get("artifact_type")
        if isinstance(artifact_type, str) and artifact_type:
            filename = f"{artifact_type}-{filename}"
        return FileResponse(path, filename=filename)

    @application.get("/runs/{run_id}/candidates/{compound_id}/structure.svg")
    def candidate_structure(run_id: str, compound_id: str) -> Response:
        svg = _call(active.service.candidate_structure_svg, run_id, compound_id)
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @application.post("/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict[str, Any]:
        return _call(active.service.cancel, run_id)

    return application


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _ndjson(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _model_error_detail(exc: Exception, model_api_key: str | None) -> str:
    detail = str(exc)
    if model_api_key:
        detail = detail.replace(model_api_key, "[redacted]")
    detail = detail.strip()[:400] or "no error detail returned"
    return f"model request failed ({type(exc).__name__}): {detail}"


app = create_app()


__all__ = ["app", "create_app"]
