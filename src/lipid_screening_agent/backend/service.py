"""Deterministic application service used by both REST endpoints and Agent tools."""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
import zipfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from lipid_screening_agent.config import (
    WorkflowConfig,
    normalize_disease_config,
    workflow_config_for_disease,
)
from lipid_screening_agent.orchestrator import InputAvailability, InputState, WorkflowService
from lipid_screening_agent.runtime import RunContext

from .store import V3Store
from .uploads import guidance_for_requirement

FINAL_ARTIFACT_LABELS = {
    "final_candidates": "完整候选排名表",
    "ranking_summary": "排名汇总 JSON",
    "run_report_json": "完整运行报告 JSON",
    "run_report_markdown": "完整运行报告 Markdown",
}


class ScreeningService:
    """One-run-per-session V3 facade with an explicit execution gate."""

    def __init__(
        self,
        *,
        store: V3Store,
        workflow: WorkflowService,
        workflow_config: WorkflowConfig,
        runs_root: str | Path,
        project_root: str | Path,
        resource_root: str | Path,
    ) -> None:
        self.store = store
        self.workflow = workflow
        self.workflow_config = workflow_config
        self.runs_root = Path(runs_root).resolve()
        self.project_root = Path(project_root).resolve()
        self.resource_root = Path(resource_root).resolve()

    def create_session(self, thread_id: str | None = None) -> dict[str, Any]:
        return self.store.create_session(thread_id or f"thread-{uuid.uuid4().hex}")

    def sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for stored in self.store.sessions(limit=limit):
            item = {
                key: stored.get(key)
                for key in (
                    "thread_id",
                    "run_id",
                    "created_at",
                    "updated_at",
                    "message_count",
                    "disease",
                    "plan_previewed",
                )
            }
            if not stored.get("run_id"):
                item["status"] = "new"
            elif not stored.get("workflow_created"):
                item["status"] = (
                    "previewed" if stored.get("plan_previewed") else "collecting"
                )
            else:
                try:
                    item["status"] = self.snapshot(str(stored["run_id"])).get(
                        "status", "unknown"
                    )
                except (KeyError, ValueError):
                    item["status"] = "unknown"
            sessions.append(item)
        return sessions

    def session_history(self, thread_id: str) -> dict[str, Any]:
        session = self.store.session(thread_id)
        conversation: list[dict[str, Any]] = []
        for record in self.store.message_records(thread_id):
            message = record["message"]
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            conversation.append(
                {
                    "role": role,
                    "content": content,
                    "created_at": record["created_at"],
                }
            )
        run_id = session.get("run_id")
        snapshot = self.snapshot(str(run_id)) if run_id else None
        return {
            "session": session,
            "messages": conversation,
            "run": snapshot,
            "export_url": f"/sessions/{quote(thread_id, safe='')}/export",
        }

    def session_export(self, thread_id: str) -> tuple[bytes, str]:
        history = self.session_history(thread_id)
        run_id = history["session"].get("run_id")
        results = self.results(str(run_id)) if run_id else None
        transcript = _conversation_markdown(history)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "session.json",
                json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True),
            )
            archive.writestr(
                "conversation.json",
                json.dumps(history["messages"], ensure_ascii=False, indent=2),
            )
            archive.writestr("conversation.md", transcript)
            if results is not None:
                inputs = [
                    {
                        key: value.get(key)
                        for key in (
                            "input_key",
                            "original_name",
                            "size_bytes",
                            "sha256",
                            "created_at",
                        )
                    }
                    for value in self.store.inputs(str(run_id)).values()
                ]
                archive.writestr(
                    "inputs.json",
                    json.dumps(inputs, ensure_ascii=False, indent=2, sort_keys=True),
                )
                archive.writestr(
                    "results.json",
                    json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
                )
                for artifact in results.get("artifacts", []):
                    artifact_id = artifact.get("artifact_id")
                    if not isinstance(artifact_id, str) or not artifact_id:
                        continue
                    try:
                        path, resolved = self.artifact_path(str(run_id), artifact_id)
                    except (KeyError, ValueError):
                        continue
                    artifact_type = str(resolved.get("artifact_type") or "result")
                    archive.write(path, f"artifacts/{artifact_type}-{path.name}")
        safe_thread = re.sub(r"[^A-Za-z0-9._-]+", "-", thread_id).strip("-") or "session"
        return buffer.getvalue(), f"smi2phen-{safe_thread}.zip"

    def record_upload_receipt(
        self,
        run_id: str,
        *,
        label: str,
        file_names: list[str],
        requirements: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.store.run(run_id)
        current = requirements or self.requirements(run_id)
        files = "、".join(Path(name).name for name in file_names)
        if current["ready"]:
            next_step = "必需输入已齐备，可以检查并预览执行计划。"
        else:
            guidance = current.get("next_required_guidance") or {}
            next_label = guidance.get("label") or current.get("next_required") or "下一项文件"
            next_step = f"下一步请上传：{next_label}。"
        content = f"已接收{label}：{files}。基础结构检查通过。{next_step}"
        message = {
            "role": "assistant",
            "content": content,
            "event": "upload_receipt",
            "run_id": run_id,
        }
        self.store.append_message(str(run["thread_id"]), message)
        return {
            "content": content,
            "file_names": [Path(name).name for name in file_names],
            "next_required": current.get("next_required"),
            "ready": current["ready"],
        }

    def create_run(
        self, *, thread_id: str, disease_name: str, disease_slug: str
    ) -> dict[str, Any]:
        if self.store.run_for_thread(thread_id) is not None:
            raise ValueError("this V3 session already has a run")
        disease = normalize_disease_config(
            {"name": disease_name, "slug": disease_slug, "species": "human"},
            defaults=self.workflow_config.disease,
        )
        run_id = f"run-{uuid.uuid4().hex}"
        RunContext.create(
            runs_root=self.runs_root,
            run_id=run_id,
            project_root=self.project_root,
            resource_dir=self.resource_root,
        )
        self.store.create_run(run_id, thread_id, asdict(disease))
        return self.snapshot(run_id)

    def run_for_thread(self, thread_id: str) -> dict[str, Any] | None:
        return self.store.run_for_thread(thread_id)

    def requirements(self, run_id: str) -> dict[str, Any]:
        run = self.store.run(run_id)
        inputs = self.store.inputs(run_id)
        expression_pairs = _expression_pairs(inputs)
        complete_expression_pairs = {
            pair_id
            for pair_id, roles in expression_pairs.items()
            if {"tpm", "metadata"}.issubset(roles)
        }
        incomplete_expression_pairs = {
            pair_id
            for pair_id, roles in expression_pairs.items()
            if not {"tpm", "metadata"}.issubset(roles)
        }
        expression_pair = (
            "available"
            if complete_expression_pairs and not incomplete_expression_pairs
            else "incomplete"
            if incomplete_expression_pairs
            else "skipped"
        )
        drug_targets = "drug_targets" in inputs
        target_mapping = "target_mapping" in inputs
        target_pair = (
            "provided"
            if drug_targets and target_mapping
            else "incomplete"
            if drug_targets or target_mapping
            else "generated"
        )
        checklist = {
            "compounds": "available" if "compound_library" in inputs else "missing",
            "disease_genes": "available" if "disease_genes" in inputs else "missing",
            "drug_targets": "available" if drug_targets else "optional",
            "target_mapping": "available" if target_mapping else "optional",
            "positive_drugs": "available" if "positive_drugs" in inputs else "optional",
            "disease_links": "available" if "disease_links" in inputs else "optional",
            "target_pair": target_pair,
            "expression_pair": expression_pair,
            "expression_pair_count": len(complete_expression_pairs),
            "incomplete_expression_pair_count": len(incomplete_expression_pairs),
        }
        next_required = next(
            (
                name
                for name in (
                    "compounds",
                    "disease_genes",
                )
                if checklist[name] == "missing"
            ),
            None,
        )
        if next_required is None and target_pair == "incomplete":
            next_required = "target_mapping" if drug_targets else "drug_targets"
        if next_required is None and expression_pair == "incomplete":
            next_required = "expression_pair"
        result = {
            "ready": next_required is None,
            "next_required": next_required,
            "inputs_locked": self.store.inputs_locked(run_id),
            "plan_previewed": run["plan_previewed"],
            "inputs": checklist,
            "uploaded_files": [
                {
                    "input_key": name,
                    "original_name": value["original_name"],
                    "size_bytes": value["size_bytes"],
                    "sha256": value["sha256"],
                    **_expression_upload_metadata(name),
                }
                for name, value in sorted(inputs.items())
            ],
            "expression_pairs": [
                {
                    "pair_id": pair_id,
                    "tpm": "tpm" in roles,
                    "metadata": "metadata" in roles,
                    "status": (
                        "available"
                        if {"tpm", "metadata"}.issubset(roles)
                        else "incomplete"
                    ),
                }
                for pair_id, roles in sorted(expression_pairs.items())
            ],
            "evidence_mode": (
                "kg_proximity_gps" if expression_pair == "available" else "kg_proximity"
            ),
            "target_source": "provided" if target_pair == "provided" else "python_netinfer",
        }
        result["next_required_guidance"] = guidance_for_requirement(next_required)
        return result

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.store.run(run_id)
        if run["workflow_created"]:
            return self.workflow.status(run_id, include_events=True)
        return {
            "run_id": run_id,
            "status": "collecting",
            "disease": run["disease"],
            "requirements": self.requirements(run_id),
            "plan_previewed": run["plan_previewed"],
            "plan": None,
            "nodes": [],
        }

    def preview_plan(self, run_id: str) -> dict[str, Any]:
        requirements = self.requirements(run_id)
        if not requirements["ready"]:
            raise ValueError(f"missing required input: {requirements['next_required']}")
        run = self.store.run(run_id)
        if run["workflow_created"]:
            return self.workflow.plan(run_id)
        plan = self.workflow.preview(
            run_id=run_id,
            input_state=self._input_state(run_id),
            config=workflow_config_for_disease(self.workflow_config, run["disease"]),
        )
        self.store.mark_plan_previewed(run_id)
        return plan

    def start(self, run_id: str, *, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise ValueError("explicit execution confirmation is required")
        requirements = self.requirements(run_id)
        if not requirements["ready"]:
            raise ValueError(f"missing required input: {requirements['next_required']}")
        run = self.store.run(run_id)
        if not run["workflow_created"]:
            context = RunContext.open_existing(
                run_dir=self.runs_root / run_id,
                run_id=run_id,
                project_root=self.project_root,
                resource_dir=self.resource_root,
                create_missing_directories=True,
            )
            self.workflow.create(
                context=context,
                input_state=self._input_state(run_id),
                config=workflow_config_for_disease(self.workflow_config, run["disease"]),
            )
            self.workflow.plan(run_id)
            self.store.mark_workflow_created(run_id)
        return self.workflow.start(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self.store.run(run_id)
        if not run["workflow_created"]:
            raise ValueError("workflow has not been created")
        return self.workflow.cancel(run_id)

    def results(self, run_id: str) -> dict[str, Any]:
        run = self.store.run(run_id)
        if not run["workflow_created"]:
            return self._with_download_metadata(
                run_id,
                {"run_id": run_id, "status": "collecting", "artifacts": []},
            )
        value = self.workflow.results(run_id)
        value.update(self._result_preview(run_id))
        return self._with_download_metadata(run_id, value)

    def artifact_path(self, run_id: str, artifact_id: str) -> tuple[Path, dict[str, Any]]:
        """Resolve one final artifact through the recorded result manifest."""

        results = self.results(run_id)
        matches = [
            artifact
            for artifact in results.get("artifacts", [])
            if artifact.get("artifact_id") == artifact_id
        ]
        if not matches:
            raise KeyError(f"unknown final artifact {artifact_id!r}")
        artifact = matches[0]
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("artifact has no downloadable relative path")
        run_root = (self.runs_root / run_id).resolve()
        path = (run_root / relative_path).resolve()
        if run_root not in path.parents or not path.is_file():
            raise ValueError("artifact path is unavailable or outside the run directory")
        return path, artifact

    def candidate_structure_svg(self, run_id: str, compound_id: str) -> str:
        candidates, _ = self._candidate_details(run_id, limit=100)
        candidate = next(
            (item for item in candidates if item.get("compound_id") == compound_id),
            None,
        )
        if candidate is None:
            raise KeyError(f"unknown displayed candidate {compound_id!r}")
        smiles = candidate.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("candidate has no SMILES")
        modules = _rdkit_modules()
        if modules is None:
            raise ValueError("2D structure rendering is unavailable in this environment")
        Chem, _, _, _, rdMolDraw2D, _ = modules
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("candidate SMILES could not be rendered")
        drawer = rdMolDraw2D.MolDraw2DSVG(320, 220)
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return str(drawer.GetDrawingText())

    def _result_preview(self, run_id: str) -> dict[str, Any]:
        candidates, candidate_count = self._candidate_details(run_id, limit=100)
        run_root = (self.runs_root / run_id).resolve()
        summary_path = run_root / "artifacts" / "final" / "ranking_summary.json"
        summary: dict[str, Any] | None = None
        try:
            decoded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(decoded, dict):
                summary = {
                    key: decoded.get(key)
                    for key in (
                        "status",
                        "evidence_mode",
                        "evidence_count",
                        "stage_counts",
                        "thresholds",
                        "method",
                    )
                }
        except (OSError, UnicodeError, json.JSONDecodeError):
            summary = None
        return {
            "candidate_preview": candidates,
            "candidate_preview_limit": 100,
            "candidate_count": candidate_count,
            "ranking_summary": summary,
            "scope": "computational_prioritization_only",
        }

    def _candidate_details(
        self, run_id: str, *, limit: int
    ) -> tuple[list[dict[str, Any]], int]:
        run_root = (self.runs_root / run_id).resolve()
        candidates_path = run_root / "artifacts" / "final" / "final_candidates.tsv"
        compounds_path = run_root / "inputs" / "prepared" / "compounds.normalized.csv"
        compounds = _read_compound_details(compounds_path)
        candidates: list[dict[str, Any]] = []
        total = 0
        try:
            with candidates_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    total += 1
                    if len(candidates) >= max(1, min(limit, 100)):
                        continue
                    item: dict[str, Any] = {
                        str(key): str(value or "") for key, value in row.items()
                    }
                    compound_id = item.get("compound_id", "")
                    normalized = compounds.get(compound_id, {})
                    smiles = str(normalized.get("SMILES") or "")
                    item["smiles"] = smiles
                    item["properties"] = _compound_properties(smiles, normalized)
                    item["structure_available"] = bool(
                        smiles and item["properties"].get("rdkit_valid")
                    )
                    item["structure_url"] = (
                        f"/runs/{quote(run_id, safe='')}/candidates/"
                        f"{quote(compound_id, safe='')}/structure.svg"
                    )
                    candidates.append(item)
        except (OSError, UnicodeError, csv.Error):
            return [], 0
        return candidates, total

    def _with_download_metadata(self, run_id: str, value: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(value)
        artifacts: list[dict[str, Any]] = []
        for item in value.get("artifacts", []):
            if not isinstance(item, dict):
                continue
            artifact = dict(item)
            artifact_id = artifact.get("artifact_id")
            relative_path = artifact.get("relative_path")
            if isinstance(relative_path, str) and relative_path:
                artifact["file_name"] = Path(relative_path).name
            if isinstance(artifact_id, str) and artifact_id:
                artifact["download_url"] = (
                    f"/runs/{quote(run_id, safe='')}/artifacts/"
                    f"{quote(artifact_id, safe='')}"
                )
            artifact["download_label"] = _artifact_download_label(artifact)
            artifacts.append(artifact)
        enriched["artifacts"] = artifacts
        return enriched

    def _input_state(self, run_id: str) -> InputState:
        inputs = self.store.inputs(run_id)
        requirements = self.requirements(run_id)
        expression_state = requirements["inputs"]["expression_pair"]
        hashes = {name: value["sha256"] for name, value in inputs.items()}
        sizes = {
            f"{name}_bytes": float(value["size_bytes"]) for name, value in inputs.items()
        }
        return InputState(
            compounds=InputAvailability.AVAILABLE
            if "compound_library" in inputs
            else InputAvailability.MISSING,
            disease_genes=InputAvailability.AVAILABLE
            if "disease_genes" in inputs
            else InputAvailability.MISSING,
            expression_pairs=(
                InputAvailability.AVAILABLE
                if expression_state == "available"
                else InputAvailability.MISSING
                if expression_state == "incomplete"
                else InputAvailability.SKIPPED
            ),
            drug_targets=InputAvailability.AVAILABLE
            if "drug_targets" in inputs
            else InputAvailability.MISSING,
            target_mapping=InputAvailability.AVAILABLE
            if "target_mapping" in inputs
            else InputAvailability.MISSING,
            input_artifact_hashes=hashes,
            input_scale=sizes,
        )


def _expression_upload_metadata(input_key: str) -> dict[str, str]:
    parsed = _expression_input_key(input_key)
    if parsed is None:
        return {}
    pair_id, role = parsed
    return {"pair_id": pair_id, "role": role}


def _expression_input_key(input_key: str) -> tuple[str, str] | None:
    if input_key == "expression_tpm":
        return ("1", "tpm")
    if input_key == "expression_metadata":
        return ("1", "metadata")
    if input_key.startswith("expression_tpm:"):
        return (input_key.split(":", 1)[1], "tpm")
    if input_key.startswith("expression_metadata:"):
        return (input_key.split(":", 1)[1], "metadata")
    return None


def _expression_pairs(inputs: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for input_key in inputs:
        parsed = _expression_input_key(input_key)
        if parsed is None:
            continue
        pair_id, role = parsed
        pairs.setdefault(pair_id, set()).add(role)
    return pairs


def _artifact_download_label(artifact: dict[str, Any]) -> str:
    artifact_type = artifact.get("artifact_type")
    file_name = artifact.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        relative_path = artifact.get("relative_path")
        file_name = Path(relative_path).name if isinstance(relative_path, str) else ""
    label = FINAL_ARTIFACT_LABELS.get(str(artifact_type or ""))
    if label and file_name:
        return f"{label}（{file_name}）"
    if label:
        return label
    if file_name:
        return f"结果文件（{file_name}）"
    return str(artifact_type or "结果文件")


def _conversation_markdown(history: dict[str, Any]) -> str:
    session = history["session"]
    lines = [
        "# 降脂药物筛选 Agent 会话记录",
        "",
        f"- Thread ID: {session['thread_id']}",
        f"- Run ID: {session.get('run_id') or '未创建'}",
        f"- Created: {session['created_at']}",
        f"- Updated: {session['updated_at']}",
        "",
        "## 对话",
        "",
    ]
    role_labels = {"user": "用户", "assistant": "Agent"}
    for message in history["messages"]:
        lines.extend(
            [
                f"### {role_labels.get(message['role'], message['role'])}",
                "",
                str(message["content"]),
                "",
            ]
        )
    return "\n".join(lines)


def _read_compound_details(path: Path) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                compound_id = str(row.get("ID") or "").strip()
                if compound_id and compound_id not in values:
                    values[compound_id] = {
                        str(key): str(value or "") for key, value in row.items()
                    }
    except (OSError, UnicodeError, csv.Error):
        return {}
    return values


@lru_cache(maxsize=1)
def _rdkit_modules() -> tuple[Any, ...] | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    return Chem, Crippen, Descriptors, Lipinski, rdMolDraw2D, rdMolDescriptors


def _compound_properties(smiles: str, normalized: dict[str, str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "formula": normalized.get("Formula") or None,
        "molecular_weight": _optional_float(normalized.get("MolWt")),
        "rdkit_valid": False,
    }
    modules = _rdkit_modules()
    if not smiles or modules is None:
        return properties
    Chem, Crippen, Descriptors, Lipinski, _, rdMolDescriptors = modules
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return properties
    properties.update(
        {
            "formula": rdMolDescriptors.CalcMolFormula(molecule),
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 2),
            "logp": round(float(Crippen.MolLogP(molecule)), 2),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 2),
            "h_bond_donors": int(Lipinski.NumHDonors(molecule)),
            "h_bond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            "rdkit_valid": True,
        }
    )
    return properties


def _optional_float(value: str | None) -> float | None:
    try:
        return round(float(value), 2) if value not in {None, ""} else None
    except ValueError:
        return None


__all__ = ["ScreeningService"]
