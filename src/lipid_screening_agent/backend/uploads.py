"""Deterministic streaming uploads into the filenames expected by the scientific DAG."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import BinaryIO

from .input_validation import validate_upload
from .store import V3Store

UPLOADS = {
    "compounds": {
        "input_key": "compound_library",
        "filename": "compounds.csv",
        "extensions": {"csv", "tsv"},
        "max_bytes": 100 * 1024 * 1024,
        "label": "化合物库",
        "required": True,
        "format": "CSV/TSV；legacy 文件通常为 smiles.csv，必须包含唯一 ID 列和 SMILES 列，可附加名称等列。",
        "source": "来自用户已有化合物库、TargetMol/商业库、实验室集合或自定义分子集合。",
        "collection_note": "整理为 ID 与 SMILES 两个必需列；商业数据请按已有授权自行导出。",
        "valid_example": "compounds-valid",
        "invalid_example": "compounds-invalid",
    },
    "disease_genes": {
        "input_key": "disease_genes",
        "filename": "disease_genes.tsv",
        "extensions": {"txt", "tsv", "csv"},
        "max_bytes": 20 * 1024 * 1024,
        "label": "疾病基因",
        "required": True,
        "format": "TXT/TSV/CSV；legacy 文件通常为 steatosis_gene.txt，每行一个 HGNC symbol，或 symbol/entrez_id 表格。",
        "source": "来自文献、差异表达分析、疾病数据库或专家整理的疾病相关基因集。",
        "collection_note": "优先统一为 HGNC symbol；也可同时提供 entrez_id，避免同义词映射歧义。",
        "valid_example": "disease-genes-valid",
        "invalid_example": "disease-genes-invalid",
    },
    "drug_targets": {
        "input_key": "drug_targets",
        "filename": "drug_targets.json",
        "extensions": {"json"},
        "max_bytes": 100 * 1024 * 1024,
        "label": "外部靶点 JSON",
        "required": False,
        "format": "可选 JSON；provided-target 路径使用，以化合物 ID 为键，每项包含 targets 数组。",
        "source": "来自已有药物—靶点数据库导出、实验结果或用户自己的预测结果。",
        "collection_note": "如果没有这一对文件，系统会使用内置 Python NetInfer 生成靶点。",
        "pair_with": "target_mapping",
        "valid_example": "drug-targets-valid",
        "invalid_example": "drug-targets-invalid",
        "catalog": False,
    },
    "target_mapping": {
        "input_key": "target_mapping",
        "filename": "target_mapping.tsv",
        "extensions": {"tsv"},
        "max_bytes": 20 * 1024 * 1024,
        "label": "外部靶点映射",
        "required": False,
        "format": "可选 TSV；provided-target 路径使用，必须包含 gene_symbol 与 entrez_id，需与外部靶点 JSON 成对上传。",
        "source": "来自 NCBI Gene、MyGene、biomaRt 或 org.Hs.eg.db 等标识映射结果。",
        "collection_note": "需与外部靶点 JSON 成对提供；没有时无需上传。",
        "pair_with": "drug_targets",
        "valid_example": "target-mapping-valid",
        "invalid_example": "target-mapping-invalid",
        "catalog": False,
    },
    "positive_drugs": {
        "input_key": "positive_drugs",
        "filename": "positive_drugs.tsv",
        "extensions": {"tsv"},
        "max_bytes": 20 * 1024 * 1024,
        "label": "KG 阳性药物先验",
        "required": False,
        "format": "可选 TSV；KG 先验输入，必须包含 input_type 与 value，input_type 可为 library_id、base_drug_name 或 base_drug_id。",
        "source": "来自文献、指南、数据库或实验确认的阳性药物先验。",
        "collection_note": "没有可靠先验时可以跳过，不要为凑齐文件而填写推测内容。",
        "valid_example": "positive-drugs-valid",
        "invalid_example": "positive-drugs-invalid",
    },
    "disease_links": {
        "input_key": "disease_links",
        "filename": "disease_links.tsv",
        "extensions": {"tsv"},
        "max_bytes": 20 * 1024 * 1024,
        "label": "KG 疾病链接先验",
        "required": False,
        "format": "可选 TSV；KG 先验输入，必须包含 input_type 与 value，input_type 可为 base_disease_id 或 base_disease_name，可附加 node_name。",
        "source": "来自当前 KG 基础资源中的疾病节点 ID 或名称。",
        "collection_note": "仅在已知基础图谱节点时提供；否则可以跳过。",
        "valid_example": "disease-links-valid",
        "invalid_example": "disease-links-invalid",
    },
    "expression_tpm": {
        "input_key": "expression_tpm",
        "filename": "TPM_matrix_1.tsv",
        "extensions": {"tsv"},
        "max_bytes": 250 * 1024 * 1024,
        "label": "表达 TPM",
        "required": False,
        "format": "可选 TSV；文件名按 TPM_matrix_<编号>.tsv 成对登记，首列必须为 GeneID，其余列为样本 ID。",
        "source": "来自 GEO、SRA、ArrayExpress、TCGA 或用户自己的表达数据。",
        "collection_note": "数据集选择、标准化和分组属于科学决策，当前版本由用户准备后上传。",
        "pair_with": "expression_metadata",
        "valid_example": "expression-tpm-valid",
        "invalid_example": "expression-tpm-invalid",
        "catalog": False,
    },
    "expression_metadata": {
        "input_key": "expression_metadata",
        "filename": "metadata_1.tsv",
        "extensions": {"tsv"},
        "max_bytes": 20 * 1024 * 1024,
        "label": "表达分组",
        "required": False,
        "format": "可选 TSV；文件名按 metadata_<编号>.tsv 成对登记，包含 sample_id、group，group 仅为 control/disease。",
        "source": "与 TPM 表对应的样本分组信息，通常来自原始研究的样本注释。",
        "collection_note": "sample_id 必须与 TPM 列名对应，group 只使用 control 或 disease。",
        "pair_with": "expression_tpm",
        "valid_example": "expression-metadata-valid",
        "invalid_example": "expression-metadata-invalid",
        "catalog": False,
    },
}

EXAMPLES = {
    "compounds-valid": "examples/provided_targets_validation/compounds.csv",
    "compounds-invalid": "examples/invalid_inputs/compounds_missing_smiles.csv",
    "disease-genes-valid": "examples/provided_targets_validation/disease_genes.tsv",
    "disease-genes-invalid": "examples/invalid_inputs/disease_genes_header_only.tsv",
    "drug-targets-valid": "examples/provided_targets_validation/drug_targets.json",
    "drug-targets-invalid": "examples/invalid_inputs/drug_targets_wrong_shape.json",
    "target-mapping-valid": "examples/provided_targets_validation/target_mapping.tsv",
    "target-mapping-invalid": "examples/invalid_inputs/target_mapping_missing_entrez.tsv",
    "positive-drugs-valid": "examples/kg_prior_validation/positive_drugs.tsv",
    "positive-drugs-invalid": "examples/invalid_inputs/positive_drugs_bad_input_type.tsv",
    "disease-links-valid": "examples/kg_prior_validation/disease_links.tsv",
    "disease-links-invalid": "examples/invalid_inputs/disease_links_missing_value.tsv",
    "expression-tpm-valid": "examples/expression_validation/TPM_matrix_1.tsv",
    "expression-tpm-invalid": "examples/invalid_inputs/TPM_matrix_wrong_first_column.tsv",
    "expression-metadata-valid": "examples/expression_validation/metadata_1.tsv",
    "expression-metadata-invalid": "examples/invalid_inputs/metadata_invalid_group.tsv",
}

EXPRESSION_PAIR_SPEC = {
    "label": "表达 TPM / metadata 对",
    "required": False,
    "format": (
        "可选多对 TSV；每一对包含一个 TPM_matrix_<编号>.tsv 和一个 metadata_<编号>.tsv。"
        "TPM 首列必须为 GeneID，metadata 必须包含 sample_id 与 group。"
    ),
    "source": "来自 GEO、SRA、ArrayExpress、TCGA 或用户自己的表达数据与样本注释。",
    "collection_note": (
        "当前版本不自动选择数据集或替用户决定标准化方案；Agent 会说明所需格式，"
        "用户准备好 TPM 与 metadata 后可在对话区或右侧上传。"
    ),
    "tpm_kind": "expression_tpm",
    "metadata_kind": "expression_metadata",
    "tpm_accept": ".tsv",
    "metadata_accept": ".tsv",
    "tpm_valid_example_url": "/examples/expression-tpm-valid",
    "tpm_invalid_example_url": "/examples/expression-tpm-invalid",
    "metadata_valid_example_url": "/examples/expression-metadata-valid",
    "metadata_invalid_example_url": "/examples/expression-metadata-invalid",
}

_EXPRESSION_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def upload_catalog() -> dict[str, object]:
    """Return browser-safe upload requirements without filesystem paths."""

    inputs = []
    for kind, rule in UPLOADS.items():
        if rule.get("catalog") is False:
            continue
        inputs.append(
            {
                "kind": kind,
                "input_key": rule["input_key"],
                "label": rule["label"],
                "required": rule["required"],
                "extensions": sorted(rule["extensions"]),
                "accept": ",".join(f".{value}" for value in sorted(rule["extensions"])),
                "max_bytes": rule["max_bytes"],
                "format": rule["format"],
                "source": rule.get("source"),
                "collection_note": rule.get("collection_note"),
                "pair_with": rule.get("pair_with"),
                "valid_example_url": f"/examples/{rule['valid_example']}",
                "invalid_example_url": f"/examples/{rule['invalid_example']}",
            }
        )
    return {
        "inputs": inputs,
        "expression_pair": dict(EXPRESSION_PAIR_SPEC),
        "note": (
            "上传时只做轻量结构检查；SMILES、基因映射和科学数据完整性由 DAG 节点验证。"
        ),
    }


def guidance_for_requirement(requirement: str | None) -> dict[str, object] | None:
    if not requirement:
        return None
    if requirement == "expression_pair":
        return {
            "kind": "expression_pair",
            "label": EXPRESSION_PAIR_SPEC["label"],
            "format": EXPRESSION_PAIR_SPEC["format"],
            "source": EXPRESSION_PAIR_SPEC["source"],
            "collection_note": EXPRESSION_PAIR_SPEC["collection_note"],
        }
    rule = UPLOADS.get(requirement)
    if rule is None:
        return None
    return {
        "kind": requirement,
        "label": rule["label"],
        "format": rule["format"],
        "source": rule.get("source"),
        "collection_note": rule.get("collection_note"),
    }


def example_path(*, project_root: str | Path, example_id: str) -> Path:
    try:
        relative = EXAMPLES[example_id]
    except KeyError as exc:
        raise KeyError(f"unknown example {example_id!r}") from exc
    root = Path(project_root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise KeyError(f"example is unavailable: {example_id!r}")
    return path


class UploadService:
    def __init__(self, *, store: V3Store, runs_root: str | Path) -> None:
        self.store = store
        self.runs_root = Path(runs_root).resolve()

    def save(
        self,
        *,
        run_id: str,
        kind: str,
        original_name: str,
        stream: BinaryIO,
        replace: bool = False,
        pair_id: str | None = None,
    ) -> dict[str, object]:
        try:
            rule = UPLOADS[kind]
        except KeyError as exc:
            raise ValueError(f"unsupported upload kind: {kind}") from exc
        if self.store.inputs_locked(run_id):
            raise ValueError("inputs are locked after workflow execution starts")
        suffix = _expression_suffix(pair_id) if kind in EXPRESSION_KINDS else None
        destination_name = _destination_filename(kind, rule, suffix=suffix)
        input_key = _input_key(kind, rule, suffix=suffix, pair_id_was_provided=pair_id is not None)
        destination, size, digest, validation = self._write_and_validate(
            run_id=run_id,
            kind=kind,
            rule=rule,
            original_name=original_name,
            stream=stream,
            destination_name=destination_name,
            replace=replace,
        )
        saved = self.store.put_input(
            run_id=run_id,
            input_key=input_key,
            original_name=Path(original_name).name,
            stored_path=str(destination),
            size_bytes=size,
            sha256=digest,
        )
        saved["validation"] = validation
        if suffix is not None:
            saved["pair_id"] = suffix
            saved["role"] = "tpm" if kind == "expression_tpm" else "metadata"
        return saved

    def save_expression_pair(
        self,
        *,
        run_id: str,
        pair_id: str,
        tpm_original_name: str,
        tpm_stream: BinaryIO,
        metadata_original_name: str,
        metadata_stream: BinaryIO,
        replace: bool = False,
    ) -> dict[str, object]:
        if self.store.inputs_locked(run_id):
            raise ValueError("inputs are locked after workflow execution starts")
        suffix = _expression_suffix(pair_id)
        tpm_rule = UPLOADS["expression_tpm"]
        metadata_rule = UPLOADS["expression_metadata"]
        tpm_destination, tpm_size, tpm_digest, tpm_validation = self._write_and_validate(
            run_id=run_id,
            kind="expression_tpm",
            rule=tpm_rule,
            original_name=tpm_original_name,
            stream=tpm_stream,
            destination_name=_destination_filename("expression_tpm", tpm_rule, suffix=suffix),
            replace=replace,
        )
        try:
            metadata_destination, metadata_size, metadata_digest, metadata_validation = (
                self._write_and_validate(
                    run_id=run_id,
                    kind="expression_metadata",
                    rule=metadata_rule,
                    original_name=metadata_original_name,
                    stream=metadata_stream,
                    destination_name=_destination_filename(
                        "expression_metadata", metadata_rule, suffix=suffix
                    ),
                    replace=replace,
                )
            )
        except Exception:
            tpm_destination.unlink(missing_ok=True)
            raise

        tpm_saved = self.store.put_input(
            run_id=run_id,
            input_key=_input_key(
                "expression_tpm", tpm_rule, suffix=suffix, pair_id_was_provided=True
            ),
            original_name=Path(tpm_original_name).name,
            stored_path=str(tpm_destination),
            size_bytes=tpm_size,
            sha256=tpm_digest,
        )
        metadata_saved = self.store.put_input(
            run_id=run_id,
            input_key=_input_key(
                "expression_metadata",
                metadata_rule,
                suffix=suffix,
                pair_id_was_provided=True,
            ),
            original_name=Path(metadata_original_name).name,
            stored_path=str(metadata_destination),
            size_bytes=metadata_size,
            sha256=metadata_digest,
        )
        tpm_saved.update({"validation": tpm_validation, "pair_id": suffix, "role": "tpm"})
        metadata_saved.update(
            {"validation": metadata_validation, "pair_id": suffix, "role": "metadata"}
        )
        return {
            "pair_id": suffix,
            "inputs": {"tpm": tpm_saved, "metadata": metadata_saved},
        }

    def _write_and_validate(
        self,
        *,
        run_id: str,
        kind: str,
        rule: dict[str, object],
        original_name: str,
        stream: BinaryIO,
        destination_name: str,
        replace: bool,
    ) -> tuple[Path, int, str, dict[str, object]]:
        safe_name = Path(original_name).name
        if safe_name != original_name or not safe_name:
            raise ValueError("invalid upload filename")
        extension = safe_name.rsplit(".", 1)[-1].casefold() if "." in safe_name else ""
        if extension not in rule["extensions"]:
            raise ValueError(f"unsupported file extension for {kind}")

        input_dir = (self.runs_root / run_id / "inputs").resolve()
        if input_dir.parent != (self.runs_root / run_id).resolve():
            raise ValueError("invalid run input directory")
        input_dir.mkdir(parents=True, exist_ok=True)
        destination = input_dir / destination_name
        if destination.exists() and not replace:
            raise ValueError("input already exists; set replace=true to replace it")

        temporary = input_dir / f".{destination.name}.{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as handle:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > int(rule["max_bytes"]):
                        raise ValueError(f"upload exceeds size limit for {kind}")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if size == 0:
                raise ValueError("upload is empty")
            try:
                validation = validate_upload(
                    kind=kind,
                    path=temporary,
                    original_name=safe_name,
                )
            except ValueError as exc:
                raise ValueError(f"basic validation failed for {kind}: {exc}") from exc
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination, size, digest.hexdigest(), validation


EXPRESSION_KINDS = frozenset({"expression_tpm", "expression_metadata"})


def _expression_suffix(pair_id: str | None) -> str:
    suffix = "1" if pair_id is None else pair_id.strip()
    if not _EXPRESSION_PAIR_ID.fullmatch(suffix):
        raise ValueError("expression pair_id must be 1-64 chars of letters, digits, '_' or '-'")
    return suffix


def _destination_filename(kind: str, rule: dict[str, object], *, suffix: str | None) -> str:
    if kind == "expression_tpm":
        return f"TPM_matrix_{suffix}.tsv"
    if kind == "expression_metadata":
        return f"metadata_{suffix}.tsv"
    return str(rule["filename"])


def _input_key(
    kind: str,
    rule: dict[str, object],
    *,
    suffix: str | None,
    pair_id_was_provided: bool,
) -> str:
    base = str(rule["input_key"])
    if kind not in EXPRESSION_KINDS:
        return base
    if not pair_id_was_provided and suffix == "1":
        return base
    return f"{base}:{suffix}"


__all__ = [
    "EXAMPLES",
    "EXPRESSION_PAIR_SPEC",
    "UPLOADS",
    "UploadService",
    "example_path",
    "guidance_for_requirement",
    "upload_catalog",
]
