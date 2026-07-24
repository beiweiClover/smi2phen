"""Prepare known DRUG mappings and independently retryable novel NetInfer batches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import NetInferConfig
from lipid_screening_agent.runtime import (
    InputError,
    ResourceError,
    RunContext,
    atomic_write_json,
    sha256_file,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import (
    add_execution_identity_arguments,
    execution_identity,
)
from ._common import (
    atomic_write_delimited,
    package_version,
    resolve_netinfer_output,
    resolve_resource_path,
    resolve_run_input_file,
)
from ._dependencies import PrepareDependencies, load_prepare_dependencies
from ._settings import validate_netinfer_settings

NODE_ID = "netinfer_prepare_inputs"
MAPPING_RELATIVE_PATH = "input_mapping.tsv"
BATCH_MANIFEST_RELATIVE_PATH = "batch_manifest.json"
TARGET_MAP_RELATIVE_PATH = "target_uniprot_to_symbol.tsv"

MAPPING_COLUMNS = (
    "ID",
    "SMILES",
    "match_key",
    "official_drug_id",
    "netinfer_input_type",
    "netinfer_input_id",
    "batch_id",
)


def standardize_to_match_key(
    smiles: Any,
    *,
    normalizer: Any,
    chooser: Any,
    dependencies: PrepareDependencies,
) -> str | None:
    """Apply the exact legacy RDKit cleanup/largest-fragment/canonical-key path."""

    if dependencies.pd.isna(smiles):
        return None
    value = str(smiles).strip()
    if not value:
        return None
    molecule = dependencies.chem.MolFromSmiles(value)
    if molecule is None:
        return None
    try:
        molecule = dependencies.standardize.Cleanup(molecule)
        molecule = chooser.choose(molecule)
        molecule = normalizer.normalize(molecule)
    except Exception:
        pass
    return str(dependencies.chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True))


def _load_compounds(path: Path, *, dependencies: PrepareDependencies) -> Any:
    try:
        compounds = dependencies.pd.read_csv(
            path,
            dtype={"ID": str, "SMILES": str},
            keep_default_na=False,
        )
    except Exception as exc:
        raise InputError(
            "compounds.normalized.csv could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    missing = sorted({"ID", "SMILES"} - set(compounds.columns))
    if missing:
        raise InputError(
            "compounds.normalized.csv is missing required columns",
            details={"missing_columns": missing},
        )
    compounds = compounds[["ID", "SMILES"]].copy()
    compounds["ID"] = compounds["ID"].astype(str).str.strip()
    compounds["SMILES"] = compounds["SMILES"].astype(str).str.strip()
    if compounds.empty or compounds["ID"].eq("").any() or compounds["SMILES"].eq("").any():
        raise InputError("normalized compounds must contain non-empty ID and SMILES values")
    duplicate_ids = compounds.loc[compounds["ID"].duplicated(keep=False), "ID"].unique().tolist()
    if duplicate_ids:
        raise InputError(
            "normalized compound IDs must be unique",
            details={"duplicate_ids": sorted(map(str, duplicate_ids))[:50]},
        )
    return compounds


def _load_official_tables(
    path: Path,
    *,
    normalizer: Any,
    chooser: Any,
    dependencies: PrepareDependencies,
    progress: Any,
) -> tuple[dict[str, str], list[tuple[str, str, str]], dict[str, int]]:
    pd = dependencies.pd
    try:
        drugs = pd.read_excel(
            path,
            sheet_name="Drug information",
            dtype=str,
            usecols=["Drug ID", "SMILES (standardized)"],
        )
        targets = pd.read_excel(
            path,
            sheet_name="Target information",
            dtype=str,
            usecols=lambda column: column in {"UniProt AC", "Gene symbol", "Gene ID"},
        )
    except Exception as exc:
        raise ResourceError(
            "NetInfer supplementary workbook could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    missing_target_columns = sorted({"UniProt AC", "Gene symbol"} - set(targets.columns))
    if missing_target_columns:
        raise ResourceError(
            "NetInfer Target information is missing required columns",
            details={"path": str(path), "missing_columns": missing_target_columns},
        )
    if "Gene ID" not in targets:
        targets["Gene ID"] = ""

    drugs = drugs.fillna("").copy()
    drugs["Drug ID"] = drugs["Drug ID"].astype(str).str.strip()
    drugs["SMILES (standardized)"] = drugs["SMILES (standardized)"].astype(str).str.strip()
    match_keys: list[str | None] = []
    total = len(drugs)
    for index, smiles in enumerate(drugs["SMILES (standardized)"], start=1):
        match_keys.append(
            standardize_to_match_key(
                smiles,
                normalizer=normalizer,
                chooser=chooser,
                dependencies=dependencies,
            )
        )
        if index % 1000 == 0 or index == total:
            progress("official_standardization", index, total)
    drugs["match_key"] = match_keys
    drugs = drugs[drugs["Drug ID"].ne("") & drugs["match_key"].notna()].copy()
    duplicate_structure_count = int(drugs.duplicated("match_key").sum())
    drugs = drugs.drop_duplicates("match_key", keep="first")
    drug_key_map = dict(zip(drugs["match_key"], drugs["Drug ID"], strict=False))

    targets = targets.fillna("").copy()
    targets["UniProt AC"] = targets["UniProt AC"].astype(str).str.strip()
    targets["Gene symbol"] = targets["Gene symbol"].astype(str).str.strip()
    targets["Gene ID"] = targets["Gene ID"].astype(str).str.strip()
    targets = targets[targets["UniProt AC"].ne("")].drop_duplicates("UniProt AC", keep="first")
    target_rows = list(
        zip(
            targets["UniProt AC"],
            targets["Gene symbol"],
            targets["Gene ID"],
            strict=False,
        )
    )
    return (
        drug_key_map,
        target_rows,
        {
            "official_drug_count": len(drugs),
            "official_duplicate_structure_count": duplicate_structure_count,
            "official_target_count": len(target_rows),
        },
    )


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    compounds_path: str | Path,
    supplementary_workbook_path: str | Path,
    settings: NetInferConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            "device_actual": "cpu",
            "torch_version": "not_used",
            "rdkit_version": package_version("rdkit"),
            "pandas_version": package_version("pandas"),
            "openpyxl_version": package_version("openpyxl"),
            "input_compound_count": 0,
            "official_drug_count": 0,
            "official_duplicate_structure_count": 0,
            "official_target_count": 0,
            "known_compound_count": 0,
            "known_drug_node_count": 0,
            "novel_compound_count": 0,
            "standardization_failed_count": 0,
            "batch_count": 0,
            "batch_size": settings.batch_size,
            "cs_edge_count": 0,
            "cs_unique_substructure_count": 0,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        validate_netinfer_settings(settings)
        compounds_file = resolve_run_input_file(
            context,
            compounds_path,
            label="compounds.normalized.csv",
            require_input_boundary=True,
        )
        workbook = resolve_resource_path(
            context,
            supplementary_workbook_path,
            label="NetInfer supplementary workbook",
        )
        mapping_output = resolve_netinfer_output(context, MAPPING_RELATIVE_PATH)
        target_output = resolve_netinfer_output(context, TARGET_MAP_RELATIVE_PATH)
        manifest_output = resolve_netinfer_output(context, BATCH_MANIFEST_RELATIVE_PATH)
        try:
            execution.resource_hashes["resources.netinfer.supplementary_workbook"] = sha256_file(
                workbook
            )
        except (OSError, RuntimeError) as exc:
            raise ResourceError("NetInfer supplementary workbook could not be hashed") from exc

        dependencies = load_prepare_dependencies()
        execution.update_metrics(
            {
                "rdkit_version": dependencies.rdkit_version,
                "pandas_version": str(dependencies.pd.__version__),
                "openpyxl_version": dependencies.openpyxl_version,
            }
        )
        normalizer = dependencies.standardize.Normalizer()
        chooser = dependencies.standardize.LargestFragmentChooser()

        def progress(stage: str, completed: int, total: int) -> None:
            execution.logger.info(
                f"{stage}_progress",
                "NetInfer preparation progress",
                completed=completed,
                total=total,
            )

        compounds = _load_compounds(compounds_file, dependencies=dependencies)
        execution.metric("input_compound_count", len(compounds))
        drug_key_map, target_rows, official_metrics = _load_official_tables(
            workbook,
            normalizer=normalizer,
            chooser=chooser,
            dependencies=dependencies,
            progress=progress,
        )
        execution.update_metrics(official_metrics)

        mapping_rows: list[dict[str, str]] = []
        novel_rows: list[dict[str, str]] = []
        total = len(compounds)
        for index, row in enumerate(compounds.itertuples(index=False), start=1):
            compound_id = str(row.ID)
            smiles = str(row.SMILES)
            match_key = standardize_to_match_key(
                smiles,
                normalizer=normalizer,
                chooser=chooser,
                dependencies=dependencies,
            )
            official_id = drug_key_map.get(match_key or "", "")
            source_type = "DRUG" if official_id else "COMPOUND"
            record = {
                "ID": compound_id,
                "SMILES": smiles,
                "match_key": match_key or "",
                "official_drug_id": official_id,
                "netinfer_input_type": source_type,
                "netinfer_input_id": official_id or compound_id,
                "batch_id": "",
            }
            mapping_rows.append(record)
            if source_type == "COMPOUND" and match_key:
                novel_rows.append(record)
            if index % 1000 == 0 or index == total:
                progress("input_standardization", index, total)

        batches: list[dict[str, Any]] = []
        all_substructures: set[str] = set()
        edge_count = 0
        for batch_number, start in enumerate(
            range(0, len(novel_rows), settings.batch_size), start=1
        ):
            batch_rows = novel_rows[start : start + settings.batch_size]
            batch_id = f"batch_{batch_number:04d}"
            for row in batch_rows:
                row["batch_id"] = batch_id
            batch_output = resolve_netinfer_output(context, f"batches/{batch_id}/CS.tsv")

            def cs_rows() -> Any:
                nonlocal edge_count
                for row in batch_rows:
                    molecule = dependencies.chem.MolFromSmiles(row["match_key"])
                    if molecule is None:
                        continue
                    fingerprint = dependencies.all_chem.GetMorganFingerprint(
                        molecule,
                        radius=settings.novel_compound_fingerprint.radius,
                        useFeatures=settings.novel_compound_fingerprint.use_features,
                    )
                    for substructure in sorted(fingerprint.GetNonzeroElements().keys()):
                        sub_id = str(substructure)
                        edge_count += 1
                        all_substructures.add(sub_id)
                        yield ("COMPOUND", row["ID"], "SUB", sub_id, "1")

            atomic_write_delimited(
                batch_output,
                cs_rows(),
                allowed_root=context.output_dir,
                delimiter="\t",
            )
            digest = sha256_file(batch_output)
            batches.append(
                {
                    "batch_id": batch_id,
                    "task_id": batch_id,
                    "compound_count": len(batch_rows),
                    "compound_ids": [row["ID"] for row in batch_rows],
                    "input_path": context.relative_path(batch_output),
                    "input_sha256": digest,
                    "prediction_path": (f"artifacts/netinfer/batches/{batch_id}/predictions.tsv"),
                }
            )
            execution.logger.info(
                "batch_prepared",
                "one independently retryable NetInfer batch was prepared",
                batch_id=batch_id,
                compound_count=len(batch_rows),
                completed=batch_number,
                total=(len(novel_rows) + settings.batch_size - 1) // settings.batch_size,
            )

        atomic_write_delimited(
            mapping_output,
            ([row[column] for column in MAPPING_COLUMNS] for row in mapping_rows),
            allowed_root=context.output_dir,
            delimiter="\t",
            header=MAPPING_COLUMNS,
        )
        atomic_write_delimited(
            target_output,
            target_rows,
            allowed_root=context.output_dir,
            delimiter="\t",
            header=("uniprot_id", "gene_symbol", "entrez_id"),
        )
        manifest = {
            "schema_version": "1.0",
            "batch_size": settings.batch_size,
            "novel_compound_count": len(novel_rows),
            "batch_count": len(batches),
            "batches": batches,
        }
        atomic_write_json(manifest_output, manifest, allowed_root=context.output_dir)

        known_count = sum(row["netinfer_input_type"] == "DRUG" for row in mapping_rows)
        standardization_failed = sum(not row["match_key"] for row in mapping_rows)
        execution.update_metrics(
            {
                "known_compound_count": known_count,
                "known_drug_node_count": len(
                    {
                        row["netinfer_input_id"]
                        for row in mapping_rows
                        if row["netinfer_input_type"] == "DRUG"
                    }
                ),
                "novel_compound_count": len(novel_rows),
                "standardization_failed_count": standardization_failed,
                "batch_count": len(batches),
                "cs_edge_count": edge_count,
                "cs_unique_substructure_count": len(all_substructures),
            }
        )
        execution.add_output("netinfer_input_mapping", mapping_output)
        execution.add_output("netinfer_target_map", target_output)
        execution.add_output("netinfer_batch_manifest", manifest_output)
        for batch in batches:
            batch_path = context.resolve_run_relative(batch["input_path"], must_exist=True)
            execution.add_output(
                "netinfer_batch_input",
                batch_path,
                instance_key=batch["batch_id"],
            )
        if official_metrics["official_duplicate_structure_count"]:
            execution.warn(
                "Official Drug information contained duplicate standardized structures; "
                "the first Drug ID was retained."
            )
        if standardization_failed:
            execution.warn(
                f"{standardization_failed} compound(s) could not produce a NetInfer match key."
            )
        execution.logger.info(
            "netinfer_inputs_prepared",
            "NetInfer mapping, target map, and batch manifest were written",
            known_compound_count=known_count,
            novel_compound_count=len(novel_rows),
            batch_count=len(batches),
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def netinfer_prepare_inputs(
    *,
    context: RunContext,
    compounds_path: str | Path,
    supplementary_workbook_path: str | Path,
    settings: NetInferConfig,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            compounds_path=compounds_path,
            supplementary_workbook_path=supplementary_workbook_path,
            settings=settings,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match official NetInfer DRUG nodes and prepare novel batches."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--compounds", required=True, type=Path)
    parser.add_argument(
        "--supplementary-workbook",
        type=Path,
        help="Optional explicit workbook below --resource-dir; config default otherwise.",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    workbook = (
        namespace.supplementary_workbook
        if namespace.supplementary_workbook is not None
        else environment.config.resources.netinfer.supplementary_workbook.raw
    )
    result = netinfer_prepare_inputs(
        context=environment.context,
        compounds_path=namespace.compounds,
        supplementary_workbook_path=workbook,
        settings=environment.config.netinfer,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    payload = result.to_dict()
    if result.status is NodeStatus.SUCCEEDED:
        manifest_path = resolve_netinfer_output(
            environment.context, BATCH_MANIFEST_RELATIVE_PATH
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["fanout_items"] = {
            "netinfer_predict_batch": [
                str(batch["batch_id"]) for batch in manifest["batches"]
            ]
        }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if result.status is NodeStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_MANIFEST_RELATIVE_PATH",
    "MAPPING_COLUMNS",
    "MAPPING_RELATIVE_PATH",
    "TARGET_MAP_RELATIVE_PATH",
    "build_parser",
    "main",
    "netinfer_prepare_inputs",
    "standardize_to_match_key",
]
