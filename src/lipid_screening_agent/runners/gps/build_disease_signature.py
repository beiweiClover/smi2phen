"""Standalone disease DEG/signature runner with non-mutating Drug_GPS ID alignment."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import GPSDiseaseSignatureConfig
from lipid_screening_agent.runtime import (
    ConfigurationError,
    InputError,
    ResourceError,
    RunContext,
    file_digest,
    sha256_file,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import (
    add_execution_identity_arguments,
    execution_identity,
)
from ._common import (
    atomic_dataframe_to_csv,
    base_environment_metrics,
    package_version,
    resolve_gps_output,
    resolve_resource_path,
    resolve_run_input_file,
)
from ._dependencies import DiseaseDependencies, load_disease_dependencies
from .algorithms import (
    build_disease_profile,
    combine_direction_consistent_degs,
    compute_deg_table,
    convert_drug_profile_index_to_entrez,
)

NODE_ID = "gps_build_disease_signature"
DISEASE_OUTPUT_RELATIVE_PATH = "artifacts/gps/Disease_GPS.csv"
ENTREZ_DRUG_OUTPUT_RELATIVE_PATH = "artifacts/gps/Drug_GPS.entrez.csv.gz"
EXPRESSION_STATUSES = ("prepared", "skipped")


def _validate_settings(settings: GPSDiseaseSignatureConfig) -> None:
    expected = {
        "test": "welch_t_test",
        "multiple_testing": "benjamini_hochberg_after_expression_filter",
        "multi_comparison_combination": "direction_consistent_intersection",
    }
    observed = {
        "test": settings.test,
        "multiple_testing": settings.multiple_testing,
        "multi_comparison_combination": settings.multi_comparison_combination,
    }
    if observed != expected:
        raise ConfigurationError(
            "GPS disease-signature algorithm selectors must preserve the legacy workflow",
            details={"expected": expected, "configured": observed},
        )


def _load_expression_manifest(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(
            "expression comparison manifest could not be read",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "1.0":
        raise InputError("expression comparison manifest has an unsupported schema")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise InputError("expression comparison manifest contains no comparisons")
    required = {
        "comparison_id",
        "tpm_path",
        "metadata_path",
        "sample_counts",
        "sha256",
    }
    validated: list[Mapping[str, Any]] = []
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping) or not required.issubset(comparison):
            raise InputError(
                "expression comparison manifest item is missing required fields",
                details={"comparison_index": index},
            )
        validated.append(comparison)
    return tuple(validated)


def _manifest_input_file(
    context: RunContext,
    relative_path: Any,
    expected_hash: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise InputError(f"{label} path/hash in the expression manifest is invalid")
    resolved = resolve_run_input_file(context, relative_path, label=label)
    try:
        digest = file_digest(resolved).sha256
    except (OSError, RuntimeError) as exc:
        raise InputError(f"{label} could not be hashed safely") from exc
    if digest != expected_hash:
        raise InputError(
            f"{label} no longer matches the expression manifest SHA-256",
            details={"path": relative_path, "expected": expected_hash, "observed": digest},
        )
    return resolved


def _load_metadata(path: Path, *, dependencies: DiseaseDependencies) -> Any:
    pd = dependencies.pd
    try:
        metadata = pd.read_csv(path, sep="\t", dtype=str)
    except Exception as exc:
        raise InputError(
            "metadata file could not be read as TSV",
            details={"path": str(path)},
        ) from exc
    missing = sorted({"sample_id", "group"} - set(metadata.columns))
    if missing:
        raise InputError(
            "metadata file is missing required columns",
            details={"path": str(path), "missing_columns": missing},
        )
    metadata = metadata[["sample_id", "group"]].copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str).str.strip()
    metadata["group"] = metadata["group"].astype(str).str.strip().str.lower()
    metadata = metadata[metadata["sample_id"].ne("") & metadata["group"].ne("")]
    groups = set(metadata["group"].unique())
    if groups != {"control", "disease"}:
        raise InputError(
            "metadata groups must contain exactly control and disease",
            details={"path": str(path), "groups": sorted(groups)},
        )
    duplicated = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].head(10).tolist()
    if duplicated:
        raise InputError(
            "metadata sample_id values must be unique",
            details={"path": str(path), "duplicate_sample_ids": duplicated},
        )
    return metadata


def _run_comparison(
    *,
    tpm_path: Path,
    metadata_path: Path,
    comparison_name: str,
    settings: GPSDiseaseSignatureConfig,
    dependencies: DiseaseDependencies,
) -> Any:
    pd = dependencies.pd
    np = dependencies.np
    metadata = _load_metadata(metadata_path, dependencies=dependencies)
    try:
        expression = pd.read_csv(tpm_path, sep="\t")
    except Exception as exc:
        raise InputError(
            "TPM file could not be read as TSV",
            details={"path": str(tpm_path)},
        ) from exc
    if expression.shape[1] < 3:
        raise InputError("TPM requires one GeneID and at least two sample columns")
    expression = expression.rename(columns={expression.columns[0]: "GeneID"})
    expression["GeneID"] = expression["GeneID"].astype(str).str.strip()
    if expression["GeneID"].eq("").any():
        raise InputError("TPM contains an empty GeneID", details={"path": str(tpm_path)})
    duplicated_genes = expression.loc[expression["GeneID"].duplicated(), "GeneID"].head(10).tolist()
    if duplicated_genes:
        raise InputError(
            "TPM contains duplicate GeneID values",
            details={"path": str(tpm_path), "duplicate_gene_ids": duplicated_genes},
        )

    sample_columns = expression.columns[1:].astype(str).tolist()
    metadata_samples = metadata["sample_id"].astype(str).tolist()
    missing_samples = sorted(set(metadata_samples) - set(sample_columns))
    if missing_samples:
        raise InputError(
            "metadata contains samples absent from the TPM matrix",
            details={"missing_sample_ids": missing_samples[:10]},
        )
    metadata_set = set(metadata_samples)
    metadata = (
        metadata.set_index("sample_id")
        .loc[[sample for sample in sample_columns if sample in metadata_set]]
        .reset_index()
    )
    selected_samples = metadata["sample_id"].tolist()
    if len(selected_samples) < 2:
        raise InputError(f"{comparison_name} has fewer than two usable samples")
    try:
        tpm_matrix = (
            expression[selected_samples]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
    except Exception as exc:
        raise InputError("TPM values could not be converted to numeric data") from exc
    if np.isnan(tpm_matrix).any():
        raise InputError("TPM contains values that cannot be converted to numbers")
    return compute_deg_table(
        gene_ids=expression["GeneID"].astype(str).tolist(),
        tpm_matrix=tpm_matrix,
        sample_ids=selected_samples,
        groups=metadata["group"].to_numpy(),
        comparison_name=comparison_name,
        tpm_filename=tpm_path.name,
        metadata_filename=metadata_path.name,
        fdr_cutoff=settings.fdr_cutoff,
        absolute_log2fc_cutoff=settings.absolute_log2fc_cutoff,
        tpm_filter_cutoff=settings.tpm_filter_cutoff,
        minimum_group_fraction_expressed=settings.minimum_group_fraction_expressed,
        np=np,
        pd=pd,
        stats=dependencies.stats,
        multipletests=dependencies.multipletests,
    )


def load_symbol_to_entrez_map(path: Path, *, dependencies: DiseaseDependencies) -> dict[str, str]:
    """Reproduce the legacy case-sensitive Symbol/Synonyms mapping and first-win order."""

    pd = dependencies.pd
    compression = "gzip" if path.suffix.casefold() == ".gz" else None
    try:
        gene_info = pd.read_csv(
            path,
            sep="\t",
            compression=compression,
            dtype=str,
            usecols=["GeneID", "Symbol", "Synonyms"],
        )
    except Exception as exc:
        raise ResourceError(
            "human gene_info could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    gene_info = gene_info.dropna(subset=["GeneID", "Symbol"]).copy()
    gene_info["GeneID"] = gene_info["GeneID"].astype(str).str.strip()
    gene_info["Symbol"] = gene_info["Symbol"].astype(str).str.strip()
    gene_info["Synonyms"] = gene_info["Synonyms"].fillna("").astype(str)
    gene_info = gene_info[gene_info["GeneID"].ne("") & gene_info["Symbol"].ne("")].drop_duplicates(
        "Symbol", keep="first"
    )
    if gene_info.empty:
        raise ResourceError("human gene_info contains no usable mappings")
    mapping = dict(zip(gene_info["Symbol"], gene_info["GeneID"], strict=False))
    for row in gene_info.itertuples(index=False):
        for synonym in str(row.Synonyms).split("|"):
            synonym = synonym.strip()
            if synonym and synonym != "-":
                mapping.setdefault(synonym, str(row.GeneID))
    return mapping


def _read_drug_profile(path: Path, *, dependencies: DiseaseDependencies) -> Any:
    try:
        profile = dependencies.pd.read_csv(path, compression="gzip", index_col=0)
    except (OSError, EOFError, gzip.BadGzipFile, ValueError) as exc:
        raise InputError(
            "Drug_GPS.csv.gz could not be read",
            details={"path": str(path)},
        ) from exc
    if profile.empty or profile.shape[1] == 0:
        raise InputError("Drug_GPS.csv.gz contains no usable matrix")
    profile.columns = profile.columns.astype(str)
    if profile.columns.duplicated().any():
        raise InputError("Drug_GPS compound IDs must be unique")
    return profile


def _zero_metrics(settings: GPSDiseaseSignatureConfig) -> dict[str, Any]:
    return {
        **base_environment_metrics(),
        "device_actual": "cpu",
        "scipy_version": package_version("scipy"),
        "statsmodels_version": package_version("statsmodels"),
        "comparison_count": 0,
        "sample_count": 0,
        "control_sample_count": 0,
        "disease_sample_count": 0,
        "genes_tested_total": 0,
        "significant_deg_total": 0,
        "direction_consistent_deg_count": 0,
        "disease_profile_gene_count": 0,
        "disease_up_gene_count": 0,
        "disease_down_gene_count": 0,
        "drug_profile_input_gene_count": 0,
        "drug_profile_entrez_gene_count": 0,
        "drug_profile_unmapped_gene_count": 0,
        "drug_profile_duplicate_entrez_row_count": 0,
        "fdr_cutoff": settings.fdr_cutoff,
        "absolute_log2fc_cutoff": settings.absolute_log2fc_cutoff,
        "tpm_filter_cutoff": settings.tpm_filter_cutoff,
        "minimum_group_fraction_expressed": settings.minimum_group_fraction_expressed,
        "elapsed_seconds": 0.0,
    }


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    expression_status: str,
    expression_manifest_path: str | Path | None,
    drug_gps_path: str | Path | None,
    gene_info_path: str | Path | None,
    settings: GPSDiseaseSignatureConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(_zero_metrics(settings))
    try:
        if expression_status not in EXPRESSION_STATUSES:
            raise InputError(
                "expression_status must be prepared or skipped",
                details={"expression_status": expression_status},
            )
        if expression_status == "skipped":
            execution.mark_skipped("expression input preparation was marked skipped")
            return
        _validate_settings(settings)
        if expression_manifest_path is None:
            raise InputError("prepared expression input requires an explicit manifest path")
        if drug_gps_path is None:
            raise InputError("prepared expression input requires an explicit Drug_GPS path")
        if gene_info_path is None:
            raise ResourceError("Drug_GPS GeneSymbol alignment requires gene_info")

        manifest_file = resolve_run_input_file(
            context,
            expression_manifest_path,
            label="expression comparison manifest",
            require_input_boundary=True,
        )
        raw_drug_file = resolve_run_input_file(context, drug_gps_path, label="Drug_GPS.csv.gz")
        gene_info_file = resolve_resource_path(context, gene_info_path, label="human gene_info")
        disease_output = resolve_gps_output(context, Path(DISEASE_OUTPUT_RELATIVE_PATH).name)
        entrez_drug_output = resolve_gps_output(
            context, Path(ENTREZ_DRUG_OUTPUT_RELATIVE_PATH).name
        )
        try:
            execution.resource_hashes["resources.gps.human_gene_info"] = sha256_file(gene_info_file)
        except (OSError, RuntimeError) as exc:
            raise ResourceError("human gene_info could not be hashed safely") from exc

        dependencies = load_disease_dependencies()
        execution.update_metrics(
            {
                "numpy_version": str(dependencies.np.__version__),
                "pandas_version": str(dependencies.pd.__version__),
                "scipy_version": package_version("scipy"),
                "statsmodels_version": package_version("statsmodels"),
            }
        )
        comparisons = _load_expression_manifest(manifest_file)
        deg_results: list[Any] = []
        sample_count = control_count = disease_count = genes_tested = significant = 0
        for index, comparison in enumerate(comparisons, start=1):
            hashes = comparison["sha256"]
            if not isinstance(hashes, Mapping):
                raise InputError("expression comparison sha256 field must be an object")
            tpm_path = _manifest_input_file(
                context,
                comparison["tpm_path"],
                hashes.get("tpm"),
                label="TPM input",
            )
            metadata_path = _manifest_input_file(
                context,
                comparison["metadata_path"],
                hashes.get("metadata"),
                label="metadata input",
            )
            comparison_name = str(comparison["comparison_id"])
            result = _run_comparison(
                tpm_path=tpm_path,
                metadata_path=metadata_path,
                comparison_name=comparison_name,
                settings=settings,
                dependencies=dependencies,
            )
            deg_results.append(result)
            group_counts = comparison.get("sample_counts")
            if not isinstance(group_counts, Mapping):
                raise InputError("expression comparison sample_counts must be an object")
            actual_control_count = int(result["n_control"].iloc[0])
            actual_disease_count = int(result["n_disease"].iloc[0])
            sample_count += actual_control_count + actual_disease_count
            control_count += actual_control_count
            disease_count += actual_disease_count
            genes_tested += len(result)
            significant_count = int(result["regulation"].isin(["up", "down"]).sum())
            significant += significant_count
            execution.logger.info(
                "deg_comparison_completed",
                "one expression comparison completed",
                completed=index,
                total=len(comparisons),
                comparison_id=comparison_name,
                genes_tested=len(result),
                significant_gene_count=significant_count,
            )

        core_degs = combine_direction_consistent_degs(deg_results, pd=dependencies.pd)
        raw_drug_profile = _read_drug_profile(raw_drug_file, dependencies=dependencies)
        mapping = load_symbol_to_entrez_map(gene_info_file, dependencies=dependencies)
        entrez_drug_profile, alignment, unmapped = convert_drug_profile_index_to_entrez(
            raw_drug_profile,
            mapping,
            np=dependencies.np,
            pd=dependencies.pd,
        )
        disease_profile = build_disease_profile(
            core_degs,
            entrez_drug_profile.index.astype(str).tolist(),
            np=dependencies.np,
        )

        atomic_dataframe_to_csv(
            entrez_drug_profile,
            entrez_drug_output,
            allowed_root=context.output_dir,
            index=True,
            index_label="GeneID",
            gzip_compression=True,
        )
        atomic_dataframe_to_csv(
            disease_profile,
            disease_output,
            allowed_root=context.output_dir,
            index=False,
        )
        execution.add_output("gps_drug_profiles_entrez", entrez_drug_output)
        execution.add_output("gps_disease_signature", disease_output)
        execution.update_metrics(
            {
                "comparison_count": len(comparisons),
                "sample_count": sample_count,
                "control_sample_count": control_count,
                "disease_sample_count": disease_count,
                "genes_tested_total": genes_tested,
                "significant_deg_total": significant,
                "direction_consistent_deg_count": len(core_degs),
                "disease_profile_gene_count": len(disease_profile),
                "disease_up_gene_count": int((disease_profile["disease_direction"] == "up").sum()),
                "disease_down_gene_count": int(
                    (disease_profile["disease_direction"] == "down").sum()
                ),
                "drug_profile_input_gene_count": alignment["input_gene_count"],
                "drug_profile_entrez_gene_count": alignment["entrez_gene_count"],
                "drug_profile_unmapped_gene_count": alignment["unmapped_gene_count"],
                "drug_profile_duplicate_entrez_row_count": alignment["duplicate_entrez_row_count"],
            }
        )
        if unmapped:
            execution.warn(
                f"Skipped {len(unmapped)} Drug_GPS GeneSymbol row(s) without an Entrez mapping."
            )
        if alignment["duplicate_entrez_row_count"]:
            execution.warn(
                "Removed "
                f"{alignment['duplicate_entrez_row_count']} duplicate Entrez-mapped "
                "Drug_GPS row(s)."
            )
        execution.logger.info(
            "disease_signature_written",
            "Disease_GPS and a separate Entrez-aligned Drug_GPS were written",
            disease_gene_count=len(disease_profile),
            drug_gene_count=len(entrez_drug_profile),
            raw_drug_profile_unchanged=True,
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def gps_build_disease_signature(
    *,
    context: RunContext,
    expression_status: str,
    expression_manifest_path: str | Path | None,
    drug_gps_path: str | Path | None,
    gene_info_path: str | Path | None,
    settings: GPSDiseaseSignatureConfig,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Build Disease_GPS, or return skipped when expression preparation was skipped."""

    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            expression_status=expression_status,
            expression_manifest_path=expression_manifest_path,
            drug_gps_path=drug_gps_path,
            gene_info_path=gene_info_path,
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
        description="Build direction-consistent Disease_GPS from prepared comparisons."
    )
    add_common_runner_arguments(parser)
    parser.add_argument(
        "--expression-status",
        choices=EXPRESSION_STATUSES,
        default="prepared",
        help="Pass skipped when prepare_expression_inputs returned status=skipped.",
    )
    parser.add_argument("--expression-manifest", type=Path)
    parser.add_argument("--drug-gps", type=Path)
    parser.add_argument(
        "--gene-info",
        type=Path,
        help="Optional explicit gene_info path below --resource-dir; config default otherwise.",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    gene_info = (
        namespace.gene_info
        if namespace.gene_info is not None
        else environment.config.resources.gps.human_gene_info.raw
    )
    result = gps_build_disease_signature(
        context=environment.context,
        expression_status=namespace.expression_status,
        expression_manifest_path=namespace.expression_manifest,
        drug_gps_path=namespace.drug_gps,
        gene_info_path=gene_info,
        settings=environment.config.gps.disease_signature,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    print(result.to_json())
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.SKIPPED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISEASE_OUTPUT_RELATIVE_PATH",
    "ENTREZ_DRUG_OUTPUT_RELATIVE_PATH",
    "build_parser",
    "gps_build_disease_signature",
    "load_symbol_to_entrez_map",
    "main",
]
