"""Standalone GPS compound perturbation-profile prediction runner."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import os
import sys
import time
import warnings
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
from lipid_screening_agent.config.models import GPSDrugProfilesConfig
from lipid_screening_agent.runtime import (
    ConfigurationError,
    EnvironmentError,
    InputError,
    ResourceError,
    RunContext,
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
    resolve_gps_output,
    resolve_resource_path,
    resolve_run_input_file,
)
from ._dependencies import PredictionDependencies, load_prediction_dependencies
from .algorithms import gps_prob_to_change

NODE_ID = "gps_predict_drug_profiles"
OUTPUT_RELATIVE_PATH = "artifacts/gps/Drug_GPS.csv.gz"
LEGACY_CELL_LINES = ("HEPG2_t0", "MCF7_t1", "PC3_t1", "VCAP_t1")


def _validate_settings(settings: GPSDrugProfilesConfig) -> None:
    if tuple(settings.cell_lines) != LEGACY_CELL_LINES:
        raise ConfigurationError(
            "gps.drug_profiles.cell_lines must preserve the legacy four-cell-line order",
            details={
                "expected": list(LEGACY_CELL_LINES),
                "configured": list(settings.cell_lines),
            },
        )
    if settings.output_cell_line != "HEPG2_t0":
        raise ConfigurationError("GPS drug-profile output_cell_line must be HEPG2_t0")
    if not settings.preserve_legacy_cell_line_order:
        raise ConfigurationError(
            "preserve_legacy_cell_line_order must remain true for reproducibility"
        )
    if settings.fingerprint.algorithm != "morgan_bit_vector":
        raise ConfigurationError("unsupported GPS fingerprint algorithm")
    if settings.fingerprint.use_features:
        raise ConfigurationError("GPS Morgan bit fingerprints require use_features=false")


def _load_compounds(path: Path, *, dependencies: PredictionDependencies) -> Any:
    pd = dependencies.pd
    try:
        table = pd.read_csv(
            path,
            dtype={"ID": str, "SMILES": str},
            keep_default_na=False,
        )
    except Exception as exc:
        raise InputError(
            "compounds.normalized.csv could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    missing = sorted({"ID", "SMILES"} - set(table.columns))
    if missing:
        raise InputError(
            "compounds.normalized.csv is missing required columns",
            details={"missing_columns": missing},
        )
    table = table.copy()
    table["ID"] = table["ID"].astype(str).str.strip()
    table["SMILES"] = table["SMILES"].astype(str).str.strip()
    if table.empty:
        raise InputError("compounds.normalized.csv contains no compounds")
    if table["ID"].eq("").any() or table["SMILES"].eq("").any():
        raise InputError("normalized compound IDs and SMILES must be non-empty")
    duplicates = table.loc[table["ID"].duplicated(keep=False), "ID"].unique().tolist()
    if duplicates:
        raise InputError(
            "normalized compound IDs must be unique",
            details={"duplicate_ids": sorted(map(str, duplicates))[:50]},
        )
    return table


def smiles_to_fingerprints(
    smiles: Sequence[str],
    *,
    n_bits: int,
    radius: int,
    dependencies: PredictionDependencies,
    progress: Any | None = None,
) -> Any:
    """Build the exact legacy Morgan bit-vector representation."""

    np = dependencies.np
    fingerprints = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for index, value in enumerate(smiles, start=1):
        molecule = dependencies.chem.MolFromSmiles(value)
        if molecule is None:
            raise InputError(
                "compounds.normalized.csv contains an RDKit-invalid SMILES",
                details={"row_index": index - 1, "smiles": value},
            )
        fingerprint = dependencies.all_chem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=radius,
            nBits=n_bits,
            useFeatures=False,
        )
        fingerprints[index - 1] = np.fromiter(
            (1.0 if character == "1" else 0.0 for character in fingerprint.ToBitString()),
            dtype=np.float32,
            count=n_bits,
        )
        if progress is not None and (index % 5000 == 0 or index == len(smiles)):
            progress(index, len(smiles))
    return fingerprints


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "GPS device was set to cuda but torch.cuda.is_available() is false",
                details={"device_requested": requested},
                retryable=False,
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ConfigurationError("GPS device must be auto, cuda, or cpu")


def _seed_everything(torch: Any, np: Any, seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _load_model_module(model_file: Path) -> Any:
    """Execute the resource-owned model definition without copying or reimplementing it."""

    spec = importlib.util.spec_from_file_location("model", model_file)
    if spec is None or spec.loader is None:
        raise EnvironmentError(
            "could not create an import specification for GPS model.py",
            details={"path": str(model_file)},
            retryable=False,
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("model")
    sys.modules["model"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        if previous is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous
        raise EnvironmentError(
            "GPS model.py could not be imported in the current environment",
            details={"path": str(model_file), "error_type": type(exc).__name__},
            retryable=False,
        ) from exc
    return module


def _load_gps_model(
    *,
    torch: Any,
    model_file: Path,
    model_definition: Path,
    device: Any,
) -> Any:
    """Load one resource checkpoint using its original ``model.MLP`` definition."""

    _load_model_module(model_definition)
    load_kwargs: dict[str, Any] = {"map_location": device}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            # PyTorch 2.6 changed the default; False restores legacy whole-object loading.
            load_kwargs["weights_only"] = False
    except (TypeError, ValueError):
        pass
    try:
        loaded = torch.load(model_file, **load_kwargs)
    except Exception as exc:
        raise EnvironmentError(
            "GPS checkpoint could not be deserialized; "
            "Torch/checkpoint compatibility is not satisfied",
            details={
                "path": str(model_file),
                "torch_version": str(getattr(torch, "__version__", "unknown")),
                "error_type": type(exc).__name__,
            },
            retryable=False,
        ) from exc
    if not isinstance(loaded, dict) or "model0" not in loaded:
        raise ResourceError(
            "GPS checkpoint does not contain the expected model0 entry",
            details={"path": str(model_file)},
        )
    try:
        model = loaded["model0"].to(device)
        model.eval()
    except Exception as exc:
        raise EnvironmentError(
            "GPS model could not be moved to the selected device",
            details={"path": str(model_file), "device": str(device)},
            retryable=False,
        ) from exc
    return model


def _load_gene_features(path: Path, *, dependencies: PredictionDependencies) -> Any:
    try:
        frame = dependencies.pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str)
        frame = frame.astype(dependencies.np.float32)
    except Exception as exc:
        raise ResourceError(
            "GPS gene-feature table could not be read as a numeric CSV",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if frame.empty or frame.index.has_duplicates:
        raise ResourceError(
            "GPS gene-feature table must contain unique gene rows",
            details={"path": str(path)},
        )
    return frame


def _predict_gene_probabilities(
    model: Any,
    fingerprints: Any,
    gene_feature: Any,
    *,
    batch_size: int,
    device: Any,
    dependencies: PredictionDependencies,
) -> Any:
    np = dependencies.np
    torch = dependencies.torch
    output = np.empty((fingerprints.shape[0], 3), dtype=np.float32)
    feature = gene_feature.astype(np.float32).reshape(1, -1)
    with torch.no_grad():
        for start in range(0, fingerprints.shape[0], batch_size):
            end = min(start + batch_size, fingerprints.shape[0])
            drug_batch = fingerprints[start:end]
            gene_batch = np.repeat(feature, end - start, axis=0)
            joined = np.concatenate([drug_batch, gene_batch], axis=1).astype(np.float32, copy=False)
            tensor = torch.from_numpy(joined).to(device)
            logits = model(tensor)
            output[start:end] = (
                dependencies.functional.softmax(logits, dim=1).detach().cpu().numpy()
            )
    return output


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    compounds_path: str | Path,
    model_code_path: str | Path,
    model_data_path: str | Path,
    settings: GPSDrugProfilesConfig,
    requested_device: str,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            **base_environment_metrics(),
            "device_requested": requested_device,
            "input_compound_count": 0,
            "fingerprint_count": 0,
            "cell_line_count": len(settings.cell_lines),
            "prediction_gene_union_count": 0,
            "output_gene_count": 0,
            "batch_size": settings.batch_size,
            "probability_threshold": settings.probability_threshold,
            "seed": settings.seed,
            "fingerprint_bits": settings.fingerprint.bits,
            "fingerprint_radius": settings.fingerprint.radius,
            "fingerprint_use_features": settings.fingerprint.use_features,
            "value_negative_one_count": 0,
            "value_zero_count": 0,
            "value_positive_one_count": 0,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        _validate_settings(settings)
        output_path = resolve_gps_output(context, Path(OUTPUT_RELATIVE_PATH).name)
        compounds_file = resolve_run_input_file(
            context,
            compounds_path,
            label="compounds.normalized.csv",
            require_input_boundary=True,
        )
        code_dir = resolve_resource_path(
            context, model_code_path, label="GPS model code directory", kind="directory"
        )
        data_dir = resolve_resource_path(
            context, model_data_path, label="GPS model data directory", kind="directory"
        )
        model_definition = resolve_resource_path(
            context, code_dir / "model.py", label="GPS model.py"
        )

        resource_files: dict[str, Path] = {"gps.model_definition": model_definition}
        model_files: dict[str, Path] = {}
        feature_files: dict[str, Path] = {}
        for cell_line in settings.cell_lines:
            model_files[cell_line] = resolve_resource_path(
                context,
                code_dir / "results" / cell_line / "multi" / "model.pkl",
                label=f"GPS {cell_line} checkpoint",
            )
            feature_files[cell_line] = resolve_resource_path(
                context,
                data_dir / "input_gene_features" / f"go_fingerprints_2k_{cell_line}.csv",
                label=f"GPS {cell_line} gene features",
            )
            resource_files[f"gps.model.{cell_line}"] = model_files[cell_line]
            resource_files[f"gps.gene_features.{cell_line}"] = feature_files[cell_line]
        for resource_id, resource_file in resource_files.items():
            try:
                execution.resource_hashes[resource_id] = sha256_file(resource_file)
            except (OSError, RuntimeError) as exc:
                raise ResourceError(
                    "GPS resource changed or could not be hashed",
                    details={"path": str(resource_file)},
                ) from exc

        dependencies = load_prediction_dependencies()
        execution.update_metrics(
            {
                "torch_version": str(getattr(dependencies.torch, "__version__", "unknown")),
                "rdkit_version": dependencies.rdkit_version,
                "numpy_version": str(dependencies.np.__version__),
                "pandas_version": str(dependencies.pd.__version__),
            }
        )
        device = _resolve_device(dependencies.torch, requested_device)
        execution.metric("device_actual", str(device))
        _seed_everything(dependencies.torch, dependencies.np, settings.seed)
        warnings.filterwarnings("ignore", message="dropout2d: Received a 2-D input")

        compounds = _load_compounds(compounds_file, dependencies=dependencies)
        execution.metric("input_compound_count", len(compounds))
        fingerprints = smiles_to_fingerprints(
            compounds["SMILES"].astype(str).tolist(),
            n_bits=settings.fingerprint.bits,
            radius=settings.fingerprint.radius,
            dependencies=dependencies,
            progress=lambda completed, total: execution.logger.info(
                "fingerprint_progress",
                "Morgan fingerprint generation progress",
                completed=completed,
                total=total,
            ),
        )
        execution.metric("fingerprint_count", fingerprints.shape[0])

        models: dict[str, Any] = {}
        features: dict[str, Any] = {}
        feature_by_gene: dict[str, dict[str, Any]] = {}
        for cell_line in settings.cell_lines:
            models[cell_line] = _load_gps_model(
                torch=dependencies.torch,
                model_file=model_files[cell_line],
                model_definition=model_definition,
                device=device,
            )
            feature_table = _load_gene_features(feature_files[cell_line], dependencies=dependencies)
            features[cell_line] = feature_table
            feature_by_gene[cell_line] = {
                gene: feature_table.loc[gene].to_numpy(dtype=dependencies.np.float32)
                for gene in feature_table.index
            }
            execution.logger.info(
                "model_loaded",
                "GPS cell-line model and gene features loaded",
                cell_line=cell_line,
                gene_count=feature_table.shape[0],
                feature_count=feature_table.shape[1],
            )

        output_genes = features[settings.output_cell_line].index.astype(str).tolist()
        union_genes = sorted(
            set().union(*(set(table.index.astype(str)) for table in features.values()))
        )
        output_gene_index = {gene: index for index, gene in enumerate(output_genes)}
        execution.metric("prediction_gene_union_count", len(union_genes))
        execution.metric("output_gene_count", len(output_genes))
        output_change = dependencies.np.zeros(
            (len(compounds), len(output_genes)), dtype=dependencies.np.int8
        )

        for gene_index, gene in enumerate(union_genes, start=1):
            probabilities: list[Any] = []
            for cell_line in settings.cell_lines:
                feature = feature_by_gene[cell_line].get(gene)
                if feature is None:
                    continue
                probability = _predict_gene_probabilities(
                    models[cell_line],
                    fingerprints,
                    feature,
                    batch_size=settings.batch_size,
                    device=device,
                    dependencies=dependencies,
                )
                probabilities.append(probability)
                if cell_line == settings.output_cell_line:
                    output_change[:, output_gene_index[gene]] = gps_prob_to_change(
                        probability,
                        settings.probability_threshold,
                        np=dependencies.np,
                    )

            # Preserve the legacy four-cell-line MEDIAN path and its call ordering.
            median_probability = dependencies.np.median(
                dependencies.np.stack(probabilities, axis=0), axis=0
            )
            gps_prob_to_change(
                median_probability,
                settings.probability_threshold,
                np=dependencies.np,
            )
            if gene_index % 50 == 0 or gene_index == len(union_genes):
                execution.logger.info(
                    "gene_prediction_progress",
                    "GPS gene prediction progress",
                    completed=gene_index,
                    total=len(union_genes),
                )
                gc.collect()

        profile = dependencies.pd.DataFrame(
            output_change.T,
            index=output_genes,
            columns=compounds["ID"].astype(str).tolist(),
        )
        profile.index.name = "GeneSymbol"
        atomic_dataframe_to_csv(
            profile,
            output_path,
            allowed_root=context.output_dir,
            index=True,
            index_label="GeneSymbol",
            gzip_compression=True,
        )
        counts = dependencies.pd.Series(profile.to_numpy().ravel()).value_counts()
        execution.update_metrics(
            {
                "value_negative_one_count": int(counts.get(-1, 0)),
                "value_zero_count": int(counts.get(0, 0)),
                "value_positive_one_count": int(counts.get(1, 0)),
            }
        )
        execution.add_output("gps_drug_profiles", output_path)
        execution.logger.info(
            "drug_profiles_written",
            "HEPG2_t0 GPS drug profiles were written",
            gene_count=len(output_genes),
            compound_count=len(compounds),
            relative_path=context.relative_path(output_path),
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def gps_predict_drug_profiles(
    *,
    context: RunContext,
    compounds_path: str | Path,
    model_code_path: str | Path,
    model_data_path: str | Path,
    settings: GPSDrugProfilesConfig,
    config_hash: str,
    code_version: str,
    device: str | None = None,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Predict and atomically commit the legacy HEPG2_t0 drug profile matrix."""

    requested_device = settings.device if device is None else device
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            compounds_path=compounds_path,
            model_code_path=model_code_path,
            model_data_path=model_data_path,
            settings=settings,
            requested_device=requested_device,
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
        description="Predict GPS drug perturbation profiles from compounds.normalized.csv."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--compounds", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = gps_predict_drug_profiles(
        context=environment.context,
        compounds_path=namespace.compounds,
        model_code_path=environment.config.resources.gps.model_code.raw,
        model_data_path=environment.config.resources.gps.model_data.raw,
        settings=environment.config.gps.drug_profiles,
        config_hash=environment.config_hash,
        code_version=__version__,
        device=namespace.device,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    print(result.to_json())
    return 0 if result.status is NodeStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LEGACY_CELL_LINES",
    "OUTPUT_RELATIVE_PATH",
    "build_parser",
    "gps_predict_drug_profiles",
    "main",
    "smiles_to_fingerprints",
]
