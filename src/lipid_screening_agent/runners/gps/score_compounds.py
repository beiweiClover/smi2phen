"""Standalone zRGES-like GPS compound reversal-scoring runner."""

from __future__ import annotations

import argparse
import gzip
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
from lipid_screening_agent.config.models import GPSScoringConfig
from lipid_screening_agent.runtime import ConfigurationError, InputError, RunContext
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import (
    add_execution_identity_arguments,
    execution_identity,
)
from ._common import (
    atomic_dataframe_to_csv,
    base_environment_metrics,
    resolve_gps_output,
    resolve_run_input_file,
)
from ._dependencies import ScoringDependencies, load_scoring_dependencies
from .algorithms import score_compound_profiles

NODE_ID = "gps_score_compounds"
OUTPUT_RELATIVE_PATH = "artifacts/gps/GPS_score.csv"


def _read_drug_profile(path: Path, *, dependencies: ScoringDependencies) -> Any:
    try:
        profile = dependencies.pd.read_csv(path, compression="gzip", index_col=0)
    except (OSError, EOFError, gzip.BadGzipFile, ValueError) as exc:
        raise InputError(
            "Entrez Drug_GPS could not be read",
            details={"path": str(path)},
        ) from exc
    if profile.empty or profile.shape[1] == 0:
        raise InputError("Entrez Drug_GPS contains no usable matrix")
    profile.index = profile.index.astype(str).str.strip()
    profile.columns = profile.columns.astype(str)
    if profile.columns.duplicated().any():
        raise InputError("Drug_GPS compound IDs must be unique")
    return profile


def _read_disease_profile(path: Path, *, dependencies: ScoringDependencies) -> Any:
    try:
        profile = dependencies.pd.read_csv(path, dtype={"GeneID": str})
    except Exception as exc:
        raise InputError(
            "Disease_GPS.csv could not be read",
            details={"path": str(path)},
        ) from exc
    missing = sorted({"GeneID", "disease_log2FC_mean"} - set(profile.columns))
    if missing:
        raise InputError(
            "Disease_GPS.csv is missing required columns",
            details={"missing_columns": missing},
        )
    return profile


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    drug_gps_path: str | Path,
    disease_gps_path: str | Path,
    settings: GPSScoringConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            **base_environment_metrics(),
            "device_actual": "cpu",
            "drug_profile_gene_count": 0,
            "disease_gene_count": 0,
            "aligned_gene_count": 0,
            "disease_up_gene_count": 0,
            "disease_down_gene_count": 0,
            "compound_count": 0,
            "random_set_size_count": 0,
            "random_background_samples": settings.random_background_samples,
            "seed": settings.seed,
            "lower_is_better": settings.lower_is_better,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        if not settings.lower_is_better:
            raise ConfigurationError("GPS score output semantics require lower_is_better=true")
        drug_file = resolve_run_input_file(context, drug_gps_path, label="Entrez Drug_GPS")
        disease_file = resolve_run_input_file(context, disease_gps_path, label="Disease_GPS.csv")
        output_path = resolve_gps_output(context, Path(OUTPUT_RELATIVE_PATH).name)
        dependencies = load_scoring_dependencies()
        execution.update_metrics(
            {
                "numpy_version": str(dependencies.np.__version__),
                "pandas_version": str(dependencies.pd.__version__),
            }
        )
        drug_profile = _read_drug_profile(drug_file, dependencies=dependencies)
        disease_profile = _read_disease_profile(disease_file, dependencies=dependencies)

        def progress(stage: str, completed: int, total: int) -> None:
            interval = 25 if stage == "random_background" else 500
            if completed == total or completed % interval == 0:
                execution.logger.info(
                    f"{stage}_progress",
                    "GPS scoring progress",
                    completed=completed,
                    total=total,
                )

        scores, metrics = score_compound_profiles(
            drug_profile,
            disease_profile,
            random_background_samples=settings.random_background_samples,
            seed=settings.seed,
            np=dependencies.np,
            pd=dependencies.pd,
            progress=progress,
        )
        expected_columns = ["ID", "GPS_score_zRGES_like_lower_better"]
        if scores.columns.tolist() != expected_columns or scores["ID"].duplicated().any():
            raise InputError("GPS score output did not satisfy its schema")
        atomic_dataframe_to_csv(
            scores,
            output_path,
            allowed_root=context.output_dir,
            index=False,
        )
        execution.update_metrics(metrics)
        execution.add_output("gps_scores", output_path)
        execution.logger.info(
            "gps_scores_written",
            "lower-is-better GPS compound scores were written",
            compound_count=len(scores),
            aligned_gene_count=metrics["aligned_gene_count"],
            relative_path=context.relative_path(output_path),
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def gps_score_compounds(
    *,
    context: RunContext,
    drug_gps_path: str | Path,
    disease_gps_path: str | Path,
    settings: GPSScoringConfig,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Score every compound using the frozen random background and zRGES-like formula."""

    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            drug_gps_path=drug_gps_path,
            disease_gps_path=disease_gps_path,
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
        description="Compute zRGES-like lower-is-better GPS compound scores."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--drug-gps", required=True, type=Path)
    parser.add_argument("--disease-gps", required=True, type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = gps_score_compounds(
        context=environment.context,
        drug_gps_path=namespace.drug_gps,
        disease_gps_path=namespace.disease_gps,
        settings=environment.config.gps.scoring,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    print(result.to_json())
    return 0 if result.status is NodeStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_RELATIVE_PATH",
    "build_parser",
    "gps_score_compounds",
    "main",
]
