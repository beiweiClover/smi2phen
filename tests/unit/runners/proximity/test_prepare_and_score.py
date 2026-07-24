import csv
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.proximity.algorithms import calculate_target_z
from lipid_screening_agent.runners.proximity.prepare_network import (
    _read_target_mapping,
    proximity_prepare_network,
)
from lipid_screening_agent.runners.proximity.score_compounds import proximity_score_compounds
from lipid_screening_agent.runtime import RunContext, atomic_write_json
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

openpyxl = pytest.importorskip(
    "openpyxl", reason="proximity unit tests require the optional openpyxl extra"
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)
TEST_SETTINGS = replace(
    CONFIG.proximity,
    randomizations=16,
    minimum_degree_bin_size=2,
    job_batch_size=1,
)


def test_production_scientific_defaults_are_unchanged() -> None:
    assert CONFIG.proximity.randomizations == 1000
    assert CONFIG.proximity.minimum_degree_bin_size == 100
    assert CONFIG.proximity.seed == 452456
    assert CONFIG.proximity.job_batch_size == 32
    assert CONFIG.proximity.randomization == "degree_matched"
    assert CONFIG.proximity.background_component == "largest_connected_component"
    assert CONFIG.proximity.lower_is_better is True


def test_legacy_z_formula_uses_seeded_degree_matched_population_std() -> None:
    z = calculate_target_z(
        (0, 1),
        real_disease_distance=np.asarray([0, 1, 2, 3], dtype=np.int32),
        random_disease_distances=np.asarray(
            [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]], dtype=np.int32
        ),
        node_to_equivalent={0: (1, 2, 3), 1: (0, 2, 3)},
        n_random=3,
        seed=452456,
    )
    assert z == -2.1213203435596424


def test_legacy_target_information_workbook_mapping(tmp_path: Path) -> None:
    workbook_path = tmp_path / "supplementary.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Target information"
    sheet.append(
        [
            "UniProt AC",
            "Name",
            "Protein family",
            "Gene symbol",
            "Gene ID",
            "Organism",
        ]
    )
    sheet.append(["P1", "one", "family", "Gene1", "1;legacy", "Human"])
    workbook.save(workbook_path)
    assert _read_target_mapping(workbook_path) == {"GENE1": "1"}


def _context(tmp_path: Path, run_id: str) -> tuple[RunContext, Path, Path]:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    prepared = project / "runs" / run_id / "inputs/prepared"
    output = project / "runs" / run_id / "artifacts/proximity"
    project.mkdir(parents=True)
    resources.mkdir(parents=True)
    context = RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resources,
        input_dir=prepared,
        output_dir=output,
    )
    return context, prepared, resources


def _runtime_manifest(context: RunContext, node_id: str, artifact_id: str) -> Path:
    return context.resolve_run_relative(
        f"artifacts/manifests/{node_id}/main/{artifact_id}.json",
        must_exist=True,
    )


def _commit_drug_targets(context: RunContext, payload: dict[str, object]) -> Path:
    target_path = context.resolve_run_relative("artifacts/netinfer/drug_targets.json")

    def operation(execution: NodeExecution) -> None:
        atomic_write_json(target_path, payload, allowed_root=context.run_dir)
        execution.add_output("drug_targets", target_path)

    result = execute_node(
        operation,
        context=context,
        node_id="netinfer_merge_targets",
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED
    return _runtime_manifest(context, "netinfer_merge_targets", result.outputs[0])


def _write_inputs(
    context: RunContext,
    prepared: Path,
    resources: Path,
    *,
    payload: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path]:
    ppi = resources / "small_interactome.tsv"
    ppi.write_text(
        "#Protein A\tProtein B\tSource\tType\n"
        "1\t2\ttest\ttest\n"
        "2\t3\ttest\ttest\n"
        "3\t4\ttest\ttest\n"
        "4\t5\ttest\ttest\n"
        "5\t6\ttest\ttest\n"
        "6\t7\ttest\ttest\n"
        "7\t8\ttest\ttest\n"
        "8\t1\ttest\ttest\n"
        "2\t5\ttest\ttest\n"
        "9\t10\ttest\ttest\n"
        "1\t2\tduplicate\ttest\n"
        "3\t3\tself\ttest\n",
        encoding="utf-8",
    )
    mapping = resources / "target_mapping.tsv"
    mapping.write_text(
        "Gene symbol\tGene ID\nGENE1\t1\nGENE2\t4\nGENE7\t7\nGENEOUT\t9\n",
        encoding="utf-8",
    )
    disease = prepared / "disease_genes.normalized.tsv"
    disease.parent.mkdir(parents=True, exist_ok=True)
    disease.write_text(
        "symbol\tentrez_id\nD1\t1\nD1_DUPLICATE\t1\nD2\t2\nOUTSIDE_LCC\t9\nNOT_IN_PPI\t99\n",
        encoding="utf-8",
    )
    if payload is None:
        payload = {
            "drug-a": {
                "smiles": "CCO",
                "targets": [
                    {
                        "gene_symbol": "GENE1",
                        "uniprot_id": "P1",
                        "evidence": "known",
                        "score": 1.0,
                    },
                    {
                        "gene_symbol": "GENE2",
                        "uniprot_id": "P2",
                        "evidence": "predicted",
                        "score": 0.8,
                        "prediction_rank": 1,
                    },
                ],
            },
            "drug-b": {
                "smiles": "CCN",
                "targets": [
                    {
                        "gene_symbol": "GENE7",
                        "uniprot_id": "P7",
                        "evidence": "predicted",
                        "score": 0.7,
                        "prediction_rank": 2,
                    }
                ],
            },
            "drug-empty": {"smiles": "CCC", "targets": []},
            "drug-unmapped": {
                "smiles": "CCCC",
                "targets": [
                    {
                        "gene_symbol": "UNKNOWN",
                        "uniprot_id": "PX",
                        "evidence": "predicted",
                        "score": 0.5,
                        "prediction_rank": 3,
                    }
                ],
            },
            "drug-outside": {
                "smiles": "CCCl",
                "targets": [
                    {
                        "gene_symbol": "GENEOUT",
                        "uniprot_id": "P9",
                        "evidence": "known",
                        "score": 0.9,
                    }
                ],
            },
        }
    targets_manifest = _commit_drug_targets(context, payload)
    return ppi, mapping, disease, targets_manifest


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _prepare(
    context: RunContext,
    ppi: Path,
    mapping: Path,
    disease: Path,
    targets_manifest: Path,
    *,
    attempt: int = 1,
    shared_cache_dir: Path | None = None,
):
    return proximity_prepare_network(
        context=context,
        ppi_path=ppi,
        disease_genes_path=disease,
        drug_targets_manifest_path=targets_manifest,
        target_mapping_path=mapping,
        settings=TEST_SETTINGS,
        config_hash=CONFIG_HASH,
        code_version="test",
        attempt=attempt,
        shared_cache_dir=shared_cache_dir,
    )


def test_prepare_lcc_mapping_lists_and_preserves_target_evidence(tmp_path: Path) -> None:
    context, prepared, resources = _context(tmp_path, "prepare")
    ppi, mapping, disease, targets_manifest = _write_inputs(context, prepared, resources)
    result = _prepare(context, ppi, mapping, disease, targets_manifest)

    assert result.status is NodeStatus.SUCCEEDED, result.to_json()
    assert result.node_id == "proximity_prepare_network"
    assert result.metrics["ppi_input_node_count"] == 10
    assert result.metrics["lcc_node_count"] == 8
    assert result.metrics["lcc_edge_count"] == 9
    assert result.metrics["disease_duplicate_count"] == 1
    assert result.metrics["disease_in_lcc_count"] == 2
    assert result.metrics["scorable_compound_count"] == 2
    assert result.metrics["skipped_compound_count"] == 3
    assert result.metrics["unmapped_target_symbol_count"] == 1
    assert result.metrics["cache_hit"] is False

    rows = _read_tsv(
        context.resolve_run_relative(
            "artifacts/proximity/prepared_drug_targets.tsv", must_exist=True
        )
    )
    assert [row["ID"] for row in rows] == ["drug-a", "drug-b"]
    target_records = json.loads(rows[0]["target_records"])
    assert [target["evidence"] for target in target_records] == [
        "known",
        "predicted",
    ]
    assert target_records[1]["prediction_rank"] == 1
    assert target_records[1]["score"] == 0.8
    skipped = _read_tsv(
        context.resolve_run_relative("artifacts/proximity/skipped_compounds.tsv", must_exist=True)
    )
    assert skipped == [
        {"ID": "drug-empty", "reason": "no_targets"},
        {"ID": "drug-outside", "reason": "no_targets_in_lcc"},
        {"ID": "drug-unmapped", "reason": "no_mapped_targets"},
    ]
    unmapped = _read_tsv(
        context.resolve_run_relative("artifacts/proximity/unmapped_targets.tsv", must_exist=True)
    )
    assert unmapped[0]["gene_symbol"] == "UNKNOWN"


def test_cache_hit_corruption_rebuild_and_input_invalidation(tmp_path: Path) -> None:
    context, prepared, resources = _context(tmp_path, "cache")
    ppi, mapping, disease, targets_manifest = _write_inputs(context, prepared, resources)
    first = _prepare(context, ppi, mapping, disease, targets_manifest)
    assert first.status is NodeStatus.SUCCEEDED
    first_key = first.metrics["cache_key"]

    second = _prepare(context, ppi, mapping, disease, targets_manifest, attempt=2)
    assert second.status is NodeStatus.SUCCEEDED
    assert second.metrics["cache_hit"] is True
    assert second.metrics["cache_source"] == "run"
    assert second.metrics["cache_key"] == first_key

    cache_path = next(
        context.resolve_run_relative("cache/proximity", must_exist=True).glob("*.npz")
    )
    cache_path.write_bytes(b"corrupt")
    rebuilt = _prepare(context, ppi, mapping, disease, targets_manifest, attempt=3)
    assert rebuilt.status is NodeStatus.SUCCEEDED, rebuilt.to_json()
    assert rebuilt.metrics["cache_hit"] is False
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache["random_distances"].shape[0] == TEST_SETTINGS.randomizations

    disease.write_text(
        "symbol\tentrez_id\nD1\t1\nD3\t3\n",
        encoding="utf-8",
    )
    invalidated = _prepare(context, ppi, mapping, disease, targets_manifest, attempt=4)
    assert invalidated.status is NodeStatus.SUCCEEDED
    assert invalidated.metrics["cache_hit"] is False
    assert invalidated.metrics["cache_key"] != first_key


def test_explicit_shared_cache_is_materialized_back_into_run_cache(
    tmp_path: Path,
) -> None:
    context, prepared, resources = _context(tmp_path, "shared-cache")
    ppi, mapping, disease, targets_manifest = _write_inputs(context, prepared, resources)
    shared_cache = tmp_path / "shared-proximity-cache"
    first = _prepare(
        context,
        ppi,
        mapping,
        disease,
        targets_manifest,
        shared_cache_dir=shared_cache,
    )
    assert first.status is NodeStatus.SUCCEEDED
    assert len(list(shared_cache.glob("*.npz"))) == 1
    for cache_file in context.resolve_run_relative("cache/proximity").glob("*.npz"):
        cache_file.unlink()

    second = _prepare(
        context,
        ppi,
        mapping,
        disease,
        targets_manifest,
        attempt=2,
        shared_cache_dir=shared_cache,
    )
    assert second.status is NodeStatus.SUCCEEDED
    assert second.metrics["cache_hit"] is True
    assert second.metrics["cache_source"] == "shared"
    assert len(list(context.resolve_run_relative("cache/proximity").glob("*.npz"))) == 1


def test_score_is_fixed_seed_reproducible_sorted_and_separate_node(tmp_path: Path) -> None:
    context, prepared, resources = _context(tmp_path, "score")
    ppi, mapping, disease, targets_manifest = _write_inputs(context, prepared, resources)
    prepared_result = _prepare(context, ppi, mapping, disease, targets_manifest)
    assert prepared_result.status is NodeStatus.SUCCEEDED
    preparation_runtime_manifest = _runtime_manifest(
        context, "proximity_prepare_network", prepared_result.outputs[0]
    )

    first = proximity_score_compounds(
        context=context,
        preparation_manifest_path=preparation_runtime_manifest,
        settings=TEST_SETTINGS,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    score_path = context.resolve_run_relative(
        "artifacts/proximity/proximity_scores.csv", must_exist=True
    )
    first_bytes = score_path.read_bytes()
    second = proximity_score_compounds(
        context=context,
        preparation_manifest_path=preparation_runtime_manifest,
        settings=TEST_SETTINGS,
        config_hash=CONFIG_HASH,
        code_version="test",
        attempt=2,
    )

    assert first.status is NodeStatus.SUCCEEDED, first.to_json()
    assert second.status is NodeStatus.SUCCEEDED, second.to_json()
    assert first.node_id == "proximity_score_compounds"
    expected_mode = "fork" if "fork" in mp.get_all_start_methods() else "serial"
    assert first.metrics["execution_mode"] == expected_mode
    assert first_bytes == score_path.read_bytes()
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["ID", "z"]
    z_values = [float(row["z"]) for row in rows]
    assert z_values == sorted(z_values)
    assert {row["ID"] for row in rows} == {"drug-a", "drug-b"}


def test_no_usable_targets_and_empty_disease_module_are_input_errors(
    tmp_path: Path,
) -> None:
    context, prepared, resources = _context(tmp_path, "errors")
    no_targets_payload = {
        "drug-empty": {"smiles": "CCO", "targets": []},
    }
    ppi, mapping, disease, targets_manifest = _write_inputs(
        context,
        prepared,
        resources,
        payload=no_targets_payload,
    )
    no_targets = _prepare(context, ppi, mapping, disease, targets_manifest)
    assert no_targets.status is NodeStatus.FAILED
    assert no_targets.error is not None
    assert no_targets.error.category == "input"
    assert "no compound" in no_targets.error.message

    usable_payload = {
        "drug-a": {
            "smiles": "CCO",
            "targets": [
                {
                    "gene_symbol": "GENE1",
                    "uniprot_id": "P1",
                    "evidence": "known",
                    "score": 1.0,
                }
            ],
        }
    }
    targets_manifest = _commit_drug_targets(context, usable_payload)
    disease.write_text(
        "symbol\tentrez_id\nOUT\t99\n",
        encoding="utf-8",
    )
    no_disease = _prepare(context, ppi, mapping, disease, targets_manifest, attempt=2)
    assert no_disease.status is NodeStatus.FAILED
    assert no_disease.error is not None
    assert no_disease.error.category == "input"
    assert "no normalized disease gene" in no_disease.error.message


def test_tampered_netinfer_artifact_is_rejected_via_manifest(tmp_path: Path) -> None:
    context, prepared, resources = _context(tmp_path, "manifest")
    ppi, mapping, disease, targets_manifest = _write_inputs(context, prepared, resources)
    context.resolve_run_relative(
        "artifacts/netinfer/drug_targets.json", must_exist=True
    ).write_text("{}", encoding="utf-8")
    result = _prepare(context, ppi, mapping, disease, targets_manifest)
    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category == "input"
    assert "invalid or stale" in result.error.message
