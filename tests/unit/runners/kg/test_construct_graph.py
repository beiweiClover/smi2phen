import builtins
import csv
import json
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.kg.construct_graph import kg_construct_graph
from lipid_screening_agent.runtime import RunContext, atomic_write_json
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

pytest.importorskip("rdkit", reason="KG construction unit tests require the optional RDKit extra")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = PROJECT_ROOT / "tests/fixtures/kg_base"
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)


class _FakeRow:
    def __init__(self, bits: tuple[int, ...]) -> None:
        self._bits = bits

    def nonzero(self):
        return (list(self._bits),)


class _FakePubChemFingerprint:
    def transform(self, smiles):
        return [_FakeRow((0, 1)) for _ in smiles]


def _context(tmp_path: Path, run_id: str) -> tuple[RunContext, Path]:
    project = tmp_path / "project"
    prepared = project / "runs" / run_id / "inputs/prepared"
    output = project / "runs" / run_id / "artifacts/kg/construction"
    project.mkdir(parents=True)
    context = RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=FIXTURE,
        input_dir=prepared,
        output_dir=output,
    )
    return context, prepared


def _write_inputs(prepared: Path, *, duplicate: bool = False) -> dict[str, Path]:
    prepared.mkdir(parents=True, exist_ok=True)
    compounds = prepared / "compounds.normalized.csv"
    compounds.write_text(
        "ID,SMILES,Name,CAS,Formula,MolWt\n"
        "matched,CCO,Submitted ethanol,,,\n"
        "new,C1CC1,Novel,,,\n"
        "invalid,not-a-smiles,Broken,,,\n" + ("new,CCN,Duplicate,,,\n" if duplicate else ""),
        encoding="utf-8",
    )
    genes = prepared / "disease_genes.normalized.tsv"
    genes.write_text("symbol\tentrez_id\nGENE1\t101\nGENE3\t303\n", encoding="utf-8")
    targets = prepared / "drug_targets.json"
    targets.write_text(
        json.dumps(
            {
                "matched": {
                    "smiles": "CCO",
                    "targets": [
                        {"gene_symbol": "GENE1", "uniprot_id": "P1", "evidence": "known"},
                        {
                            "gene_symbol": "GENE2",
                            "uniprot_id": "P2",
                            "evidence": "predicted",
                            "prediction_rank": 2,
                            "score": 0.7,
                        },
                        {
                            "gene_symbol": "UNKNOWN",
                            "uniprot_id": "PX",
                            "evidence": "predicted",
                            "prediction_rank": 1,
                            "score": 0.8,
                        },
                    ],
                },
                "new": {
                    "smiles": "C1CC1",
                    "targets": [
                        {"gene_symbol": "GENE3", "uniprot_id": "P3", "evidence": "known"},
                        {
                            "gene_symbol": "GENE2",
                            "uniprot_id": "P2",
                            "evidence": "predicted",
                            "prediction_rank": 1,
                            "score": 0.9,
                        },
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    positive = prepared / "positive_drugs.tsv"
    positive.write_text(
        "input_type\tvalue\nlibrary_id\tnew\nbase_drug_name\tEthanol\nlibrary_id\tnew\n",
        encoding="utf-8",
    )
    links = prepared / "disease_links.tsv"
    links.write_text(
        "input_type\tvalue\nbase_disease_name\tLipid disorder\n",
        encoding="utf-8",
    )
    return {
        "compounds": compounds,
        "genes": genes,
        "targets": targets,
        "positive": positive,
        "links": links,
    }


def _run(context: RunContext, paths: dict[str, Path], *, optional: bool = True):
    return kg_construct_graph(
        context=context,
        compounds_path=paths["compounds"],
        disease_genes_path=paths["genes"],
        drug_targets_path=paths["targets"],
        base_nodes_path=FIXTURE / "node.csv",
        base_edges_path=FIXTURE / "edges.csv",
        base_manifest_path=FIXTURE / "manifest.json",
        base_drug_smiles_path=FIXTURE / "base_drug_smiles.tsv",
        target_mapping_path=FIXTURE / "target_map.tsv",
        positive_drugs_path=paths["positive"] if optional else None,
        disease_links_path=paths["links"] if optional else None,
        settings=CONFIG.kg.construction,
        disease=CONFIG.disease,
        config_hash=CONFIG_HASH,
        code_version="test",
        fingerprint_factory=lambda workers: _FakePubChemFingerprint(),
    )


def _commit_drug_targets(context: RunContext, source: Path) -> Path:
    target = context.resolve_run_relative("artifacts/netinfer/drug_targets.json")

    def operation(execution: NodeExecution) -> None:
        atomic_write_json(
            target,
            json.loads(source.read_text(encoding="utf-8")),
            allowed_root=context.run_dir,
        )
        execution.add_output("drug_targets", target)

    result = execute_node(
        operation,
        context=context,
        node_id="netinfer_merge_targets",
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED
    return context.resolve_run_relative(
        f"artifacts/manifests/netinfer_merge_targets/main/{result.outputs[0]}.json",
        must_exist=True,
    )


def _rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def test_constructs_new_and_matched_drugs_optional_edges_reports_and_dedup(tmp_path: Path) -> None:
    context, prepared = _context(tmp_path, "complete")
    result = _run(context, _write_inputs(prepared))
    assert result.status is NodeStatus.SUCCEEDED
    assert len(result.outputs) == 7

    output = context.output_dir
    nodes = _rows(output / "node.csv", delimiter="\t")
    edges = _rows(output / "edges.csv")
    graph = _rows(output / "kg.csv")
    report = _rows(output / "drug_smiles_match_report.tsv", delimiter="\t")
    invalid = _rows(output / "invalid_smiles.tsv", delimiter="\t")
    unmapped = _rows(output / "unmapped_target_symbols.tsv", delimiter="\t")

    assert [row["node_id"] for row in nodes].count("mol:base:ethanol") == 1
    matched = next(row for row in report if row["library_id"] == "matched")
    novel = next(row for row in report if row["library_id"] == "new")
    assert matched["match_type"] == "base_canonical_smiles"
    assert novel["output_node_id"].startswith("mol:user:")
    assert next(row for row in report if row["library_id"] == "invalid")["match_type"] == (
        "invalid_smiles_kept_new"
    )
    assert invalid == [{"ID": "invalid", "SMILES": "not-a-smiles", "reason": "invalid_smiles"}]
    assert unmapped[0]["gene_symbol"] == "UNKNOWN"

    triples = [(row["x_id"], row["relation"], row["y_id"]) for row in edges]
    assert len(triples) == len(set(triples))
    assert triples.count(("protein:ncbi:101", "disease_protein", "disease:base:lipid")) == 1
    assert (novel["output_node_id"], "drug_disease", CONFIG.disease.custom_node_id) in triples
    assert (
        CONFIG.disease.custom_node_id,
        "disease_disease",
        "disease:base:lipid",
    ) in triples
    assert any(
        row["x_id"] == novel["output_node_id"] and row["relation"] == "drug_substructure"
        for row in edges
    )
    assert any(
        row["source_db"] == "NetInfer" and row["original_relation"] == "NetInfer_top5_known_first"
        for row in edges
    )
    assert len(graph) == len(edges)
    assert result.metrics["num_edges_deduplicated"] >= 2

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "C:\\" not in serialized
    assert "/data/" not in serialized
    assert manifest["resources"]["base_graph"]["resource_id"] == "kg-base:unit-fixture-v1"
    assert manifest["disease"]["custom_node_id"] == CONFIG.disease.custom_node_id
    assert manifest["configuration"]["netinfer_dti_top_n"] == 5


def test_optional_inputs_may_be_absent(tmp_path: Path) -> None:
    context, prepared = _context(tmp_path, "optional-absent")
    result = _run(context, _write_inputs(prepared), optional=False)
    assert result.status is NodeStatus.SUCCEEDED
    manifest = json.loads((context.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["positive_drugs"] is None
    assert manifest["inputs"]["disease_links"] is None
    assert manifest["optional_inputs"] == {"positive_drugs": [], "disease_links": []}


def test_committed_stage04_drug_targets_manifest_is_consumed(tmp_path: Path) -> None:
    context, prepared = _context(tmp_path, "committed-targets")
    paths = _write_inputs(prepared)
    target_manifest = _commit_drug_targets(context, paths["targets"])
    result = kg_construct_graph(
        context=context,
        compounds_path=paths["compounds"],
        disease_genes_path=paths["genes"],
        drug_targets_manifest_path=target_manifest,
        base_nodes_path=FIXTURE / "node.csv",
        base_edges_path=FIXTURE / "edges.csv",
        base_manifest_path=FIXTURE / "manifest.json",
        base_drug_smiles_path=FIXTURE / "base_drug_smiles.tsv",
        target_mapping_path=FIXTURE / "target_map.tsv",
        settings=CONFIG.kg.construction,
        disease=CONFIG.disease,
        config_hash=CONFIG_HASH,
        code_version="test",
        fingerprint_factory=lambda workers: _FakePubChemFingerprint(),
    )
    assert result.status is NodeStatus.SUCCEEDED
    manifest = json.loads((context.output_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_manifest = json.loads(target_manifest.read_text(encoding="utf-8"))
    assert manifest["inputs"]["drug_targets"]["artifact_id"] == runtime_manifest["artifact_id"]


def test_duplicate_compound_id_is_a_structured_input_failure(tmp_path: Path) -> None:
    context, prepared = _context(tmp_path, "duplicate")
    result = _run(context, _write_inputs(prepared, duplicate=True))
    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert "duplicate" in result.error.message.casefold()


def test_missing_scikit_fingerprints_is_environment_error(tmp_path: Path, monkeypatch) -> None:
    context, prepared = _context(tmp_path, "missing-skfp")
    paths = _write_inputs(prepared)
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "skfp.fingerprints" or name.startswith("skfp"):
            raise ModuleNotFoundError("injected missing scikit-fingerprints")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    result = kg_construct_graph(
        context=context,
        compounds_path=paths["compounds"],
        disease_genes_path=paths["genes"],
        drug_targets_path=paths["targets"],
        base_nodes_path=FIXTURE / "node.csv",
        base_edges_path=FIXTURE / "edges.csv",
        base_manifest_path=FIXTURE / "manifest.json",
        base_drug_smiles_path=FIXTURE / "base_drug_smiles.tsv",
        target_mapping_path=FIXTURE / "target_map.tsv",
        settings=CONFIG.kg.construction,
        disease=CONFIG.disease,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "environment"
    assert "scikit-fingerprints" in result.error.message
