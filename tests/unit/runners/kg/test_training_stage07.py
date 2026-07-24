import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.kg._training_common import seed_configuration_hash
from lipid_screening_agent.runners.kg.aggregate_seeds import aggregate_rankings, kg_aggregate_seeds
from lipid_screening_agent.runners.kg.finetune_seed import _cache_matches, kg_finetune_seed
from lipid_screening_agent.runners.kg.prepare_training_data import (
    kg_prepare_training_data,
    prepare_frames,
)
from lipid_screening_agent.runners.kg.pretrain import kg_pretrain
from lipid_screening_agent.runtime import RunContext, sha256_file
from lipid_screening_agent.runtime.execution import execute_node

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)


def _context(tmp_path: Path, run_id: str, output_name: str) -> RunContext:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)
    return RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resources,
        input_dir=project / "runs" / run_id / "inputs/prepared",
        output_dir=project / "runs" / run_id / output_name,
    )


def _small_graph_frames():
    nodes = pd.DataFrame(
        [
            (10, "drug:1", "drug"),
            (12, "disease:custom", "disease"),
            (15, "drug:2", "drug"),
            (18, "disease:other", "disease"),
            (20, "drug:3", "drug"),
            (21, "drug:4", "drug"),
            (22, "drug:5", "drug"),
            (23, "drug:6", "drug"),
        ],
        columns=["node_index", "node_id", "node_type"],
    )
    rows = []
    for index, drug in enumerate(["drug:1", "drug:2", "drug:3", "drug:4", "drug:5", "drug:6"]):
        rows.append(
            (
                "drug",
                drug,
                nodes.set_index("node_id").loc[drug, "node_index"],
                "drug_disease",
                "disease",
                "disease:custom" if index == 0 else "disease:other",
                12 if index == 0 else 18,
            )
        )
    rows.append(rows[0])
    rows.extend(
        [
            ("disease", "disease:other", 18, "disease_disease", "disease", "disease:custom", 12),
            ("disease", "disease:custom", 12, "disease_disease", "disease", "disease:other", 18),
        ]
    )
    graph = pd.DataFrame(
        rows,
        columns=["x_type", "x_id", "x_index", "relation", "y_type", "y_id", "y_index"],
    )
    return nodes, graph


def test_prepare_deduplicates_adds_reverse_relations_and_pins_custom_positive():
    nodes, graph = _small_graph_frames()
    settings = replace(CONFIG.kg.training_data, seed=11, valid_fraction=0.2, test_fraction=0.2)
    typed, base, directed, metrics = prepare_frames(
        nodes,
        graph,
        settings=settings,
        custom_disease_id="disease:custom",
        numpy=np,
        pandas=pd,
    )

    assert typed.sort_values("node_index").node_index.tolist() == typed.node_index.tolist()
    assert typed[typed.node_type == "drug"].type_local_index.tolist() == list(range(6))
    assert len(base) == len(graph) - 2  # duplicate plus homogeneous reverse canonicalized
    pinned = base[(base.relation == "drug_disease") & (base.y_id == "disease:custom")]
    assert pinned.split.tolist() == ["train"]
    assert "rev_drug_disease" in set(directed.relation)
    assert metrics["pinned_edge_count"] == 1
    assert not directed.duplicated(["x_id", "relation", "y_id", "split"]).any()


def test_prepare_resolves_three_committed_stage06_artifact_manifests(tmp_path: Path):
    construction = _context(tmp_path, "manifest-handoff", "artifacts/kg/construction")
    nodes, graph = _small_graph_frames()
    node_path = construction.output_dir / "node.csv"
    graph_path = construction.output_dir / "kg.csv"
    scientific_path = construction.output_dir / "manifest.json"
    node_path.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(node_path, sep="\t", index=False)
    graph.to_csv(graph_path, index=False)
    scientific_path.write_text(
        json.dumps(
            {
                "disease": {"custom_node_id": "disease:custom"},
                "configuration": {"candidate_source_tag": "UserLibrary:"},
            }
        ),
        encoding="utf-8",
    )

    def commit(execution):
        execution.add_output("kg_nodes", node_path)
        execution.add_output("kg_graph", graph_path)
        execution.add_output("kg_construction_manifest", scientific_path)

    upstream = execute_node(
        commit,
        context=construction,
        node_id="kg_construct_graph",
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert upstream.status is NodeStatus.SUCCEEDED
    manifests = {}
    for path in (construction.run_dir / "artifacts/manifests/kg_construct_graph/main").glob(
        "*.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[payload["artifact_type"]] = path
    training = RunContext.create(
        runs_root=construction.runs_root,
        run_id=construction.run_id,
        project_root=construction.project_root,
        resource_dir=construction.resource_dir,
        input_dir=construction.input_dir,
        output_dir=construction.run_dir / "artifacts/kg/training_data",
        exist_ok=True,
    )
    result = kg_prepare_training_data(
        context=training,
        settings=replace(CONFIG.kg.training_data, valid_fraction=0.2, test_fraction=0.2),
        disease=replace(CONFIG.disease, custom_node_id="disease:custom"),
        config_hash=CONFIG_HASH,
        code_version="test",
        nodes_manifest_path=manifests["kg_nodes"],
        graph_manifest_path=manifests["kg_graph"],
        construction_manifest_path=manifests["kg_construction_manifest"],
    )
    assert result.status is NodeStatus.SUCCEEDED
    assert (training.output_dir / "node_typed.csv").is_file()
    assert (training.output_dir / "kg_directed.csv.gz").is_file()


def _ranking(seed: int, count: int = 5):
    order = list(range(count))
    order = order[seed % count :] + order[: seed % count]
    rows = []
    for rank, node_index in enumerate(order, 1):
        rows.append(
            {
                "node_id": f"drug:{node_index}",
                "node_name": f"Drug {node_index}",
                "source_ids": f"UserLibrary:C{node_index}",
                "compound_ids": json.dumps([f"C{node_index}"]),
                "rank": rank,
                "score": 1.0 - rank / 10.0 + seed / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_five_seed_aggregation_computes_complete_metrics():
    seeds = [5, 6, 7, 8, 9]
    frames = []
    for seed in seeds:
        frame = _ranking(seed)
        frame["seed"] = seed
        frames.append(frame)
    result = aggregate_rankings(frames, seeds=seeds, pandas=pd, numpy=np)

    assert len(result) == 5
    assert set(
        [
            "rank_mean",
            "rank_median",
            "rank_std",
            "rank_min",
            "rank_max",
            "score_mean",
            "score_std",
            "n_seeds",
            "top100_freq",
            "top200_freq",
        ]
    ) <= set(result.columns)
    assert (result.n_seeds == 5).all()
    assert (result.top100_freq == 1.0).all()
    assert result.rank_mean.is_monotonic_increasing


def _seed_dirs(tmp_path: Path, seeds, *, seed_hash="abc123"):
    output = {}
    for seed in seeds:
        directory = tmp_path / f"seed_{seed:03d}"
        directory.mkdir(parents=True)
        _ranking(seed).to_csv(directory / "custom_disease_drug_ranking.csv", index=False)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "status": "succeeded",
                    "seed": seed,
                    "seed_configuration_hash": seed_hash,
                    "best_epoch": 1,
                    "best_valid_metric": 0.5,
                    "training_seconds": 0.1,
                }
            ),
            encoding="utf-8",
        )
        output[seed] = directory
    return output


def test_aggregate_honors_modified_seed_configuration(tmp_path: Path):
    context = _context(tmp_path, "modified-seeds", "artifacts/kg/aggregate")
    finetune = replace(CONFIG.kg.finetune, seeds=(2, 4, 6))
    result = kg_aggregate_seeds(
        context=context,
        finetune_settings=finetune,
        aggregation_settings=CONFIG.kg.aggregation,
        config_hash=CONFIG_HASH,
        code_version="test",
        seed_result_dirs=_seed_dirs(tmp_path / "results", finetune.seeds),
    )
    assert result.status is NodeStatus.SUCCEEDED
    summary = json.loads(
        (context.output_dir / "abc123/ensemble_summary.json").read_text(encoding="utf-8")
    )
    assert summary["configured_seeds"] == [2, 4, 6]
    assert summary["rra"]["executed"] is False


def test_aggregate_returns_blocked_and_lists_missing_seed(tmp_path: Path):
    context = _context(tmp_path, "missing-seed", "artifacts/kg/aggregate")
    seeds = CONFIG.kg.finetune.seeds
    result = kg_aggregate_seeds(
        context=context,
        finetune_settings=CONFIG.kg.finetune,
        aggregation_settings=CONFIG.kg.aggregation,
        config_hash=CONFIG_HASH,
        code_version="test",
        seed_result_dirs=_seed_dirs(tmp_path / "results", seeds[:-1]),
    )
    assert result.status is NodeStatus.BLOCKED
    assert list(result.metrics["missing_seeds"]) == [9]
    assert not list(context.output_dir.rglob("kg_ensemble_ranking.csv"))


def test_duplicate_seed_configuration_fails_structurally(tmp_path: Path):
    context = _context(tmp_path, "duplicate-seed", "artifacts/kg/aggregate")
    finetune = replace(CONFIG.kg.finetune, seeds=(5, 5, 6))
    result = kg_aggregate_seeds(
        context=context,
        finetune_settings=finetune,
        aggregation_settings=CONFIG.kg.aggregation,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.FAILED
    assert result.error.category.value == "input"


def test_seed_cache_rejects_mismatched_manifest(tmp_path: Path):
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "ranking": "custom_disease_drug_ranking.csv",
            "history": "history.json",
            "summary": "summary.json",
            "seed_manifest": "manifest.json",
        }.items()
    }
    paths["ranking"].write_text("node_id,rank,score\ndrug:1,1,0.8\n", encoding="utf-8")
    paths["history"].write_text("[]", encoding="utf-8")
    paths["summary"].write_text("{}", encoding="utf-8")
    paths["seed_manifest"].write_text(
        json.dumps(
            {
                "cache_key": "old-key",
                "file_sha256": {
                    name: sha256_file(paths[name]) for name in ("ranking", "history", "summary")
                },
            }
        ),
        encoding="utf-8",
    )
    assert _cache_matches(paths, "old-key")
    assert not _cache_matches(paths, "new-key")


def test_finetune_seed_returns_cached_without_importing_dgl(tmp_path: Path):
    context = _context(tmp_path, "cached-seed", "artifacts/kg/finetune")
    upstream = context.run_dir / "artifacts/kg/upstream"
    upstream.mkdir(parents=True)
    nodes = upstream / "node_typed.csv"
    edges = upstream / "kg_directed.csv.gz"
    training_manifest = upstream / "training_manifest.json"
    checkpoint = upstream / "best_pretrain_model.pt"
    pretrain_config = upstream / "config.json"
    nodes.write_text("node_id,node_type,type_local_index\n", encoding="utf-8")
    pd.DataFrame(
        columns=["x_type", "relation", "y_type", "x_idx", "y_idx", "x_id", "y_id", "split"]
    ).to_csv(edges, index=False, compression="gzip")
    training_manifest.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"cache-only-checkpoint")
    pretrain_config.write_text(
        json.dumps({"checkpoint_format": "txgnn_minimal.HeteroRGCN.state_dict.v1"}),
        encoding="utf-8",
    )
    seed_hash = seed_configuration_hash(
        seeds=CONFIG.kg.finetune.seeds,
        finetune_config=asdict(CONFIG.kg.finetune),
        checkpoint_sha256=sha256_file(checkpoint),
        training_manifest_sha256=sha256_file(training_manifest),
    )
    directory = context.output_dir / seed_hash / "seed_005"
    directory.mkdir(parents=True)
    paths = {
        "ranking": directory / "custom_disease_drug_ranking.csv",
        "history": directory / "history.json",
        "summary": directory / "summary.json",
        "seed_manifest": directory / "manifest.json",
    }
    paths["ranking"].write_text("node_id,rank,score\ndrug:1,1,0.8\n", encoding="utf-8")
    paths["history"].write_text("[]", encoding="utf-8")
    paths["summary"].write_text('{"status":"succeeded","seed":5}', encoding="utf-8")
    cache_key = json.dumps(
        {"seed": 5, "seed_configuration_hash": seed_hash, "code_version": "test"},
        sort_keys=True,
        separators=(",", ":"),
    )
    paths["seed_manifest"].write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "file_sha256": {
                    name: sha256_file(paths[name]) for name in ("ranking", "history", "summary")
                },
            }
        ),
        encoding="utf-8",
    )
    result = kg_finetune_seed(
        context=context,
        settings=CONFIG.kg.finetune,
        seed=5,
        config_hash=CONFIG_HASH,
        code_version="test",
        nodes_path=nodes,
        edges_path=edges,
        training_manifest_path=training_manifest,
        checkpoint_path=checkpoint,
        pretrain_config_path=pretrain_config,
    )
    assert result.status is NodeStatus.CACHED
    assert result.metrics["seed"] == 5


def test_pretrain_missing_dgl_is_a_structured_environment_error(tmp_path: Path):
    try:
        __import__("dgl")
    except ImportError:
        pass
    else:
        pytest.skip("test targets the missing-DGL environment boundary")
    context = _context(tmp_path, "missing-dgl", "artifacts/kg/pretrain")
    upstream = context.run_dir / "artifacts/kg/training_data"
    upstream.mkdir(parents=True)
    nodes = upstream / "node_typed.csv"
    edges = upstream / "kg_directed.csv.gz"
    manifest = upstream / "manifest.json"
    nodes.write_text("node_id,node_type,type_local_index\n", encoding="utf-8")
    pd.DataFrame(
        columns=["x_type", "relation", "y_type", "x_idx", "y_idx", "x_id", "y_id", "split"]
    ).to_csv(edges, index=False, compression="gzip")
    manifest.write_text("{}", encoding="utf-8")
    result = kg_pretrain(
        context=context,
        settings=CONFIG.kg.pretrain,
        config_hash=CONFIG_HASH,
        code_version="test",
        nodes_path=nodes,
        edges_path=edges,
        training_manifest_path=manifest,
        allow_cpu_small_graph=True,
    )
    assert result.status is NodeStatus.FAILED
    assert result.error.category.value == "environment"
    assert "dgl" in result.error.details["missing_or_incompatible"]


def test_model_initializes_on_a_small_heterograph():
    dgl = pytest.importorskip("dgl")
    pytest.importorskip("torch")
    from lipid_screening_agent.runners.kg.model import HeteroRGCN, initialize_node_embedding

    graph = dgl.heterograph(
        {
            ("drug", "drug_disease", "disease"): ([0, 1], [0, 1]),
            ("disease", "rev_drug_disease", "drug"): ([0, 1], [0, 1]),
        },
        num_nodes_dict={"drug": 2, "disease": 2},
    )
    initialize_node_embedding(graph, 8)
    model = HeteroRGCN(graph, 8, 8, 8, attention=False, proto=False)
    assert model.w_rels.shape == (2, 8)
    assert set(model.state_dict()) >= {"w_rels"}
