import csv
import json
from dataclasses import replace
from pathlib import Path

from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.ranking.generate_run_report import generate_run_report
from lipid_screening_agent.runners.ranking.rank_candidates import (
    FINAL_COLUMNS,
    compute_candidate_ranking,
    rank_candidates,
    rank_percentiles,
)
from lipid_screening_agent.runtime import RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)


def _context(tmp_path: Path, run_id: str, output: str = "artifacts/final") -> RunContext:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    return RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resources,
        input_dir=project / "runs" / run_id / "inputs/prepared",
        output_dir=project / "runs" / run_id / output,
        exist_ok=True,
    )


def _write_csv(path: Path, fields, rows, *, delimiter=",") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _inputs(context: RunContext, *, source_ids=False, empty=False, duplicate_proximity=False):
    root = context.run_dir / "artifacts/upstream"
    kg_fields = ["node_id", "node_name", "rank_mean", "rank_median", "rank_std", "score_mean"]
    mapping_field = "source_ids" if source_ids else "compound_ids"
    kg_fields.append(mapping_field)
    kg_rows = [
        {
            "node_id": "drug:a",
            "node_name": "Alpha",
            "rank_mean": 1,
            "rank_median": 1,
            "rank_std": 0.1,
            "score_mean": 0.9,
            mapping_field: "UserLibrary:A" if source_ids else '["A"]',
        },
        {
            "node_id": "drug:b",
            "node_name": "Beta",
            "rank_mean": 2,
            "rank_median": 2,
            "rank_std": 0.1,
            "score_mean": 0.8,
            mapping_field: "UserLibrary:B;DrugBank:DB1" if source_ids else '["B"]',
        },
        {
            "node_id": "drug:c",
            "node_name": "Gamma",
            "rank_mean": 3,
            "rank_median": 3,
            "rank_std": 0.1,
            "score_mean": 0.7,
            mapping_field: "UserLibrary:C" if source_ids else '["C"]',
        },
    ]
    proximity_rows = [
        {"ID": "A", "z": 0.2 if empty else -1.0},
        {"ID": "B", "z": 0.4 if empty else -3.0},
        {"ID": "C", "z": 0.5},
    ]
    if duplicate_proximity:
        proximity_rows.append({"ID": "A", "z": -2.0})
    gps_rows = [
        {"ID": "A", "GPS_score_zRGES_like_lower_better": -4.0},
        {"ID": "B", "GPS_score_zRGES_like_lower_better": -1.0},
        {"ID": "C", "GPS_score_zRGES_like_lower_better": 0.2},
    ]
    return (
        _write_csv(root / "kg.csv", kg_fields, kg_rows),
        _write_csv(root / "proximity.csv", ["ID", "z"], proximity_rows),
        _write_csv(
            root / "gps.csv",
            ["ID", "GPS_score_zRGES_like_lower_better"],
            gps_rows,
        ),
    )


def _read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_two_evidence_hand_calculation_and_output_contract(tmp_path: Path):
    context = _context(tmp_path, "core")
    kg, proximity, _ = _inputs(context)
    result = rank_candidates(
        context=context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status=NodeStatus.SKIPPED,
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED, result.to_json()
    rows = _read_tsv(context.output_dir / "final_candidates.tsv")
    assert list(rows[0]) == list(FINAL_COLUMNS)
    assert [row["compound_id"] for row in rows] == ["A", "B"]
    assert [int(row["final_rank"]) for row in rows] == [1, 2]
    assert all(row["evidence_mode"] == "kg_proximity" for row in rows)
    assert all(int(row["evidence_count"]) == 2 for row in rows)
    assert [float(row["consensus_rank_percentile_mean"]) for row in rows] == [0.75, 0.75]
    assert all(row["gps_score"] == "" for row in rows)
    summary = json.loads((context.output_dir / "ranking_summary.json").read_text())
    assert summary["stage_counts"]["kg_top_n_unique_compounds"] == 3
    assert summary["stage_counts"]["proximity_threshold_pass"] == 2
    assert summary["stage_counts"]["final_candidates"] == 2
    assert summary["thresholds"]["gps"]["applied"] is False


def test_three_evidence_hand_calculation(tmp_path: Path):
    context = _context(tmp_path, "enhanced")
    kg, proximity, gps = _inputs(context)
    result = rank_candidates(
        context=context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_scores_path=gps,
        gps_status=NodeStatus.SUCCEEDED,
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED
    rows = _read_tsv(context.output_dir / "final_candidates.tsv")
    assert [row["compound_id"] for row in rows] == ["A", "B"]
    assert float(rows[0]["consensus_rank_percentile_mean"]) == 2 / 3
    assert float(rows[1]["consensus_rank_percentile_mean"]) == 5 / 6
    assert [float(row["gps_score"]) for row in rows] == [-4.0, -1.0]


def test_kg_consensus_rank_preserves_deterministic_ensemble_tie_breakers():
    kg_rows = [
        {
            "node_id": "node:z",
            "node_name": "Better tie break",
            "compound_ids": '["Z"]',
            "rank_mean": "1",
            "rank_median": "1",
            "rank_std": "0.1",
            "score_mean": "0.9",
        },
        {
            "node_id": "node:a",
            "node_name": "Worse tie break",
            "compound_ids": '["A"]',
            "rank_mean": "1",
            "rank_median": "2",
            "rank_std": "0.1",
            "score_mean": "0.9",
        },
    ]
    proximity_rows = [{"ID": "A", "z": "-1"}, {"ID": "Z", "z": "-1"}]
    gps_rows = [
        {"ID": "A", "GPS_score_zRGES_like_lower_better": "-1"},
        {"ID": "Z", "GPS_score_zRGES_like_lower_better": "-1"},
    ]

    candidates, summary = compute_candidate_ranking(
        kg_rows=kg_rows,
        kg_fields=tuple(kg_rows[0]),
        proximity_rows=proximity_rows,
        proximity_fields=("ID", "z"),
        gps_rows=gps_rows,
        gps_fields=("ID", "GPS_score_zRGES_like_lower_better"),
        settings=CONFIG.ranking,
    )

    assert [row["compound_id"] for row in candidates] == ["Z", "A"]
    assert candidates[0]["consensus_rank_percentile_mean"] == 0.5
    assert candidates[1]["consensus_rank_percentile_mean"] == 2 / 3
    assert summary["component_rank_sources"]["kg"] == (
        "deterministic_kg_top_n_selection_position"
    )


def test_gps_failed_blocks_without_automatic_downgrade(tmp_path: Path):
    context = _context(tmp_path, "gps-failed")
    kg, proximity, _ = _inputs(context)
    result = rank_candidates(
        context=context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status=NodeStatus.FAILED,
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.BLOCKED
    assert result.metrics["gps_upstream_status"] == "failed"
    assert not (context.output_dir / "final_candidates.tsv").exists()


def test_source_id_mapping_and_duplicate_kg_mapping_keep_best_node(tmp_path: Path):
    context = _context(tmp_path, "source-map")
    kg, proximity, _ = _inputs(context, source_ids=True)
    rows = list(csv.DictReader(kg.open(encoding="utf-8")))
    duplicate = dict(rows[-1])
    duplicate["node_id"] = "drug:a-worse"
    duplicate["node_name"] = "Wrong Alpha"
    duplicate["rank_mean"] = "10"
    duplicate["source_ids"] = "UserLibrary:A"
    _write_csv(kg, rows[0].keys(), [*rows, duplicate])
    result = rank_candidates(
        context=context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status="skipped",
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED
    final = _read_tsv(context.output_dir / "final_candidates.tsv")
    assert final[0]["compound_id"] == "A"
    assert final[0]["compound_name"] == "Alpha"
    summary = json.loads((context.output_dir / "ranking_summary.json").read_text())
    assert summary["stage_counts"]["kg_duplicate_compound_mappings_resolved"] == 1


def test_duplicate_score_id_missing_column_and_non_numeric_are_structured_errors(
    tmp_path: Path,
):
    duplicate_context = _context(tmp_path, "duplicate")
    kg, proximity, _ = _inputs(duplicate_context, duplicate_proximity=True)
    duplicate = rank_candidates(
        context=duplicate_context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status="skipped",
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert duplicate.status is NodeStatus.FAILED
    assert duplicate.error.category.value == "input"

    missing_context = _context(tmp_path, "missing-column")
    kg, proximity, _ = _inputs(missing_context)
    _write_csv(proximity, ["ID", "wrong"], [{"ID": "A", "wrong": -1}])
    missing = rank_candidates(
        context=missing_context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status="skipped",
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert missing.status is NodeStatus.FAILED
    assert "missing required columns" in missing.error.message

    numeric_context = _context(tmp_path, "non-numeric")
    kg, proximity, _ = _inputs(numeric_context)
    _write_csv(proximity, ["ID", "z"], [{"ID": "A", "z": "bad"}])
    numeric = rank_candidates(
        context=numeric_context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status="skipped",
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert numeric.status is NodeStatus.FAILED
    assert "non-numeric" in numeric.error.message


def test_empty_intersection_writes_header_and_normal_summary(tmp_path: Path):
    context = _context(tmp_path, "empty")
    kg, proximity, _ = _inputs(context, empty=True)
    result = rank_candidates(
        context=context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status="skipped",
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED
    assert _read_tsv(context.output_dir / "final_candidates.tsv") == []
    assert (context.output_dir / "final_candidates.tsv").read_text().splitlines()[0] == "\t".join(
        FINAL_COLUMNS
    )
    summary = json.loads((context.output_dir / "ranking_summary.json").read_text())
    assert summary["status"] == "no_candidates_passed"
    assert summary["stage_counts"]["final_candidates"] == 0
    assert summary["thresholds"]["auto_relax_thresholds"] is False


def test_tied_min_rank_percentiles_and_explicit_descending_direction():
    assert rank_percentiles({"B": 1.0, "A": 1.0, "C": 2.0}, direction="ascending") == {
        "A": 1 / 3,
        "B": 1 / 3,
        "C": 1.0,
    }
    assert rank_percentiles({"A": 1.0, "B": 3.0, "C": 3.0}, direction="descending") == {
        "B": 1 / 3,
        "C": 1 / 3,
        "A": 1.0,
    }


def test_evidence_directions_are_explicit_and_drive_selection_and_ranking():
    assert CONFIG.ranking.kg.direction == "ascending"
    assert CONFIG.ranking.proximity.direction == "ascending"
    assert CONFIG.ranking.gps.direction == "ascending"
    descending = replace(
        CONFIG.ranking,
        kg=replace(CONFIG.ranking.kg, top_n=2, direction="descending"),
        proximity=replace(
            CONFIG.ranking.proximity,
            direction="descending",
            filter=replace(CONFIG.ranking.proximity.filter, operator="gt", value=0.0),
        ),
    )
    kg_rows = [
        {
            "node_id": "n-a",
            "node_name": "A",
            "compound_ids": '["A"]',
            "rank_mean": "1",
            "score_mean": "0.8",
        },
        {
            "node_id": "n-b",
            "node_name": "B",
            "compound_ids": '["B"]',
            "rank_mean": "2",
            "score_mean": "0.9",
        },
    ]
    candidates, summary = compute_candidate_ranking(
        kg_rows=kg_rows,
        kg_fields=tuple(kg_rows[0]),
        proximity_rows=[{"ID": "A", "z": "1"}, {"ID": "B", "z": "3"}],
        proximity_fields=("ID", "z"),
        settings=descending,
    )
    assert [row["compound_id"] for row in candidates] == ["B", "A"]
    assert summary["thresholds"]["kg"]["direction"] == "descending"
    assert summary["thresholds"]["proximity"]["direction"] == "descending"


def test_configured_top_n_is_not_automatically_changed():
    settings = replace(CONFIG.ranking, kg=replace(CONFIG.ranking.kg, top_n=1))
    kg_rows = [
        {
            "node_id": "n1",
            "node_name": "One",
            "compound_ids": '["A"]',
            "rank_mean": "1",
            "score_mean": "0.9",
        },
        {
            "node_id": "n2",
            "node_name": "Two",
            "compound_ids": '["B"]',
            "rank_mean": "2",
            "score_mean": "0.8",
        },
    ]
    candidates, summary = compute_candidate_ranking(
        kg_rows=kg_rows,
        kg_fields=tuple(kg_rows[0]),
        proximity_rows=[{"ID": "A", "z": "1"}, {"ID": "B", "z": "-1"}],
        proximity_fields=("ID", "z"),
        settings=settings,
    )
    assert candidates == []
    assert summary["thresholds"]["kg"]["top_n"] == 1
    assert summary["status"] == "no_candidates_passed"


def test_run_report_summarizes_nodes_skips_environment_and_outputs(tmp_path: Path):
    ranking_context = _context(tmp_path, "report")
    kg, proximity, _ = _inputs(ranking_context)
    ranked = rank_candidates(
        context=ranking_context,
        kg_ranking_path=kg,
        proximity_scores_path=proximity,
        gps_status="skipped",
        settings=CONFIG.ranking,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert ranked.status is NodeStatus.SUCCEEDED
    run_manifest = ranking_context.run_dir / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "run_id": "report",
                "input_sources": [{"name": "library.csv", "source": "user_upload"}],
                "planning": {
                    "skipped_nodes": [
                        {
                            "node_id": "netinfer_predict_known",
                            "task_id": "main",
                            "reason": "provided targets",
                        }
                    ]
                },
                "toxicity": {"must_not_appear": True},
            }
        ),
        encoding="utf-8",
    )
    input_manifest = ranking_context.run_dir / "inputs/input_manifest.json"
    input_manifest.parent.mkdir(parents=True, exist_ok=True)
    input_manifest.write_text(
        json.dumps({"inputs": [{"input_key": "compound_library", "source": "user_upload"}]}),
        encoding="utf-8",
    )
    skipped = ranking_context.run_dir / "inputs/prepared/invalid_smiles.tsv"
    skipped.parent.mkdir(parents=True, exist_ok=True)
    skipped.write_text("ID\tSMILES\treason\nbad\t?\tparse_error\n", encoding="utf-8")

    report_context = _context(tmp_path, "report", output="reports")
    result = generate_run_report(
        context=report_context,
        config=CONFIG,
        run_manifest_path=run_manifest,
        input_manifest_path=input_manifest,
        ranking_summary_path=ranking_context.output_dir / "ranking_summary.json",
        final_candidates_path=ranking_context.output_dir / "final_candidates.tsv",
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert result.status is NodeStatus.SUCCEEDED, result.to_json()
    report = json.loads((report_context.output_dir / "run_report.json").read_text())
    assert report["workflow"]["evidence_mode"] == "kg_proximity"
    assert report["candidate_count"] == 2
    assert report["node_status_counts"]["succeeded"] == 1
    assert report["skips"]["artifact_record_total"] == 1
    assert report["skips"]["node_records"] == [
        {
            "node_id": "netinfer_predict_known",
            "reason": "provided targets",
            "source": "plan",
            "task_id": "main",
        }
    ]
    assert report["environment"]["host"]["python_version"]
    assert "toxicity" not in json.dumps(report).lower()
    markdown = (report_context.output_dir / "run_report.md").read_text(encoding="utf-8")
    assert "不构成临床疗效" in markdown
    assert "Skipped nodes (planned or executed): 1" in markdown
    assert "artifacts/final/final_candidates.tsv" in markdown
