import pytest

from lipid_screening_agent.runners.netinfer.algorithms import (
    merge_compound_targets,
    parse_raw_prediction_rows,
)
from lipid_screening_agent.runtime import InputError


def test_raw_output_is_headerless_and_exactly_six_columns() -> None:
    with pytest.raises(InputError, match="six tab-separated fields"):
        parse_raw_prediction_rows(
            [["source_type", "source_id", "target_type"]],
            source_label="fixture",
        )


def test_merge_is_known_first_ranked_and_gene_symbol_deduplicated() -> None:
    predictions = parse_raw_prediction_rows(
        [
            ["DRUG", "D1", "TARGET", "P2", "0.8", "1"],
            ["DRUG", "D1", "TARGET", "P1", "1.0", "-"],
            ["DRUG", "D1", "TARGET", "P1", "0.7", "2"],
            ["COMPOUND", "novel", "TARGET", "P4", "0.4", "2"],
            ["COMPOUND", "novel", "TARGET", "P3", "0.5", "1"],
            ["COMPOUND", "novel", "TARGET", "P5", "0.3", "11"],
        ],
        source_label="fixture",
    )
    mapping = [
        {
            "ID": "known-user",
            "SMILES": "CCO",
            "match_key": "CCO",
            "netinfer_input_type": "DRUG",
            "netinfer_input_id": "D1",
        },
        {
            "ID": "novel",
            "SMILES": "C1CC1",
            "match_key": "C1CC1",
            "netinfer_input_type": "COMPOUND",
            "netinfer_input_id": "novel",
        },
    ]

    targets, missing, metrics = merge_compound_targets(
        mapping,
        predictions,
        {"P1": "GENE1", "P2": "GENE2", "P3": "GENE3", "P4": "GENE4"},
        top_n_predicted=10,
    )

    known_targets = targets["known-user"]["targets"]
    assert [item["gene_symbol"] for item in known_targets] == ["GENE1", "GENE2"]
    assert [item["evidence"] for item in known_targets] == ["known", "predicted"]
    novel_targets = targets["novel"]["targets"]
    assert [item["gene_symbol"] for item in novel_targets] == ["GENE3", "GENE4"]
    assert [item["prediction_rank"] for item in novel_targets] == [1, 2]
    assert missing == []
    assert metrics["known_target_count"] == 1
    assert metrics["predicted_target_count"] == 3


def test_merge_reports_compounds_without_raw_rows() -> None:
    mapping = [
        {
            "ID": "empty",
            "SMILES": "CCO",
            "match_key": "CCO",
            "netinfer_input_type": "COMPOUND",
            "netinfer_input_id": "empty",
        }
    ]

    targets, missing, metrics = merge_compound_targets(
        mapping, (), {"P1": "GENE1"}, top_n_predicted=10
    )

    assert targets == {"empty": {"smiles": "CCO", "targets": []}}
    assert missing == [("empty", "no_prediction_rows")]
    assert metrics["missing_prediction_count"] == 1
