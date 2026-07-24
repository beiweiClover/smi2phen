import numpy as np
import pandas as pd
import pytest

from lipid_screening_agent.runners.gps.algorithms import (
    classify_deg,
    combine_direction_consistent_degs,
    convert_drug_profile_index_to_entrez,
    gsea_es_from_positions,
    score_compound_profiles,
)


def _comparison(name: str, rows: list[tuple[str, float, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GeneID": gene,
                "comparison": name,
                "log2FC_disease_vs_control": log2fc,
                "padj_BH": adjusted,
                "regulation": regulation,
            }
            for gene, log2fc, adjusted, regulation in rows
        ]
    )


def test_deg_thresholds_and_direction_consistent_intersection() -> None:
    regulation = classify_deg(
        np.asarray([0.5, 0.5001, -0.5, -0.7, 2.0]),
        np.asarray([0.05, 0.05, 0.05, 0.01, 0.001]),
        np.asarray([True, True, True, True, False]),
        fdr_cutoff=0.05,
        absolute_log2fc_cutoff=0.5,
        np=np,
    )
    assert regulation.tolist() == ["up", "up", "down", "down", "not_sig"]

    first = _comparison(
        "comparison_1",
        [("1", 1.0, 0.01, "up"), ("2", -1.2, 0.02, "down")],
    )
    second = _comparison(
        "comparison_2",
        [("1", 2.0, 0.03, "up"), ("2", 1.3, 0.04, "up")],
    )
    combined = combine_direction_consistent_degs([first, second], pd=pd)

    assert combined["GeneID"].tolist() == ["1"]
    assert combined["regulation"].tolist() == ["up"]
    assert combined["disease_log2FC_mean"].tolist() == [pytest.approx(1.5)]
    assert combined["n_comparisons"].tolist() == [2]


def test_drug_profile_id_alignment_is_nonmutating_and_first_row_wins() -> None:
    source = pd.DataFrame(
        {"cmp-a": [1, -1, 0], "cmp-b": [0, 1, -1]},
        index=pd.Index(["GENEA", "ALIASA", "UNKNOWN"], name="GeneSymbol"),
    )
    original = source.copy(deep=True)

    converted, metrics, unmapped = convert_drug_profile_index_to_entrez(
        source,
        {"GENEA": "1", "ALIASA": "1"},
        np=np,
        pd=pd,
    )

    pd.testing.assert_frame_equal(source, original)
    assert converted.index.name == "GeneID"
    assert converted.index.tolist() == ["1"]
    assert converted.loc["1", "cmp-a"] == 1
    assert metrics == {
        "input_gene_count": 3,
        "mapped_gene_count": 2,
        "unmapped_gene_count": 1,
        "duplicate_entrez_row_count": 2,
        "entrez_gene_count": 1,
    }
    assert unmapped == ("UNKNOWN",)


def test_gsea_and_compound_scores_are_deterministic_and_schema_stable() -> None:
    assert gsea_es_from_positions([0, 1], 4, np=np) == pytest.approx(1.0)
    drug = pd.DataFrame(
        {"cmp-a": [1, -1, 0], "cmp-b": [-1, 1, 1]},
        index=pd.Index(["1", "2", "3"], name="GeneID"),
    )
    disease = pd.DataFrame(
        {
            "GeneID": ["1", "2", "3"],
            "disease_log2FC_mean": [-2.0, 1.0, 3.0],
            "disease_direction": ["down", "up", "up"],
        }
    )

    first, metrics = score_compound_profiles(
        drug,
        disease,
        random_background_samples=50,
        seed=42,
        np=np,
        pd=pd,
    )
    second, _ = score_compound_profiles(
        drug,
        disease,
        random_background_samples=50,
        seed=42,
        np=np,
        pd=pd,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == ["ID", "GPS_score_zRGES_like_lower_better"]
    assert set(first["ID"]) == {"cmp-a", "cmp-b"}
    assert first["ID"].is_unique
    assert metrics["aligned_gene_count"] == 3
    assert metrics["compound_count"] == 2


def test_gps_modules_import_without_importing_heavy_dependencies(monkeypatch) -> None:
    import importlib
    import sys

    for module_name in [
        "lipid_screening_agent.runners.gps",
        "lipid_screening_agent.runners.gps.predict_drug_profiles",
        "lipid_screening_agent.runners.gps.build_disease_signature",
        "lipid_screening_agent.runners.gps.score_compounds",
    ]:
        sys.modules.pop(module_name, None)
    for dependency in ("torch", "rdkit", "scipy", "statsmodels"):
        monkeypatch.delitem(sys.modules, dependency, raising=False)

    imported = importlib.import_module("lipid_screening_agent.runners.gps")

    assert callable(imported.gps_predict_drug_profiles)
    assert "torch" not in sys.modules
    assert "rdkit" not in sys.modules
