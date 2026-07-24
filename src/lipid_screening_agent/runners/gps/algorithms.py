"""Pure data/model-adjacent algorithms migrated from the three legacy GPS notebooks.

The functions in this module do not know about run directories, manifests, CLI arguments, or
workflow branching. Scientific runners provide already validated data and commit the results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lipid_screening_agent.runtime import InputError


def gps_prob_to_change(probability: Any, threshold: float, *, np: Any) -> Any:
    """Apply the legacy three-class argmax and confidence threshold."""

    max_probability = probability.max(axis=1)
    prediction = probability.argmax(axis=1).astype(np.int8) - 1
    prediction[max_probability < threshold] = 0
    return prediction.astype(np.int8)


def classify_deg(
    log2fc: Any,
    adjusted_p: Any,
    pass_expression_filter: Any,
    *,
    fdr_cutoff: float,
    absolute_log2fc_cutoff: float,
    np: Any,
) -> Any:
    """Classify up/down/not-significant genes with the frozen DEG thresholds."""

    regulation = np.full(len(log2fc), "not_sig", dtype=object)
    up = pass_expression_filter & (adjusted_p <= fdr_cutoff) & (log2fc >= absolute_log2fc_cutoff)
    down = pass_expression_filter & (adjusted_p <= fdr_cutoff) & (log2fc <= -absolute_log2fc_cutoff)
    regulation[up] = "up"
    regulation[down] = "down"
    return regulation


def bh_adjust_for_filtered(
    p_values: Any,
    pass_filter: Any,
    *,
    np: Any,
    multipletests: Callable[..., Any],
) -> Any:
    """Run Benjamini-Hochberg only on genes passing the expression filter."""

    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    if int(pass_filter.sum()) > 0:
        adjusted[pass_filter] = multipletests(p_values[pass_filter], method="fdr_bh")[1]
    return adjusted


def compute_deg_table(
    *,
    gene_ids: Sequence[str],
    tpm_matrix: Any,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    comparison_name: str,
    tpm_filename: str,
    metadata_filename: str,
    fdr_cutoff: float,
    absolute_log2fc_cutoff: float,
    tpm_filter_cutoff: float,
    minimum_group_fraction_expressed: float,
    np: Any,
    pd: Any,
    stats: Any,
    multipletests: Callable[..., Any],
) -> Any:
    """Compute one comparison using log2(TPM+1), Welch t-test, and filtered BH."""

    matrix = np.asarray(tpm_matrix, dtype=np.float32)
    group = np.asarray(groups, dtype=object)
    if matrix.ndim != 2 or matrix.shape != (len(gene_ids), len(sample_ids)):
        raise InputError(
            "TPM matrix dimensions do not match GeneID/sample metadata",
            details={
                "matrix_shape": list(matrix.shape),
                "gene_count": len(gene_ids),
                "sample_count": len(sample_ids),
            },
        )
    if np.isnan(matrix).any():
        raise InputError("TPM matrix contains nonnumeric or missing values")
    control_mask = group == "control"
    disease_mask = group == "disease"
    if int(control_mask.sum()) == 0 or int(disease_mask.sum()) == 0:
        raise InputError(f"{comparison_name} has no control or disease samples")

    logged = np.log2(matrix + 1.0)
    control_log = logged[:, control_mask]
    disease_log = logged[:, disease_mask]
    control_tpm = matrix[:, control_mask]
    disease_tpm = matrix[:, disease_mask]

    log2fc = disease_log.mean(axis=1) - control_log.mean(axis=1)
    expressed_control = (control_tpm >= tpm_filter_cutoff).mean(axis=1)
    expressed_disease = (disease_tpm >= tpm_filter_cutoff).mean(axis=1)
    pass_expression_filter = (expressed_control >= minimum_group_fraction_expressed) | (
        expressed_disease >= minimum_group_fraction_expressed
    )

    test = stats.ttest_ind(
        disease_log,
        control_log,
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )
    p_values = np.asarray(test.pvalue, dtype=np.float64)
    p_values[~np.isfinite(p_values)] = 1.0
    adjusted = bh_adjust_for_filtered(
        p_values,
        pass_expression_filter,
        np=np,
        multipletests=multipletests,
    )
    regulation = classify_deg(
        log2fc,
        adjusted,
        pass_expression_filter,
        fdr_cutoff=fdr_cutoff,
        absolute_log2fc_cutoff=absolute_log2fc_cutoff,
        np=np,
    )
    return pd.DataFrame(
        {
            "GeneID": list(gene_ids),
            "comparison": comparison_name,
            "tpm_file": tpm_filename,
            "metadata_file": metadata_filename,
            "n_disease": int(disease_mask.sum()),
            "n_control": int(control_mask.sum()),
            "log2FC_disease_vs_control": log2fc,
            "pvalue_welch_ttest": p_values,
            "padj_BH": adjusted,
            "pass_expression_filter": pass_expression_filter,
            "regulation": regulation,
        }
    )


def combine_direction_consistent_degs(results: Sequence[Any], *, pd: Any) -> Any:
    """Intersect significant DEG sets and retain genes with one shared direction."""

    if not results:
        raise InputError("at least one DEG comparison is required")
    significant_tables: list[Any] = []
    for result in results:
        if result.empty:
            significant = result.copy()
            comparison = "unknown"
        else:
            comparison = str(result["comparison"].iloc[0])
            significant = result[result["regulation"].isin(["up", "down"])].copy()
        significant_tables.append(
            significant[["GeneID", "log2FC_disease_vs_control", "padj_BH", "regulation"]].rename(
                columns={
                    "log2FC_disease_vs_control": f"log2FC_{comparison}",
                    "padj_BH": f"padj_BH_{comparison}",
                    "regulation": f"regulation_{comparison}",
                }
            )
        )

    core = significant_tables[0].copy()
    for table in significant_tables[1:]:
        core = pd.merge(core, table, on="GeneID", how="inner")
    if core.empty:
        raise InputError("no significant DEG intersection was produced")

    regulation_columns = [column for column in core.columns if column.startswith("regulation_")]
    log2fc_columns = [column for column in core.columns if column.startswith("log2FC_")]
    adjusted_columns = [column for column in core.columns if column.startswith("padj_BH_")]
    if len(regulation_columns) > 1:
        core = core[core[regulation_columns].nunique(axis=1).eq(1)].copy()
        if core.empty:
            raise InputError("multiple comparisons have no direction-consistent DEG intersection")

    core["regulation"] = core[regulation_columns[0]]
    core["disease_log2FC_mean"] = core[log2fc_columns].mean(axis=1)
    core["mean_abs_log2FC"] = core[log2fc_columns].abs().mean(axis=1)
    core["max_padj_BH"] = core[adjusted_columns].max(axis=1)
    core["n_comparisons"] = len(significant_tables)
    if len(adjusted_columns) == 2:
        sort_columns = [
            "regulation",
            "mean_abs_log2FC",
            adjusted_columns[0],
            adjusted_columns[1],
        ]
        ascending = [True, False, True, True]
    else:
        sort_columns = ["regulation", "mean_abs_log2FC", "max_padj_BH"]
        ascending = [True, False, True]
    return core.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def convert_drug_profile_index_to_entrez(
    drug_profile: Any,
    symbol_to_entrez: Mapping[str, str],
    *,
    np: Any,
    pd: Any,
) -> tuple[Any, dict[str, int], tuple[str, ...]]:
    """Return a new Entrez-indexed profile without mutating the symbol-indexed input."""

    source = drug_profile.copy()
    source.index = source.index.astype(str).str.strip()
    numeric_fraction = float(pd.Series(source.index.astype(str)).str.fullmatch(r"\d+").mean())
    unmapped: tuple[str, ...] = ()
    mapped_count = len(source)
    if numeric_fraction > 0.95:
        converted = source.copy()
    else:
        mapped = pd.Series(source.index, index=source.index).map(symbol_to_entrez)
        mask = mapped.notna()
        unmapped = tuple(source.index[~mask].astype(str).tolist())
        converted = source.loc[mask].copy()
        converted.index = mapped[mask].astype(str).to_numpy()
        mapped_count = int(mask.sum())
    duplicate_count = int(converted.index.duplicated(keep=False).sum())
    converted = converted[~converted.index.duplicated(keep="first")].copy()
    if converted.empty:
        raise InputError("Drug_GPS contains no genes that map to Entrez GeneID")
    converted.index = converted.index.astype(str)
    converted.index.name = "GeneID"
    converted = converted.astype(np.int8)
    return (
        converted,
        {
            "input_gene_count": len(source),
            "mapped_gene_count": mapped_count,
            "unmapped_gene_count": len(unmapped),
            "duplicate_entrez_row_count": duplicate_count,
            "entrez_gene_count": len(converted),
        },
        unmapped,
    )


def build_disease_profile(core_degs: Any, drug_gene_ids: Sequence[str], *, np: Any) -> Any:
    """Restrict direction-consistent DEGs to the drug-profile gene space."""

    gene_set = {str(gene).strip() for gene in drug_gene_ids}
    disease = core_degs[["GeneID", "disease_log2FC_mean", "regulation"]].copy()
    disease["GeneID"] = disease["GeneID"].astype(str).str.strip()
    disease = disease[disease["GeneID"].isin(gene_set)].copy()
    disease = disease.drop_duplicates("GeneID", keep="first")
    if disease.empty:
        raise InputError(
            "direction-consistent DEGs do not intersect the Entrez Drug_GPS gene space"
        )
    disease["disease_direction"] = np.where(
        disease["disease_log2FC_mean"] > 0,
        "up",
        "down",
    )
    return (
        disease[["GeneID", "disease_log2FC_mean", "disease_direction"]]
        .sort_values(
            ["disease_log2FC_mean", "GeneID"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )


def gsea_es_from_positions(hit_positions: Any, n_genes: int, *, np: Any) -> float:
    """Compute the legacy unweighted GSEA enrichment score from 0-based positions."""

    positions = np.sort(np.asarray(hit_positions, dtype=np.int64))
    n_hits = positions.size
    if n_hits == 0 or n_hits == n_genes:
        return 0.0
    hit_rank = np.arange(1, n_hits + 1, dtype=np.float64)
    miss_before_hit = positions - (hit_rank - 1)
    running_before_hit = (hit_rank - 1) / n_hits - miss_before_hit / (n_genes - n_hits)
    miss_after_hit = (positions + 1) - hit_rank
    running_after_hit = hit_rank / n_hits - miss_after_hit / (n_genes - n_hits)
    maximum = float(max(0.0, np.max(running_after_hit)))
    minimum = float(min(0.0, np.min(running_before_hit)))
    return maximum if abs(maximum) >= abs(minimum) else minimum


def random_es_distribution(
    n_genes: int,
    set_size: int,
    n_random: int,
    rng: Any,
    *,
    np: Any,
) -> Any:
    """Generate the legacy size-matched random GSEA background."""

    if set_size == 0 or set_size == n_genes:
        return np.zeros(n_random, dtype=float)
    output = np.empty(n_random, dtype=float)
    for index in range(n_random):
        positions = rng.choice(n_genes, size=set_size, replace=False)
        output[index] = gsea_es_from_positions(positions, n_genes, np=np)
    return output


def z_transform_from_stats(value: float, mean: float, sd: float, *, np: Any) -> float:
    """Apply the legacy zero fallback for degenerate random backgrounds."""

    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float((value - mean) / sd)


def score_compound_profiles(
    drug_profile: Any,
    disease_profile: Any,
    *,
    random_background_samples: int,
    seed: int,
    np: Any,
    pd: Any,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[Any, dict[str, int]]:
    """Compute deterministic zRGES-like lower-is-better scores for every compound."""

    if random_background_samples <= 0:
        raise InputError("random_background_samples must be positive")
    disease = disease_profile[["GeneID", "disease_log2FC_mean"]].copy()
    disease["GeneID"] = disease["GeneID"].astype(str).str.strip()
    disease["disease_log2FC_mean"] = pd.to_numeric(disease["disease_log2FC_mean"], errors="coerce")
    disease = disease.dropna(subset=["GeneID", "disease_log2FC_mean"])
    disease = disease[disease["GeneID"].ne("") & disease["disease_log2FC_mean"].ne(0)]
    disease = disease.drop_duplicates("GeneID", keep="first")
    if disease.empty:
        raise InputError("Disease_GPS contains no nonzero disease genes")

    drugs = drug_profile.copy()
    drugs.index = drugs.index.astype(str).str.strip()
    drugs = drugs[~drugs.index.duplicated(keep="first")]
    signature = disease[disease["GeneID"].isin(drugs.index)].copy()
    signature = signature.sort_values("GeneID").reset_index(drop=True)
    genes = signature["GeneID"].tolist()
    if not genes:
        raise InputError("Disease_GPS and Drug_GPS have no GeneID intersection")

    disease_rank = signature.sort_values("disease_log2FC_mean")["GeneID"].tolist()
    rank_position = {gene: position for position, gene in enumerate(disease_rank)}
    subset = drugs.loc[genes]
    drug_ids = subset.columns.astype(str).to_numpy()
    try:
        matrix = subset.to_numpy(dtype=np.int8).T
    except (TypeError, ValueError) as exc:
        raise InputError("Drug_GPS values must be numeric -1/0/1") from exc
    unique_values = set(np.unique(matrix).astype(int).tolist())
    if not unique_values <= {-1, 0, 1}:
        raise InputError(
            "Drug_GPS contains values outside -1/0/1",
            details={"observed_values": sorted(unique_values)},
        )

    gene_positions = np.asarray([rank_position[gene] for gene in genes], dtype=np.int64)
    n_ranked = len(genes)
    unique_sizes = sorted(
        set(np.sum(matrix == 1, axis=1).astype(int)) | set(np.sum(matrix == -1, axis=1).astype(int))
    )
    rng = np.random.default_rng(seed)
    random_stats: dict[int, tuple[float, float]] = {}
    for index, size in enumerate(unique_sizes, start=1):
        distribution = random_es_distribution(
            n_ranked,
            int(size),
            random_background_samples,
            rng,
            np=np,
        )
        random_stats[int(size)] = (
            float(np.mean(distribution)),
            float(np.std(distribution, ddof=1)),
        )
        if progress is not None:
            progress("random_background", index, len(unique_sizes))

    rows: list[dict[str, Any]] = []
    for index, drug_id in enumerate(drug_ids, start=1):
        effect = matrix[index - 1]
        up_positions = gene_positions[effect > 0]
        down_positions = gene_positions[effect < 0]
        es_up = gsea_es_from_positions(up_positions, n_ranked, np=np)
        es_down = gsea_es_from_positions(down_positions, n_ranked, np=np)
        up_mean, up_sd = random_stats[len(up_positions)]
        down_mean, down_sd = random_stats[len(down_positions)]
        z_up = z_transform_from_stats(es_up, up_mean, up_sd, np=np)
        z_down = z_transform_from_stats(es_down, down_mean, down_sd, np=np)
        rows.append(
            {
                "ID": str(drug_id),
                "GPS_score_zRGES_like_lower_better": -(z_up - z_down),
            }
        )
        if progress is not None:
            progress("compound_scoring", index, len(drug_ids))

    result = (
        pd.DataFrame(rows)
        .sort_values(
            ["GPS_score_zRGES_like_lower_better", "ID"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )
    return result, {
        "drug_profile_gene_count": len(drugs),
        "disease_gene_count": len(disease),
        "aligned_gene_count": len(genes),
        "disease_up_gene_count": int((signature["disease_log2FC_mean"] > 0).sum()),
        "disease_down_gene_count": int((signature["disease_log2FC_mean"] < 0).sum()),
        "compound_count": len(drug_ids),
        "random_set_size_count": len(unique_sizes),
    }


__all__ = [
    "bh_adjust_for_filtered",
    "build_disease_profile",
    "classify_deg",
    "combine_direction_consistent_degs",
    "compute_deg_table",
    "convert_drug_profile_index_to_entrez",
    "gps_prob_to_change",
    "gsea_es_from_positions",
    "random_es_distribution",
    "score_compound_profiles",
    "z_transform_from_stats",
]
