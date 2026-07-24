"""PyTorch/SciPy implementation of the wSDTNBI predictor used by NetInfer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from lipid_screening_agent.runtime import EnvironmentError, InputError


@dataclass(frozen=True, slots=True)
class WSDTNBIConfig:
    alpha: float = 0.4
    beta: float = 0.2
    gamma: float = -0.5
    delta: int = 20
    epsilon: int = 4
    k: int = 2
    top_n: int = 10
    batch_size: int = 256
    device: str = "auto"
    dtype: str = "float32"


def _ordered_unique(values: Sequence[str] | pd.Series) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _read_edge_file(
    path: Path,
    *,
    columns: tuple[str, str, str, str, str],
    expected_source_type: str,
    expected_target_type: str,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=list(columns),
            dtype=str,
            keep_default_na=False,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise InputError(
            "NetInfer network could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if frame.empty:
        raise InputError("NetInfer network is empty", details={"path": str(path)})
    identity_columns = columns[:4]
    for column in identity_columns:
        frame[column] = frame[column].str.strip()
    if frame[list(identity_columns)].eq("").any(axis=None):
        raise InputError(
            "NetInfer network contains an empty identity field",
            details={"path": str(path)},
        )
    observed_sources = sorted(set(frame[columns[0]]) - {expected_source_type})
    observed_targets = sorted(set(frame[columns[2]]) - {expected_target_type})
    if observed_sources or observed_targets:
        raise InputError(
            "NetInfer network has unexpected node types",
            details={
                "path": str(path),
                "unexpected_source_types": observed_sources[:20],
                "unexpected_target_types": observed_targets[:20],
            },
        )
    weights = pd.to_numeric(frame[columns[4]], errors="coerce")
    invalid = weights.isna() | ~np.isfinite(weights) | (weights < 0)
    if invalid.any():
        raise InputError(
            "NetInfer network contains a non-finite or negative weight",
            details={"path": str(path), "first_invalid_row": int(np.flatnonzero(invalid)[0] + 1)},
        )
    frame[columns[4]] = weights.astype(float)
    return frame


def _to_torch_sparse(
    matrix: sparse.spmatrix,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    values = coo.data.astype(numpy_dtype, copy=False)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=coo.shape,
        device=device,
        dtype=dtype,
    ).coalesce()


class WSDTNBIEngine:
    """In-process replacement for the legacy CUDA 10.1 NetInfer executable."""

    def __init__(self, dt_file: Path, ds_file: Path, config: WSDTNBIConfig) -> None:
        self.config = config
        self._validate_config()
        self.device = self._select_device(config.device)
        self.dtype = torch.float64 if config.dtype == "float64" else torch.float32

        self.dt = _read_edge_file(
            Path(dt_file),
            columns=("source_type", "drug", "target_type", "target", "weight"),
            expected_source_type="DRUG",
            expected_target_type="TARGET",
        ).drop_duplicates(["drug", "target"], keep="last")
        self.ds = _read_edge_file(
            Path(ds_file),
            columns=("source_type", "drug", "sub_type", "sub", "weight"),
            expected_source_type="DRUG",
            expected_target_type="SUB",
        ).drop_duplicates(["drug", "sub"], keep="last")

        self.drugs = _ordered_unique(self.dt["drug"])
        self.targets = _ordered_unique(self.dt["target"])
        self.substructures = _ordered_unique(self.ds["sub"])
        if not self.drugs or not self.targets or not self.substructures:
            raise InputError("NetInfer network does not contain all required node types")

        self.drug_to_index = {value: index for index, value in enumerate(self.drugs)}
        self.target_to_index = {value: index for index, value in enumerate(self.targets)}
        self.sub_to_index = {
            value: index for index, value in enumerate(self.substructures)
        }
        self.n_drugs = len(self.drugs)
        self.n_targets = len(self.targets)
        self.n_substructures = len(self.substructures)
        self.target_offset = self.n_drugs + self.n_substructures
        self.n_nodes = self.target_offset + self.n_targets

        self._build_input_matrices()
        self._build_transition_matrix()
        self._build_target_lookup()

    def _validate_config(self) -> None:
        config = self.config
        if config.device not in {"auto", "cpu", "cuda"}:
            raise InputError("NetInfer device must be auto, cpu, or cuda")
        if config.dtype not in {"float32", "float64"}:
            raise InputError("NetInfer dtype must be float32 or float64")
        if (
            not 0 <= config.alpha <= 1
            or not 0 <= config.beta <= 1
            or config.delta < 0
            or config.epsilon < 1
            or config.k < 1
            or config.top_n < 1
            or config.batch_size < 1
        ):
            raise InputError("NetInfer wSDTNBI parameters are outside their valid ranges")
        if not all(
            math.isfinite(float(value))
            for value in (config.alpha, config.beta, config.gamma, config.delta)
        ):
            raise InputError("NetInfer wSDTNBI parameters must be finite")

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise EnvironmentError("NetInfer requested CUDA but no CUDA device is available")
        return torch.device(requested)

    def _build_input_matrices(self) -> None:
        dt_rows = self.dt["drug"].map(self.drug_to_index).to_numpy()
        dt_columns = self.dt["target"].map(self.target_to_index).to_numpy()
        dt_weights = self.dt["weight"].to_numpy(dtype=float)
        self.dt_weighted = sparse.csr_matrix(
            (dt_weights, (dt_rows, dt_columns)),
            shape=(self.n_drugs, self.n_targets),
        )
        self.dt_binary = sparse.csr_matrix(
            (np.ones_like(dt_weights), (dt_rows, dt_columns)),
            shape=(self.n_drugs, self.n_targets),
        )
        self.dt_binary.data[:] = 1.0

        ds_rows = self.ds["drug"].map(self.drug_to_index)
        valid = ds_rows.notna()
        valid_rows = ds_rows[valid].astype(int).to_numpy()
        valid_columns = (
            self.ds.loc[valid, "sub"].map(self.sub_to_index).astype(int).to_numpy()
        )
        self.ds_binary = sparse.csr_matrix(
            (np.ones(len(valid_rows)), (valid_rows, valid_columns)),
            shape=(self.n_drugs, self.n_substructures),
        )
        self.ds_binary.data[:] = 1.0
        self.official_sub_degree = np.asarray(self.ds_binary.sum(axis=1)).ravel()
        self.official_target_degree = np.asarray(self.dt_binary.sum(axis=1)).ravel()
        self.ds_torch = _to_torch_sparse(
            self.ds_binary, device=self.device, dtype=self.dtype
        )
        self.official_sub_degree_torch = torch.as_tensor(
            self.official_sub_degree, device=self.device, dtype=self.dtype
        )

    def _build_transition_matrix(self) -> None:
        has_dti = self.official_target_degree > 0
        drug_sub = self.ds_binary.multiply(has_dti[:, None]).tocoo()
        drug_target = self.dt_binary.multiply(has_dti[:, None]).tocoo()
        beta = self.config.beta

        row = np.concatenate(
            [
                drug_sub.row,
                self.n_drugs + drug_sub.col,
                drug_target.row,
                self.target_offset + drug_target.col,
            ]
        ).astype(np.int64, copy=False)
        column = np.concatenate(
            [
                self.n_drugs + drug_sub.col,
                drug_sub.row,
                self.target_offset + drug_target.col,
                drug_target.row,
            ]
        ).astype(np.int64, copy=False)
        values = np.concatenate(
            [
                beta * drug_sub.data,
                beta * drug_sub.data,
                (1.0 - beta) * drug_target.data,
                (1.0 - beta) * drug_target.data,
            ]
        )
        adjacency = sparse.coo_matrix(
            (values, (row, column)), shape=(self.n_nodes, self.n_nodes)
        ).tocsr()
        column_sum = np.asarray(adjacency.sum(axis=0)).ravel()
        scale = np.zeros_like(column_sum)
        nonzero = column_sum > 0
        scale[nonzero] = column_sum[nonzero] ** self.config.gamma
        coo = adjacency.tocoo()
        adjusted = sparse.coo_matrix(
            (coo.data * scale[coo.col], (coo.row, coo.col)),
            shape=adjacency.shape,
        ).tocsr()
        row_sum = np.asarray(adjusted.sum(axis=1)).ravel()
        coo = adjusted.tocoo()
        transition = sparse.coo_matrix(
            (coo.data / row_sum[coo.row], (coo.row, coo.col)),
            shape=adjusted.shape,
        ).tocsr()
        self.transition_transpose = _to_torch_sparse(
            transition.T.tocsr(), device=self.device, dtype=self.dtype
        )

    def _build_target_lookup(self) -> None:
        known_indices: list[torch.Tensor] = []
        known_weights: list[torch.Tensor] = []
        binary = self.dt_binary.tocsc()
        weighted = self.dt_weighted.tocsc()
        for target_index in range(self.n_targets):
            start, end = binary.indptr[target_index : target_index + 2]
            known = binary.indices[start:end]
            w_start, w_end = weighted.indptr[target_index : target_index + 2]
            weight_map = dict(
                zip(
                    weighted.indices[w_start:w_end].tolist(),
                    weighted.data[w_start:w_end].tolist(),
                )
            )
            weights = np.array([weight_map[int(index)] for index in known])
            known_indices.append(
                torch.as_tensor(known, device=self.device, dtype=torch.long)
            )
            known_weights.append(
                torch.as_tensor(weights, device=self.device, dtype=self.dtype)
            )
        self.target_known_indices = known_indices
        self.target_known_weights = known_weights
        self.known_targets_by_drug: dict[str, list[tuple[str, float]]] = {}
        for drug, group in self.dt.groupby("drug", sort=False):
            ordered = group.sort_values("weight", ascending=False, kind="stable")
            self.known_targets_by_drug[str(drug)] = list(
                zip(ordered["target"], ordered["weight"].astype(float))
            )

    def _resources(
        self,
        substructures: sparse.csr_matrix,
        total_sub_degree: np.ndarray,
        known_targets: sparse.csr_matrix | None,
    ) -> tuple[torch.Tensor, np.ndarray, sparse.csr_matrix | None]:
        size = substructures.shape[0]
        resources = torch.zeros(
            (size, self.n_nodes), device=self.device, dtype=self.dtype
        )
        for row_index in range(size):
            start, end = substructures.indptr[row_index : row_index + 2]
            columns = substructures.indices[start:end]
            if len(columns) and total_sub_degree[row_index] > 0:
                indices = torch.as_tensor(
                    columns, device=self.device, dtype=torch.long
                )
                resources[row_index, self.n_drugs + indices] = (
                    self.config.alpha / total_sub_degree[row_index]
                )
        if known_targets is None:
            return resources, np.zeros(size), None
        known_counts = np.asarray(known_targets.sum(axis=1)).ravel()
        for row_index in range(size):
            start, end = known_targets.indptr[row_index : row_index + 2]
            columns = known_targets.indices[start:end]
            if len(columns) and known_counts[row_index] > 0:
                indices = torch.as_tensor(
                    columns, device=self.device, dtype=torch.long
                )
                resources[row_index, self.target_offset + indices] = (
                    (1.0 - self.config.alpha) / known_counts[row_index]
                )
        return resources, known_counts, known_targets

    def _propagate(self, resources: torch.Tensor) -> torch.Tensor:
        values = resources
        for _ in range(self.config.k):
            values = torch.sparse.mm(self.transition_transpose, values.T).T
        return values[:, self.target_offset :]

    def _normalize(self, scores: torch.Tensor, known_counts: np.ndarray) -> torch.Tensor:
        normalized = torch.zeros_like(scores)
        for row_index in range(scores.shape[0]):
            threshold_index = min(
                int(self.config.delta + known_counts[row_index] + 1),
                self.n_targets,
            )
            order = torch.argsort(scores[row_index], descending=True)
            top = order[:threshold_index]
            if top.numel() == 0:
                continue
            threshold = scores[row_index, top[-1]]
            normalized[row_index, top] = 1.0
            if torch.abs(threshold).item() > 0:
                rest = order[threshold_index:]
                normalized[row_index, rest] = scores[row_index, rest] / threshold
        return normalized

    def _similarity_scores(
        self, substructures: sparse.csr_matrix, total_sub_degree: np.ndarray
    ) -> torch.Tensor:
        query = torch.as_tensor(
            substructures.toarray(), device=self.device, dtype=self.dtype
        )
        intersection = torch.sparse.mm(self.ds_torch, query.T).T
        query_degree = torch.as_tensor(
            total_sub_degree, device=self.device, dtype=self.dtype
        ).view(-1, 1)
        union = query_degree + self.official_sub_degree_torch.view(1, -1) - intersection
        similarity = torch.where(
            union > 0, intersection / union, torch.zeros_like(intersection)
        )
        result = torch.zeros(
            (substructures.shape[0], self.n_targets),
            device=self.device,
            dtype=self.dtype,
        )
        for target_index, known in enumerate(self.target_known_indices):
            if known.numel() == 0:
                continue
            values = similarity.index_select(1, known)
            count = min(self.config.epsilon, values.shape[1])
            if values.shape[1] > count:
                selected = torch.argsort(values, dim=1)[:, -count:]
            else:
                selected = (
                    torch.arange(values.shape[1], device=self.device)
                    .view(1, -1)
                    .expand(values.shape[0], -1)
                )
            result[:, target_index] = self.target_known_weights[target_index][
                selected
            ].mean(dim=1)
        return result

    def _predict(
        self,
        query_ids: Sequence[str],
        substructures: sparse.csr_matrix,
        total_sub_degree: np.ndarray,
        known_targets: sparse.csr_matrix | None,
    ) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}
        with torch.no_grad():
            for batch_start in range(0, len(query_ids), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(query_ids))
                batch_sub = substructures[batch_start:batch_end].tocsr()
                batch_degree = total_sub_degree[batch_start:batch_end]
                batch_known = (
                    None
                    if known_targets is None
                    else known_targets[batch_start:batch_end].tocsr()
                )
                resources, known_counts, batch_known = self._resources(
                    batch_sub, batch_degree, batch_known
                )
                scores = self._normalize(
                    self._propagate(resources), known_counts
                ) * self._similarity_scores(batch_sub, batch_degree)
                for local_index, query_id in enumerate(
                    query_ids[batch_start:batch_end]
                ):
                    row = scores[local_index].clone()
                    # The legacy formula assigns a target-weight average even when every
                    # query substructure is absent from DS. Do not turn that zero-overlap
                    # case into arbitrary positive predictions.
                    if batch_sub.indptr[local_index] == batch_sub.indptr[local_index + 1]:
                        row.zero_()
                    if batch_known is not None:
                        start, end = batch_known.indptr[local_index : local_index + 2]
                        known_columns = batch_known.indices[start:end]
                        if len(known_columns):
                            row[
                                torch.as_tensor(
                                    known_columns,
                                    device=self.device,
                                    dtype=torch.long,
                                )
                            ] = -torch.inf
                    order = torch.argsort(row, descending=True)
                    predictions: list[dict[str, Any]] = []
                    for target_index in order.detach().cpu().tolist():
                        score = float(row[target_index].item())
                        if not math.isfinite(score) or score <= 0:
                            continue
                        predictions.append(
                            {
                                "target": self.targets[target_index],
                                "score": score,
                                "rank": len(predictions) + 1,
                            }
                        )
                        if len(predictions) == self.config.top_n:
                            break
                    results[str(query_id)] = predictions
        return results

    def predict_official_drugs(
        self, drug_ids: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        valid = [str(value) for value in drug_ids if str(value) in self.drug_to_index]
        if not valid:
            return {}
        indices = np.array([self.drug_to_index[value] for value in valid])
        return self._predict(
            valid,
            self.ds_binary[indices].tocsr(),
            self.official_sub_degree[indices],
            self.dt_binary[indices].tocsr(),
        )

    def predict_compounds(
        self, compound_ids: Sequence[str], cs_file: Path
    ) -> dict[str, list[dict[str, Any]]]:
        compound_ids = [str(value) for value in compound_ids]
        compound_to_index = {
            value: index for index, value in enumerate(compound_ids)
        }
        cs = _read_edge_file(
            Path(cs_file),
            columns=("source_type", "compound", "sub_type", "sub", "weight"),
            expected_source_type="COMPOUND",
            expected_target_type="SUB",
        )
        cs = cs[cs["compound"].isin(compound_to_index)].drop_duplicates(
            ["compound", "sub"], keep="last"
        )
        total_degree = np.zeros(len(compound_ids))
        for compound, degree in cs.groupby("compound", sort=False)["sub"].nunique().items():
            total_degree[compound_to_index[str(compound)]] = float(degree)
        represented = cs[cs["sub"].isin(self.sub_to_index)]
        rows = represented["compound"].map(compound_to_index).astype(int).to_numpy()
        columns = represented["sub"].map(self.sub_to_index).astype(int).to_numpy()
        matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(len(compound_ids), self.n_substructures),
        )
        matrix.data[:] = 1.0
        return self._predict(compound_ids, matrix, total_degree, None)

    def known_targets_for_drug(self, drug_id: str) -> list[tuple[str, float]]:
        return list(self.known_targets_by_drug.get(str(drug_id), ()))

    def summary(self) -> dict[str, Any]:
        return {
            "device_actual": str(self.device),
            "dtype_actual": str(self.dtype).removeprefix("torch."),
            "network_drug_count": self.n_drugs,
            "network_target_count": self.n_targets,
            "network_substructure_count": self.n_substructures,
            "network_node_count": self.n_nodes,
        }


__all__ = ["WSDTNBIConfig", "WSDTNBIEngine"]
