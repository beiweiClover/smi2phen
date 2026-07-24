"""Lazy NetInfer preparation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lipid_screening_agent.runtime import EnvironmentError


@dataclass(frozen=True, slots=True)
class PrepareDependencies:
    pd: Any
    chem: Any
    all_chem: Any
    standardize: Any
    rdkit_version: str
    openpyxl_version: str


def load_prepare_dependencies() -> PrepareDependencies:
    """Import pandas, openpyxl, and RDKit only when preparation runs."""

    try:
        import openpyxl
        import pandas as pd
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception as exc:
        raise EnvironmentError(
            "NetInfer input preparation requires pandas, openpyxl, and RDKit",
            details={
                "dependencies": ["pandas", "openpyxl", "rdkit"],
                "import_error_type": type(exc).__name__,
            },
            retryable=False,
        ) from exc
    return PrepareDependencies(
        pd=pd,
        chem=Chem,
        all_chem=AllChem,
        standardize=rdMolStandardize,
        rdkit_version=str(getattr(rdBase, "rdkitVersion", "unknown")),
        openpyxl_version=str(getattr(openpyxl, "__version__", "unknown")),
    )


__all__ = ["PrepareDependencies", "load_prepare_dependencies"]
