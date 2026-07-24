"""Lazy scientific dependency loading with stable environment-error classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lipid_screening_agent.runtime import EnvironmentError


@dataclass(frozen=True, slots=True)
class PredictionDependencies:
    np: Any
    pd: Any
    torch: Any
    functional: Any
    chem: Any
    all_chem: Any
    rdkit_version: str


@dataclass(frozen=True, slots=True)
class DiseaseDependencies:
    np: Any
    pd: Any
    stats: Any
    multipletests: Any


@dataclass(frozen=True, slots=True)
class ScoringDependencies:
    np: Any
    pd: Any


def load_prediction_dependencies() -> PredictionDependencies:
    """Import NumPy/pandas/Torch/RDKit only when model execution starts."""

    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as functional
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem
    except Exception as exc:
        raise EnvironmentError(
            "GPS drug-profile prediction requires NumPy, pandas, PyTorch, and RDKit",
            details={
                "dependencies": ["numpy", "pandas", "torch", "rdkit"],
                "import_error_type": type(exc).__name__,
            },
            retryable=False,
        ) from exc
    return PredictionDependencies(
        np=np,
        pd=pd,
        torch=torch,
        functional=functional,
        chem=Chem,
        all_chem=AllChem,
        rdkit_version=str(getattr(rdBase, "rdkitVersion", "unknown")),
    )


def load_disease_dependencies() -> DiseaseDependencies:
    """Import the legacy DEG dependency set only when DEG execution starts."""

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:
        raise EnvironmentError(
            "GPS disease-signature construction requires NumPy, pandas, SciPy, and statsmodels",
            details={
                "dependencies": ["numpy", "pandas", "scipy", "statsmodels"],
                "import_error_type": type(exc).__name__,
            },
            retryable=False,
        ) from exc
    return DiseaseDependencies(
        np=np,
        pd=pd,
        stats=stats,
        multipletests=multipletests,
    )


def load_scoring_dependencies() -> ScoringDependencies:
    """Import the GPS scoring dependency set only when scoring starts."""

    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise EnvironmentError(
            "GPS compound scoring requires NumPy and pandas",
            details={
                "dependencies": ["numpy", "pandas"],
                "import_error_type": type(exc).__name__,
            },
            retryable=False,
        ) from exc
    return ScoringDependencies(np=np, pd=pd)


__all__ = [
    "DiseaseDependencies",
    "PredictionDependencies",
    "ScoringDependencies",
    "load_disease_dependencies",
    "load_prediction_dependencies",
    "load_scoring_dependencies",
]
