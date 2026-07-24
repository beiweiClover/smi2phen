"""Shared, serializable error types for the V2 runtime.

The common runtime deliberately keeps these exceptions free of scientific dependencies.  Every
error accepts a plain message so callers do not need to know about the optional structured fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class LipidScreeningError(Exception):
    """Base class for expected V2 runtime failures."""

    category = "runtime"
    code = "runtime_error"
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.details = dict(details or {})
        self.retryable = self.default_retryable if retryable is None else bool(retryable)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for node results and logs."""

        return {
            "type": type(self).__name__,
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


class ConfigurationError(LipidScreeningError):
    """The workflow configuration is missing, malformed, or internally inconsistent."""

    category = "configuration"
    code = "configuration_error"


class InputError(LipidScreeningError):
    """A registered input is missing or violates its input contract."""

    category = "input"
    code = "input_error"


class ResourceError(LipidScreeningError):
    """A configured read-only resource is missing or unusable."""

    category = "resource"
    code = "resource_error"


class EnvironmentError(LipidScreeningError):
    """The execution environment cannot satisfy a declared runtime requirement."""

    category = "environment"
    code = "environment_error"


class ExecutionError(LipidScreeningError):
    """A node could not complete its registered execution."""

    category = "execution"
    code = "execution_error"
    default_retryable = True


class OutputContractError(LipidScreeningError):
    """A produced output is missing, unsafe, or violates its artifact contract."""

    category = "output_contract"
    code = "output_contract_error"


class PathSafetyError(OutputContractError):
    """A path would escape or overlap a protected runtime boundary."""

    category = "output_contract"
    code = "path_safety_error"


__all__ = [
    "ConfigurationError",
    "EnvironmentError",
    "ExecutionError",
    "InputError",
    "LipidScreeningError",
    "OutputContractError",
    "PathSafetyError",
    "ResourceError",
]
