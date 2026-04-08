from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    suggestion: str
    severity: str = "error"
    block_name: str | None = None
    port_name: str | None = None
    endpoint: str | None = None
    connection: str | None = None
    time: float | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        return self.severity == "warning"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }

        location: dict[str, Any] = {}
        if self.block_name is not None:
            location["block_name"] = self.block_name
        if self.port_name is not None:
            location["port_name"] = self.port_name
        if self.endpoint is not None:
            location["endpoint"] = self.endpoint
        if self.connection is not None:
            location["connection"] = self.connection
        if self.time is not None:
            location["time"] = self.time
        if location:
            data["location"] = location

        return data


@dataclass(frozen=True, slots=True)
class ValidationReport:
    system_name: str
    diagnostics: tuple[Diagnostic, ...]
    _summary_data: Mapping[str, Any]

    @property
    def is_valid(self) -> bool:
        return all(not diagnostic.is_error for diagnostic in self.diagnostics)

    @property
    def error_count(self) -> int:
        return sum(1 for diagnostic in self.diagnostics if diagnostic.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for diagnostic in self.diagnostics if diagnostic.is_warning)

    def summary(self) -> dict[str, Any]:
        return deepcopy(dict(self._summary_data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "summary": self.summary(),
        }
