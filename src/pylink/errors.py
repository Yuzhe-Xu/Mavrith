from __future__ import annotations

from .diagnostics import Diagnostic


class PylinkError(Exception):
    """Base exception for the framework."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.suggestion = suggestion

        fragments: list[str] = [message]
        if code is not None:
            fragments.insert(0, f"code={code}")
        if suggestion is not None:
            fragments.append(f"suggestion={suggestion}")
        super().__init__(" | ".join(fragments))


class ModelValidationError(PylinkError):
    """Raised when the system graph or block declarations are invalid."""

    @classmethod
    def from_diagnostic(cls, diagnostic: Diagnostic) -> "ModelValidationError":
        return cls(
            diagnostic.message,
            code=diagnostic.code,
            suggestion=diagnostic.suggestion,
        )


class AlgebraicLoopError(ModelValidationError):
    """Raised when direct-feedthrough dependencies create an algebraic loop."""


class SimulationError(PylinkError):
    """Raised when a block or the simulator fails during execution."""

    def __init__(
        self,
        message: str,
        *,
        block_name: str | None = None,
        time: float | None = None,
        port_name: str | None = None,
        connection: str | None = None,
        cause: BaseException | None = None,
        code: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.base_message = message
        self.block_name = block_name
        self.time = time
        self.port_name = port_name
        self.connection = connection
        self.cause = cause

        fragments: list[str] = [message]
        if block_name is not None:
            fragments.append(f"block={block_name}")
        if time is not None:
            fragments.append(f"time={time}")
        if port_name is not None:
            fragments.append(f"port={port_name}")
        if connection is not None:
            fragments.append(f"connection={connection}")
        if cause is not None:
            fragments.append(f"cause={cause!r}")
        super().__init__(
            " | ".join(fragments),
            code=code,
            suggestion=suggestion,
        )

    @classmethod
    def from_exception(
        cls,
        message: str,
        *,
        block_name: str | None = None,
        time: float | None = None,
        port_name: str | None = None,
        connection: str | None = None,
        cause: BaseException,
        code: str | None = None,
        suggestion: str | None = None,
    ) -> "SimulationError":
        return cls(
            message,
            block_name=block_name,
            time=time,
            port_name=port_name,
            connection=connection,
            cause=cause,
            code=code,
            suggestion=suggestion,
        )

    @classmethod
    def from_diagnostic(cls, diagnostic: Diagnostic) -> "SimulationError":
        return cls(
            diagnostic.message,
            block_name=diagnostic.block_name,
            time=diagnostic.time,
            port_name=diagnostic.port_name,
            connection=diagnostic.connection,
            code=diagnostic.code,
            suggestion=diagnostic.suggestion,
        )
