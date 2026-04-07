from __future__ import annotations


class PylinkError(Exception):
    """Base exception for the framework."""


class ModelValidationError(PylinkError):
    """Raised when the system graph or block declarations are invalid."""


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
    ) -> None:
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
        super().__init__(" | ".join(fragments))

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
    ) -> "SimulationError":
        return cls(
            message,
            block_name=block_name,
            time=time,
            port_name=port_name,
            connection=connection,
            cause=cause,
        )
