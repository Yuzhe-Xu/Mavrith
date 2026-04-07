from __future__ import annotations

from dataclasses import dataclass, field

from .core import Block
from .errors import ModelValidationError


@dataclass(frozen=True, slots=True)
class Endpoint:
    block_name: str
    port_name: str

    @classmethod
    def parse(cls, raw: str) -> "Endpoint":
        if raw.count(".") != 1:
            raise ModelValidationError(
                f"Endpoint {raw!r} must use the format '<block>.<port>'."
            )
        block_name, port_name = raw.split(".", maxsplit=1)
        if not block_name or not port_name:
            raise ModelValidationError(
                f"Endpoint {raw!r} must use the format '<block>.<port>'."
            )
        return cls(block_name=block_name, port_name=port_name)

    def __str__(self) -> str:
        return f"{self.block_name}.{self.port_name}"


@dataclass(frozen=True, slots=True)
class Connection:
    source: Endpoint
    target: Endpoint

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}"


@dataclass(slots=True)
class System:
    name: str = "system"
    blocks: dict[str, Block] = field(default_factory=dict, init=False)
    connections: list[Connection] = field(default_factory=list, init=False)

    def add_block(self, name: str, block: Block) -> "System":
        if not name:
            raise ModelValidationError("Block names must be non-empty.")
        if "." in name:
            raise ModelValidationError("Block names cannot contain '.'.")
        if name in self.blocks:
            raise ModelValidationError(f"Block name {name!r} is already in use.")
        self.blocks[name] = block
        return self

    def connect(self, source: str, target: str) -> "System":
        self.connections.append(Connection(source=Endpoint.parse(source), target=Endpoint.parse(target)))
        return self
