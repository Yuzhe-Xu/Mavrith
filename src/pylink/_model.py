from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import Block, ContinuousBlock, DiscreteBlock, PortDirection, SignalSpec
from .system import System


@dataclass(frozen=True, slots=True)
class _NormalizedPort:
    name: str
    direction: PortDirection
    required: bool
    signal_spec: SignalSpec

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "required": self.required,
            "signal_spec": self.signal_spec.summary(),
        }


@dataclass(frozen=True, slots=True)
class _NormalizedBlock:
    name: str
    class_name: str
    kind: str
    direct_feedthrough: bool
    sample_time: float | None
    input_ports: tuple[_NormalizedPort, ...]
    output_ports: tuple[_NormalizedPort, ...]

    def get_input_spec(self, port_name: str) -> _NormalizedPort | None:
        return next((spec for spec in self.input_ports if spec.name == port_name), None)

    def get_output_spec(self, port_name: str) -> _NormalizedPort | None:
        return next((spec for spec in self.output_ports if spec.name == port_name), None)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "kind": self.kind,
            "direct_feedthrough": self.direct_feedthrough,
            "sample_time": self.sample_time,
            "inputs": [spec.summary() for spec in self.input_ports],
            "outputs": [spec.summary() for spec in self.output_ports],
        }


@dataclass(frozen=True, slots=True)
class _NormalizedConnection:
    source_block_name: str
    source_port_name: str
    target_block_name: str
    target_port_name: str

    @property
    def source(self) -> str:
        return f"{self.source_block_name}.{self.source_port_name}"

    @property
    def target(self) -> str:
        return f"{self.target_block_name}.{self.target_port_name}"

    def summary(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "connection": f"{self.source} -> {self.target}",
        }


@dataclass(frozen=True, slots=True)
class _NormalizedModel:
    name: str
    blocks: tuple[_NormalizedBlock, ...]
    connections: tuple[_NormalizedConnection, ...]

    def get_block(self, block_name: str) -> _NormalizedBlock | None:
        return next((block for block in self.blocks if block.name == block_name), None)


def _block_kind(block: Block) -> str:
    if isinstance(block, ContinuousBlock):
        return "continuous"
    if isinstance(block, DiscreteBlock):
        return "discrete"
    return "stateless"


def normalize_system(system: System) -> _NormalizedModel:
    normalized_blocks: list[_NormalizedBlock] = []
    for block_name, block in system.blocks.items():
        sample_time = float(block.sample_time) if isinstance(block, DiscreteBlock) else None
        normalized_blocks.append(
            _NormalizedBlock(
                name=block_name,
                class_name=block.__class__.__name__,
                kind=_block_kind(block),
                direct_feedthrough=block.direct_feedthrough,
                sample_time=sample_time,
                input_ports=tuple(
                    _NormalizedPort(
                        name=spec.name,
                        direction=spec.direction,
                        required=spec.required,
                        signal_spec=spec.signal_spec,
                    )
                    for spec in block.input_ports
                ),
                output_ports=tuple(
                    _NormalizedPort(
                        name=spec.name,
                        direction=spec.direction,
                        required=spec.required,
                        signal_spec=spec.signal_spec,
                    )
                    for spec in block.output_ports
                ),
            )
        )

    normalized_connections = tuple(
        _NormalizedConnection(
            source_block_name=connection.source.block_name,
            source_port_name=connection.source.port_name,
            target_block_name=connection.target.block_name,
            target_port_name=connection.target.port_name,
        )
        for connection in system.connections
    )

    return _NormalizedModel(
        name=system.name,
        blocks=tuple(normalized_blocks),
        connections=normalized_connections,
    )


def build_model_summary(
    model: _NormalizedModel,
    *,
    block_order: tuple[str, ...] | None,
    config: Any | None = None,
    hierarchy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    discrete_blocks = [
        {"name": block.name, "sample_time": block.sample_time}
        for block in model.blocks
        if block.kind == "discrete"
    ]
    continuous_blocks = [block.name for block in model.blocks if block.kind == "continuous"]

    config_summary: dict[str, Any] | None = None
    if config is not None:
        config_summary = {
            "start": config.start,
            "stop": config.stop,
            "dt": config.dt,
        }

    return {
        "system_name": model.name,
        "blocks": [block.summary() for block in model.blocks],
        "connections": [connection.summary() for connection in model.connections],
        "execution_order": list(block_order) if block_order is not None else None,
        "hierarchy": hierarchy,
        "stateful_blocks": {
            "discrete": discrete_blocks,
            "continuous": continuous_blocks,
        },
        "time_grid_constraints": {
            "config": config_summary,
            "rules": [
                "(stop - start) must be an integer multiple of dt.",
                "Each discrete block sample_time must be an integer multiple of dt.",
            ],
        },
    }
