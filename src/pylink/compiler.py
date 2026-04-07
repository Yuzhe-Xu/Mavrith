from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Mapping

from .core import ContinuousBlock, DiscreteBlock
from .errors import AlgebraicLoopError, ModelValidationError
from .system import Endpoint, System


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    system: System
    block_order: tuple[str, ...]
    input_bindings: Mapping[str, Mapping[str, Endpoint]]
    fanout: Mapping[tuple[str, str], tuple[Endpoint, ...]]
    discrete_blocks: tuple[str, ...]
    continuous_blocks: tuple[str, ...]


def compile_system(system: System) -> ExecutionPlan:
    if not system.blocks:
        raise ModelValidationError("Systems must contain at least one block.")

    input_bindings: dict[str, dict[str, Endpoint]] = {name: {} for name in system.blocks}
    fanout: dict[tuple[str, str], list[Endpoint]] = defaultdict(list)

    for connection in system.connections:
        try:
            source_block = system.blocks[connection.source.block_name]
        except KeyError as exc:
            raise ModelValidationError(
                f"Unknown source block {connection.source.block_name!r} in connection {connection}."
            ) from exc
        try:
            target_block = system.blocks[connection.target.block_name]
        except KeyError as exc:
            raise ModelValidationError(
                f"Unknown target block {connection.target.block_name!r} in connection {connection}."
            ) from exc

        source_spec = source_block.get_output_spec(connection.source.port_name)
        if source_spec is None:
            raise ModelValidationError(
                f"Connection {connection} references missing output port {connection.source.port_name!r}."
            )
        target_spec = target_block.get_input_spec(connection.target.port_name)
        if target_spec is None:
            raise ModelValidationError(
                f"Connection {connection} references missing input port {connection.target.port_name!r}."
            )
        if connection.target.port_name in input_bindings[connection.target.block_name]:
            existing = input_bindings[connection.target.block_name][connection.target.port_name]
            raise ModelValidationError(
                f"Input port {connection.target} already has a connection from {existing}."
            )

        input_bindings[connection.target.block_name][connection.target.port_name] = connection.source
        fanout[(connection.source.block_name, connection.source.port_name)].append(connection.target)

    for block_name, block in system.blocks.items():
        for spec in block.input_ports:
            if spec.required and spec.name not in input_bindings[block_name]:
                raise ModelValidationError(
                    f"Required input {block_name}.{spec.name} is not connected."
                )

    dependencies: dict[str, set[str]] = {name: set() for name in system.blocks}
    reverse_edges: dict[str, set[str]] = {name: set() for name in system.blocks}
    for connection in system.connections:
        target_block = system.blocks[connection.target.block_name]
        if target_block.direct_feedthrough:
            source_name = connection.source.block_name
            target_name = connection.target.block_name
            if source_name == target_name:
                raise AlgebraicLoopError(
                    f"Block {target_name!r} creates a direct-feedthrough self-loop."
                )
            if source_name not in dependencies[target_name]:
                dependencies[target_name].add(source_name)
                reverse_edges[source_name].add(target_name)

    in_degree = {name: len(dependencies[name]) for name in system.blocks}
    queue = deque(name for name in system.blocks if in_degree[name] == 0)
    ordered: list[str] = []

    while queue:
        current = queue.popleft()
        ordered.append(current)
        for downstream in reverse_edges[current]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

    if len(ordered) != len(system.blocks):
        blocked = sorted(name for name, degree in in_degree.items() if degree > 0)
        raise AlgebraicLoopError(
            "Direct-feedthrough algebraic loop detected involving "
            + ", ".join(blocked)
            + "."
        )

    discrete_blocks = tuple(
        name for name, block in system.blocks.items() if isinstance(block, DiscreteBlock)
    )
    continuous_blocks = tuple(
        name for name, block in system.blocks.items() if isinstance(block, ContinuousBlock)
    )

    frozen_bindings = {
        block_name: dict(bindings) for block_name, bindings in input_bindings.items()
    }
    frozen_fanout = {key: tuple(targets) for key, targets in fanout.items()}

    return ExecutionPlan(
        system=system,
        block_order=tuple(ordered),
        input_bindings=frozen_bindings,
        fanout=frozen_fanout,
        discrete_blocks=discrete_blocks,
        continuous_blocks=continuous_blocks,
    )
