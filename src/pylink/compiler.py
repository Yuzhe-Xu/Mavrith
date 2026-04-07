from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from ._hierarchy import flatten_system
from ._model import _NormalizedModel, build_model_summary, normalize_system
from .diagnostics import Diagnostic
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
    _summary_data: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return deepcopy(dict(self._summary_data))


@dataclass(frozen=True, slots=True)
class _ModelAnalysis:
    model: _NormalizedModel
    diagnostics: tuple[Diagnostic, ...]
    input_bindings: Mapping[str, Mapping[str, Endpoint]]
    fanout: Mapping[tuple[str, str], tuple[Endpoint, ...]]
    block_order: tuple[str, ...] | None
    discrete_blocks: tuple[str, ...]
    continuous_blocks: tuple[str, ...]


def _diagnostic(
    code: str,
    message: str,
    suggestion: str,
    *,
    block_name: str | None = None,
    port_name: str | None = None,
    endpoint: str | None = None,
    connection: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=message,
        suggestion=suggestion,
        block_name=block_name,
        port_name=port_name,
        endpoint=endpoint,
        connection=connection,
    )


def _analysis_error(diagnostic: Diagnostic) -> ModelValidationError:
    if diagnostic.code in {"DIRECT_FEEDTHROUGH_SELF_LOOP", "ALGEBRAIC_LOOP"}:
        return AlgebraicLoopError.from_diagnostic(diagnostic)
    return ModelValidationError.from_diagnostic(diagnostic)


def _format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "unspecified"
    if shape == ():
        return "()"
    return str(shape)


def _collect_signal_spec_diagnostics(
    *,
    source_spec,
    target_block_name: str,
    target_port_name: str,
    target_spec,
    connection_text: str,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

    if (
        source_spec.signal_spec.dtype is not None
        and target_spec.signal_spec.dtype is not None
        and source_spec.signal_spec.dtype != target_spec.signal_spec.dtype
    ):
        diagnostics.append(
            _diagnostic(
                "INCOMPATIBLE_PORT_TYPE",
                (
                    f"Connection {connection_text} links dtype {source_spec.signal_spec.dtype!r} "
                    f"to incompatible dtype {target_spec.signal_spec.dtype!r}."
                ),
                "Use matching SignalSpec dtype declarations on both ports or leave one side unspecified.",
                block_name=target_block_name,
                port_name=target_port_name,
                endpoint=f"{target_block_name}.{target_port_name}",
                connection=connection_text,
            )
        )

    if (
        source_spec.signal_spec.shape is not None
        and target_spec.signal_spec.shape is not None
        and source_spec.signal_spec.shape != target_spec.signal_spec.shape
    ):
        diagnostics.append(
            _diagnostic(
                "INCOMPATIBLE_PORT_SHAPE",
                (
                    f"Connection {connection_text} links shape {_format_shape(source_spec.signal_spec.shape)} "
                    f"to incompatible shape {_format_shape(target_spec.signal_spec.shape)}."
                ),
                "Use matching SignalSpec shape declarations on both ports or leave one side unspecified.",
                block_name=target_block_name,
                port_name=target_port_name,
                endpoint=f"{target_block_name}.{target_port_name}",
                connection=connection_text,
            )
        )

    return diagnostics


def _analyze_model(model: _NormalizedModel) -> _ModelAnalysis:
    diagnostics: list[Diagnostic] = []
    input_bindings: dict[str, dict[str, Endpoint]] = {block.name: {} for block in model.blocks}
    fanout: dict[tuple[str, str], list[Endpoint]] = defaultdict(list)

    if not model.blocks:
        diagnostics.append(
            _diagnostic(
                "EMPTY_SYSTEM",
                "Systems must contain at least one block.",
                "Add at least one block before compiling or validating the system.",
            )
        )
        return _ModelAnalysis(
            model=model,
            diagnostics=tuple(diagnostics),
            input_bindings={},
            fanout={},
            block_order=None,
            discrete_blocks=(),
            continuous_blocks=(),
        )

    block_lookup = {block.name: block for block in model.blocks}
    valid_connections = []

    for connection in model.connections:
        source_endpoint = Endpoint(
            block_name=connection.source_block_name,
            port_name=connection.source_port_name,
        )
        target_endpoint = Endpoint(
            block_name=connection.target_block_name,
            port_name=connection.target_port_name,
        )
        connection_text = f"{connection.source} -> {connection.target}"

        source_block = block_lookup.get(connection.source_block_name)
        target_block = block_lookup.get(connection.target_block_name)

        source_valid = False
        target_valid = False

        if source_block is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_SOURCE_BLOCK",
                    f"Unknown source block {connection.source_block_name!r} in connection {connection_text}.",
                    "Add the missing source block or correct the source block name.",
                    block_name=connection.source_block_name,
                    endpoint=connection.source,
                    connection=connection_text,
                )
            )
        else:
            source_spec = source_block.get_output_spec(connection.source_port_name)
            if source_spec is None:
                diagnostics.append(
                    _diagnostic(
                        "UNKNOWN_SOURCE_PORT",
                        f"Connection {connection_text} references missing output port {connection.source_port_name!r}.",
                        "Use a declared output port on the source block.",
                        block_name=connection.source_block_name,
                        port_name=connection.source_port_name,
                        endpoint=connection.source,
                        connection=connection_text,
                    )
                )
            else:
                source_valid = True

        if target_block is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_TARGET_BLOCK",
                    f"Unknown target block {connection.target_block_name!r} in connection {connection_text}.",
                    "Add the missing target block or correct the target block name.",
                    block_name=connection.target_block_name,
                    endpoint=connection.target,
                    connection=connection_text,
                )
            )
        else:
            target_spec = target_block.get_input_spec(connection.target_port_name)
            if target_spec is None:
                diagnostics.append(
                    _diagnostic(
                        "UNKNOWN_TARGET_PORT",
                        f"Connection {connection_text} references missing input port {connection.target_port_name!r}.",
                        "Use a declared input port on the target block.",
                        block_name=connection.target_block_name,
                        port_name=connection.target_port_name,
                        endpoint=connection.target,
                        connection=connection_text,
                    )
                )
            else:
                target_valid = True

        if not (source_valid and target_valid):
            continue

        existing = input_bindings[connection.target_block_name].get(connection.target_port_name)
        if existing is not None:
            diagnostics.append(
                _diagnostic(
                    "DUPLICATE_INPUT_CONNECTION",
                    f"Input port {connection.target} already has a connection from {existing}.",
                    "Remove one upstream connection or route the signals through an explicit merge block.",
                    block_name=connection.target_block_name,
                    port_name=connection.target_port_name,
                    endpoint=connection.target,
                    connection=connection_text,
                )
            )
            continue

        source_spec = block_lookup[connection.source_block_name].get_output_spec(connection.source_port_name)
        target_spec = block_lookup[connection.target_block_name].get_input_spec(connection.target_port_name)
        assert source_spec is not None
        assert target_spec is not None
        signal_spec_diagnostics = _collect_signal_spec_diagnostics(
            source_spec=source_spec,
            target_block_name=connection.target_block_name,
            target_port_name=connection.target_port_name,
            target_spec=target_spec,
            connection_text=connection_text,
        )
        if signal_spec_diagnostics:
            diagnostics.extend(signal_spec_diagnostics)
            continue

        input_bindings[connection.target_block_name][connection.target_port_name] = source_endpoint
        fanout[(connection.source_block_name, connection.source_port_name)].append(target_endpoint)
        valid_connections.append(connection)

    for block in model.blocks:
        for spec in block.input_ports:
            if spec.required and spec.name not in input_bindings[block.name]:
                diagnostics.append(
                    _diagnostic(
                        "MISSING_REQUIRED_INPUT",
                        f"Required input {block.name}.{spec.name} is not connected.",
                        "Connect the required input or declare the port with required=False.",
                        block_name=block.name,
                        port_name=spec.name,
                        endpoint=f"{block.name}.{spec.name}",
                    )
                )

    dependencies: dict[str, set[str]] = {block.name: set() for block in model.blocks}
    reverse_edges: dict[str, list[str]] = {block.name: [] for block in model.blocks}

    for connection in valid_connections:
        target_block = block_lookup[connection.target_block_name]
        if not target_block.direct_feedthrough:
            continue

        source_name = connection.source_block_name
        target_name = connection.target_block_name
        if source_name == target_name:
            diagnostics.append(
                _diagnostic(
                    "DIRECT_FEEDTHROUGH_SELF_LOOP",
                    f"Block {target_name!r} creates a direct-feedthrough self-loop.",
                    "Break the loop with a stateful block or make direct_feedthrough=False only if the output does not depend on current inputs.",
                    block_name=target_name,
                    connection=f"{connection.source} -> {connection.target}",
                )
            )
            continue

        if source_name not in dependencies[target_name]:
            dependencies[target_name].add(source_name)
            reverse_edges[source_name].append(target_name)

    in_degree = {block.name: len(dependencies[block.name]) for block in model.blocks}
    queue = deque(block.name for block in model.blocks if in_degree[block.name] == 0)
    ordered: list[str] = []

    while queue:
        current = queue.popleft()
        ordered.append(current)
        for downstream in reverse_edges[current]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

    block_order: tuple[str, ...] | None = None
    if len(ordered) == len(model.blocks):
        block_order = tuple(ordered)
    else:
        blocked = tuple(block.name for block in model.blocks if in_degree[block.name] > 0)
        diagnostics.append(
            _diagnostic(
                "ALGEBRAIC_LOOP",
                "Direct-feedthrough algebraic loop detected involving " + ", ".join(blocked) + ".",
                "Insert state into the feedback path or make direct_feedthrough=False only for outputs that do not depend on current inputs.",
                connection=", ".join(blocked),
            )
        )

    discrete_blocks = tuple(block.name for block in model.blocks if block.kind == "discrete")
    continuous_blocks = tuple(block.name for block in model.blocks if block.kind == "continuous")

    return _ModelAnalysis(
        model=model,
        diagnostics=tuple(diagnostics),
        input_bindings={block_name: dict(bindings) for block_name, bindings in input_bindings.items()},
        fanout={key: tuple(targets) for key, targets in fanout.items()},
        block_order=block_order,
        discrete_blocks=discrete_blocks,
        continuous_blocks=continuous_blocks,
    )


def _analyze_system(system: System) -> _ModelAnalysis:
    return _analyze_model(normalize_system(system))


def _build_execution_plan(
    system: System,
    analysis: _ModelAnalysis,
    *,
    hierarchy_summary: dict[str, Any] | None = None,
) -> ExecutionPlan:
    assert analysis.block_order is not None
    return ExecutionPlan(
        system=system,
        block_order=analysis.block_order,
        input_bindings=analysis.input_bindings,
        fanout=analysis.fanout,
        discrete_blocks=analysis.discrete_blocks,
        continuous_blocks=analysis.continuous_blocks,
        _summary_data=build_model_summary(
            analysis.model,
            block_order=analysis.block_order,
            config=None,
            hierarchy=hierarchy_summary,
        ),
    )


def compile_system(system: System) -> ExecutionPlan:
    flatten_result = flatten_system(system)
    if flatten_result.diagnostics:
        raise _analysis_error(flatten_result.diagnostics[0])
    assert flatten_result.flat_system is not None
    analysis = _analyze_system(flatten_result.flat_system)
    if analysis.diagnostics:
        raise _analysis_error(analysis.diagnostics[0])
    return _build_execution_plan(
        flatten_result.flat_system,
        analysis,
        hierarchy_summary=flatten_result.hierarchy_summary,
    )
