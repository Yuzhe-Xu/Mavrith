from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ._hierarchy import flatten_system
from ._model import _NormalizedBlock, _NormalizedModel, build_model_summary, normalize_system
from .diagnostics import Diagnostic
from .errors import AlgebraicLoopError, ModelValidationError
from .system import Endpoint, System


@dataclass(frozen=True, slots=True)
class ResolvedRateGroup:
    sample_time: float
    offset: float
    declared_priority: int | None
    resolved_priority: int
    block_names: tuple[str, ...]
    declaration_index: int

    def summary(self) -> dict[str, Any]:
        return {
            "sample_time": self.sample_time,
            "offset": self.offset,
            "declared_priority": self.declared_priority,
            "resolved_priority": self.resolved_priority,
            "blocks": list(self.block_names),
        }


@dataclass(frozen=True, slots=True)
class CrossRateConnection:
    source: str
    target: str
    classification: str
    source_kind: str
    target_kind: str
    source_sample_time: float | None
    source_offset: float | None
    source_resolved_priority: int | None
    target_sample_time: float | None
    target_offset: float | None
    target_resolved_priority: int | None

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "classification": self.classification,
            "source_kind": self.source_kind,
            "target_kind": self.target_kind,
            "source_rate_group": {
                "sample_time": self.source_sample_time,
                "offset": self.source_offset,
                "resolved_priority": self.source_resolved_priority,
            }
            if self.source_sample_time is not None
            else None,
            "target_rate_group": {
                "sample_time": self.target_sample_time,
                "offset": self.target_offset,
                "resolved_priority": self.target_resolved_priority,
            }
            if self.target_sample_time is not None
            else None,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    system: System
    block_order: tuple[str, ...]
    input_bindings: Mapping[str, Mapping[str, Endpoint]]
    fanout: Mapping[tuple[str, str], tuple[Endpoint, ...]]
    discrete_blocks: tuple[str, ...]
    continuous_blocks: tuple[str, ...]
    rate_groups: tuple[ResolvedRateGroup, ...]
    resolved_priorities: Mapping[str, int]
    cross_rate_connections: tuple[CrossRateConnection, ...]
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
    rate_groups: tuple[ResolvedRateGroup, ...]
    resolved_priorities: Mapping[str, int]
    cross_rate_connections: tuple[CrossRateConnection, ...]
    execution_notes: Mapping[str, Any]


def _diagnostic(
    code: str,
    message: str,
    suggestion: str,
    *,
    severity: str = "error",
    block_name: str | None = None,
    port_name: str | None = None,
    endpoint: str | None = None,
    connection: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=message,
        suggestion=suggestion,
        severity=severity,
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


def _error_diagnostics(diagnostics: tuple[Diagnostic, ...] | list[Diagnostic]) -> tuple[Diagnostic, ...]:
    return tuple(diagnostic for diagnostic in diagnostics if diagnostic.is_error)


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


def _classify_connection(source_block: _NormalizedBlock, target_block: _NormalizedBlock) -> str:
    if source_block.kind != "discrete" and target_block.kind != "discrete":
        return "same-rate"
    if source_block.kind == "discrete" and target_block.kind != "discrete":
        return "slow-to-fast"
    if source_block.kind != "discrete" and target_block.kind == "discrete":
        return "fast-to-slow"
    assert source_block.sample_time is not None
    assert source_block.offset is not None
    assert target_block.sample_time is not None
    assert target_block.offset is not None
    if (
        source_block.sample_time == target_block.sample_time
        and source_block.offset == target_block.offset
    ):
        return "same-rate"
    if source_block.sample_time == target_block.sample_time:
        return "same-period-different-offset"
    if source_block.sample_time > target_block.sample_time:
        return "slow-to-fast"
    return "fast-to-slow"


def _resolve_rate_groups(
    model: _NormalizedModel,
) -> tuple[tuple[ResolvedRateGroup, ...], dict[str, int], list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    blocks_by_group: dict[tuple[float, float], list[_NormalizedBlock]] = defaultdict(list)
    declaration_index: dict[tuple[float, float], int] = {}

    for index, block in enumerate(model.blocks):
        if block.kind != "discrete":
            continue
        assert block.sample_time is not None
        assert block.offset is not None
        key = (block.sample_time, block.offset)
        blocks_by_group[key].append(block)
        declaration_index.setdefault(key, index)

    declared_priority_by_group: dict[tuple[float, float], int | None] = {}
    explicit_owner_by_priority: dict[int, tuple[float, float]] = {}

    for key, blocks in blocks_by_group.items():
        explicit_priorities = {block.priority for block in blocks if block.priority is not None}
        if len(explicit_priorities) > 1:
            diagnostics.append(
                _diagnostic(
                    "INCONSISTENT_RATE_GROUP_PRIORITY",
                    (
                        f"Rate group sample_time={key[0]} offset={key[1]} mixes priorities "
                        f"{sorted(explicit_priorities)}."
                    ),
                    "Reuse one priority value for every block in the same rate group.",
                    block_name=blocks[0].name,
                )
            )
            declared_priority_by_group[key] = min(explicit_priorities)
            continue

        declared_priority = next(iter(explicit_priorities), None)
        declared_priority_by_group[key] = declared_priority
        if declared_priority is None:
            continue
        owner = explicit_owner_by_priority.get(declared_priority)
        if owner is not None and owner != key:
            diagnostics.append(
                _diagnostic(
                    "DUPLICATE_RATE_GROUP_PRIORITY",
                    (
                        "Resolved rate groups cannot share the same explicit priority value "
                        f"{declared_priority}."
                    ),
                    "Choose a unique priority for each discrete rate group.",
                    block_name=blocks[0].name,
                )
            )
            continue
        explicit_owner_by_priority[declared_priority] = key

    resolved_priority_by_group: dict[tuple[float, float], int] = {}
    used_priorities = set(explicit_owner_by_priority)
    for key, declared_priority in declared_priority_by_group.items():
        if declared_priority is not None:
            resolved_priority_by_group[key] = declared_priority

    next_priority = 0
    auto_keys = sorted(
        blocks_by_group,
        key=lambda key: (key[0], key[1], declaration_index[key]),
    )
    for key in auto_keys:
        if key in resolved_priority_by_group:
            continue
        while next_priority in used_priorities:
            next_priority += 1
        resolved_priority_by_group[key] = next_priority
        used_priorities.add(next_priority)
        next_priority += 1

    rate_groups = tuple(
        sorted(
            (
                ResolvedRateGroup(
                    sample_time=key[0],
                    offset=key[1],
                    declared_priority=declared_priority_by_group[key],
                    resolved_priority=resolved_priority_by_group[key],
                    block_names=tuple(block.name for block in blocks_by_group[key]),
                    declaration_index=declaration_index[key],
                )
                for key in blocks_by_group
            ),
            key=lambda group: (
                group.resolved_priority,
                group.sample_time,
                group.offset,
                group.declaration_index,
            ),
        )
    )

    resolved_priorities = {
        block_name: group.resolved_priority
        for group in rate_groups
        for block_name in group.block_names
    }
    return rate_groups, resolved_priorities, diagnostics


def _enrich_model_with_rate_metadata(
    model: _NormalizedModel,
    *,
    rate_groups: tuple[ResolvedRateGroup, ...],
    resolved_priorities: Mapping[str, int],
) -> _NormalizedModel:
    group_summary_by_block = {
        block_name: {
            "sample_time": group.sample_time,
            "offset": group.offset,
            "declared_priority": group.declared_priority,
            "resolved_priority": group.resolved_priority,
        }
        for group in rate_groups
        for block_name in group.block_names
    }
    enriched_blocks = tuple(
        replace(
            block,
            resolved_priority=resolved_priorities.get(block.name),
            rate_group=deepcopy(group_summary_by_block.get(block.name)),
        )
        for block in model.blocks
    )
    return _NormalizedModel(
        name=model.name,
        blocks=enriched_blocks,
        connections=model.connections,
    )


def _analyze_cross_rate_connections(
    *,
    model: _NormalizedModel,
    valid_connections,
    resolved_priorities: Mapping[str, int],
) -> tuple[tuple[CrossRateConnection, ...], list[Diagnostic]]:
    block_lookup = {block.name: block for block in model.blocks}
    diagnostics: list[Diagnostic] = []
    crossings: list[CrossRateConnection] = []

    for connection in valid_connections:
        source_block = block_lookup[connection.source_block_name]
        target_block = block_lookup[connection.target_block_name]
        classification = _classify_connection(source_block, target_block)
        crossing = CrossRateConnection(
            source=connection.source,
            target=connection.target,
            classification=classification,
            source_kind=source_block.kind,
            target_kind=target_block.kind,
            source_sample_time=source_block.sample_time,
            source_offset=source_block.offset,
            source_resolved_priority=resolved_priorities.get(source_block.name),
            target_sample_time=target_block.sample_time,
            target_offset=target_block.offset,
            target_resolved_priority=resolved_priorities.get(target_block.name),
        )
        crossings.append(crossing)
        if classification == "same-rate":
            continue
        diagnostics.append(
            _diagnostic(
                "CROSS_RATE_CONNECTION",
                (
                    f"Connection {connection.source} -> {connection.target} crosses rates with "
                    f"{classification} semantics."
                ),
                "This connection uses implicit hold semantics; inspect the resolved rate groups and priorities in the model summary.",
                severity="warning",
                block_name=connection.target_block_name,
                port_name=connection.target_port_name,
                endpoint=connection.target,
                connection=f"{connection.source} -> {connection.target}",
            )
        )

    return tuple(crossings), diagnostics


def _analyze_model(model: _NormalizedModel) -> _ModelAnalysis:
    diagnostics: list[Diagnostic] = []
    input_bindings: dict[str, dict[str, Endpoint]] = {block.name: {} for block in model.blocks}
    fanout: dict[tuple[str, str], list[Endpoint]] = defaultdict(list)
    rate_groups, resolved_priorities, rate_diagnostics = _resolve_rate_groups(model)
    diagnostics.extend(rate_diagnostics)

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
            rate_groups=rate_groups,
            resolved_priorities={},
            cross_rate_connections=(),
            execution_notes={},
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

    cross_rate_connections, cross_rate_diagnostics = _analyze_cross_rate_connections(
        model=model,
        valid_connections=valid_connections,
        resolved_priorities=resolved_priorities,
    )
    diagnostics.extend(cross_rate_diagnostics)

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
    enriched_model = _enrich_model_with_rate_metadata(
        model,
        rate_groups=rate_groups,
        resolved_priorities=resolved_priorities,
    )

    execution_notes = {
        "block_order_role": (
            "execution_order resolves same-time direct-feedthrough evaluation only; "
            "sample-hit execution order comes from resolved rate groups and priorities."
        ),
        "discrete_schedule_role": (
            "At each sample hit, discrete rate groups execute in ascending resolved_priority order. "
            "Lower numbers run first."
        ),
    }

    return _ModelAnalysis(
        model=enriched_model,
        diagnostics=tuple(diagnostics),
        input_bindings={block_name: dict(bindings) for block_name, bindings in input_bindings.items()},
        fanout={key: tuple(targets) for key, targets in fanout.items()},
        block_order=block_order,
        discrete_blocks=discrete_blocks,
        continuous_blocks=continuous_blocks,
        rate_groups=rate_groups,
        resolved_priorities=dict(resolved_priorities),
        cross_rate_connections=cross_rate_connections,
        execution_notes=execution_notes,
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
        rate_groups=analysis.rate_groups,
        resolved_priorities=analysis.resolved_priorities,
        cross_rate_connections=analysis.cross_rate_connections,
        _summary_data=build_model_summary(
            analysis.model,
            block_order=analysis.block_order,
            config=None,
            hierarchy=hierarchy_summary,
            rate_groups=[group.summary() for group in analysis.rate_groups],
            cross_rate_connections=[item.summary() for item in analysis.cross_rate_connections],
            execution_notes=dict(analysis.execution_notes),
        ),
    )


def compile_system(system: System) -> ExecutionPlan:
    flatten_result = flatten_system(system)
    flatten_errors = _error_diagnostics(flatten_result.diagnostics)
    if flatten_errors:
        raise _analysis_error(flatten_errors[0])
    assert flatten_result.flat_system is not None
    analysis = _analyze_system(flatten_result.flat_system)
    analysis_errors = _error_diagnostics(analysis.diagnostics)
    if analysis_errors:
        raise _analysis_error(analysis_errors[0])
    return _build_execution_plan(
        flatten_result.flat_system,
        analysis,
        hierarchy_summary=flatten_result.hierarchy_summary,
    )
