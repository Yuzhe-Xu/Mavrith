from __future__ import annotations

from copy import copy
from dataclasses import dataclass

from .core import Block, PortSpec, SignalSpec
from .diagnostics import Diagnostic
from .system import Endpoint, ExposedInput, ExposedOutput, Subsystem, System


@dataclass(frozen=True, slots=True)
class _RelativeEndpoint:
    path: tuple[str, ...]
    port_name: str

    def prefixed(self, prefix: tuple[str, ...]) -> "_RelativeEndpoint":
        return _RelativeEndpoint(path=prefix + self.path, port_name=self.port_name)

    def block_name(self) -> str:
        return "/".join(self.path)


@dataclass(frozen=True, slots=True)
class _RelativeConnection:
    source: _RelativeEndpoint
    target: _RelativeEndpoint

    def prefixed(self, prefix: tuple[str, ...]) -> "_RelativeConnection":
        return _RelativeConnection(
            source=self.source.prefixed(prefix),
            target=self.target.prefixed(prefix),
        )


@dataclass(frozen=True, slots=True)
class _RelativeBlock:
    path: tuple[str, ...]
    block: Block
    optional_inputs: frozenset[str]

    def prefixed(self, prefix: tuple[str, ...]) -> "_RelativeBlock":
        return _RelativeBlock(
            path=prefix + self.path,
            block=self.block,
            optional_inputs=self.optional_inputs,
        )


@dataclass(frozen=True, slots=True)
class _FlattenedInput:
    name: str
    required: bool
    signal_spec: SignalSpec
    targets: tuple[_RelativeEndpoint, ...]


@dataclass(frozen=True, slots=True)
class _FlattenedOutput:
    name: str
    signal_spec: SignalSpec
    source: _RelativeEndpoint


@dataclass(frozen=True, slots=True)
class _FlattenedContainer:
    blocks: tuple[_RelativeBlock, ...]
    connections: tuple[_RelativeConnection, ...]
    exposed_inputs: dict[str, _FlattenedInput]
    exposed_outputs: dict[str, _FlattenedOutput]


@dataclass(frozen=True, slots=True)
class _ResolvedOutput:
    signal_spec: SignalSpec
    source: _RelativeEndpoint


@dataclass(frozen=True, slots=True)
class _ResolvedInput:
    signal_spec: SignalSpec
    targets: tuple[_RelativeEndpoint, ...]
    subsystem_port: tuple[str, str] | None
    required: bool


@dataclass(frozen=True, slots=True)
class FlattenResult:
    flat_system: System | None
    diagnostics: tuple[Diagnostic, ...]
    hierarchy_summary: dict[str, object]


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


def _format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "unspecified"
    if shape == ():
        return "()"
    return str(shape)


def _collect_signal_spec_diagnostics(
    *,
    source_spec: SignalSpec,
    target_spec: SignalSpec,
    connection_text: str,
    block_name: str | None,
    port_name: str | None,
    endpoint: str | None,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if (
        source_spec.dtype is not None
        and target_spec.dtype is not None
        and source_spec.dtype != target_spec.dtype
    ):
        diagnostics.append(
            _diagnostic(
                "INCOMPATIBLE_PORT_TYPE",
                (
                    f"Connection {connection_text} links dtype {source_spec.dtype!r} "
                    f"to incompatible dtype {target_spec.dtype!r}."
                ),
                "Use matching SignalSpec dtype declarations on both ports or leave one side unspecified.",
                block_name=block_name,
                port_name=port_name,
                endpoint=endpoint,
                connection=connection_text,
            )
        )
    if (
        source_spec.shape is not None
        and target_spec.shape is not None
        and source_spec.shape != target_spec.shape
    ):
        diagnostics.append(
            _diagnostic(
                "INCOMPATIBLE_PORT_SHAPE",
                (
                    f"Connection {connection_text} links shape {_format_shape(source_spec.shape)} "
                    f"to incompatible shape {_format_shape(target_spec.shape)}."
                ),
                "Use matching SignalSpec shape declarations on both ports or leave one side unspecified.",
                block_name=block_name,
                port_name=port_name,
                endpoint=endpoint,
                connection=connection_text,
            )
        )
    return diagnostics


def _merge_signal_specs(
    *,
    existing: SignalSpec,
    candidate: SignalSpec,
    code_prefix: str,
    endpoint: str,
    diagnostics: list[Diagnostic],
) -> SignalSpec:
    dtype = existing.dtype
    if dtype is None:
        dtype = candidate.dtype
    elif candidate.dtype is not None and candidate.dtype != dtype:
        diagnostics.append(
            _diagnostic(
                f"{code_prefix}_TYPE",
                (
                    f"Subsystem boundary {endpoint} combines incompatible dtypes "
                    f"{existing.dtype!r} and {candidate.dtype!r}."
                ),
                "Use one consistent SignalSpec dtype across the exposed subsystem boundary.",
                endpoint=endpoint,
            )
        )
    shape = existing.shape
    if shape is None:
        shape = candidate.shape
    elif candidate.shape is not None and candidate.shape != shape:
        diagnostics.append(
            _diagnostic(
                f"{code_prefix}_SHAPE",
                (
                    f"Subsystem boundary {endpoint} combines incompatible shapes "
                    f"{_format_shape(existing.shape)} and {_format_shape(candidate.shape)}."
                ),
                "Use one consistent SignalSpec shape across the exposed subsystem boundary.",
                endpoint=endpoint,
            )
        )
    return SignalSpec(dtype=dtype, shape=shape)


def _clone_block(block: Block, optional_inputs: frozenset[str]) -> Block:
    clone = copy(block)
    clone.input_ports = tuple(
        PortSpec.input(
            spec.name,
            required=False if spec.name in optional_inputs else spec.required,
            spec=spec.signal_spec,
        )
        for spec in block.input_ports
    )
    clone.output_ports = tuple(
        PortSpec.output(spec.name, spec=spec.signal_spec) for spec in block.output_ports
    )
    return clone


def _resolve_output(
    *,
    container: System | Subsystem,
    endpoint: Endpoint,
    flattened_children: dict[str, _FlattenedContainer],
    diagnostics: list[Diagnostic],
    connection_text: str | None = None,
) -> _ResolvedOutput | None:
    component = container.get_component(endpoint.block_name)
    if component is None:
        diagnostics.append(
            _diagnostic(
                "UNKNOWN_SOURCE_COMPONENT",
                f"Unknown source component {endpoint.block_name!r}.",
                "Add the missing source component or correct the endpoint spelling.",
                block_name=endpoint.block_name,
                endpoint=str(endpoint),
                connection=connection_text,
            )
        )
        return None

    if isinstance(component, Block):
        spec = component.get_output_spec(endpoint.port_name)
        if spec is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_SOURCE_PORT",
                    f"Connection {connection_text or endpoint} references missing output port {endpoint.port_name!r}.",
                    "Use a declared output port on the source block.",
                    block_name=endpoint.block_name,
                    port_name=endpoint.port_name,
                    endpoint=str(endpoint),
                    connection=connection_text,
                )
            )
            return None
        return _ResolvedOutput(
            signal_spec=spec.signal_spec,
            source=_RelativeEndpoint(path=(endpoint.block_name,), port_name=endpoint.port_name),
        )

    flattened = flattened_children[endpoint.block_name]
    exposed = flattened.exposed_outputs.get(endpoint.port_name)
    if exposed is None:
        diagnostics.append(
            _diagnostic(
                "UNKNOWN_SUBSYSTEM_OUTPUT",
                f"Connection {connection_text or endpoint} references missing subsystem output {endpoint.port_name!r}.",
                "Expose the requested subsystem output or correct the endpoint spelling.",
                block_name=endpoint.block_name,
                port_name=endpoint.port_name,
                endpoint=str(endpoint),
                connection=connection_text,
            )
        )
        return None
    return _ResolvedOutput(
        signal_spec=exposed.signal_spec,
        source=exposed.source.prefixed((endpoint.block_name,)),
    )


def _resolve_input(
    *,
    container: System | Subsystem,
    endpoint: Endpoint,
    flattened_children: dict[str, _FlattenedContainer],
    diagnostics: list[Diagnostic],
    connection_text: str | None = None,
) -> _ResolvedInput | None:
    component = container.get_component(endpoint.block_name)
    if component is None:
        diagnostics.append(
            _diagnostic(
                "UNKNOWN_TARGET_COMPONENT",
                f"Unknown target component {endpoint.block_name!r}.",
                "Add the missing target component or correct the endpoint spelling.",
                block_name=endpoint.block_name,
                endpoint=str(endpoint),
                connection=connection_text,
            )
        )
        return None

    if isinstance(component, Block):
        spec = component.get_input_spec(endpoint.port_name)
        if spec is None:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_TARGET_PORT",
                    f"Connection {connection_text or endpoint} references missing input port {endpoint.port_name!r}.",
                    "Use a declared input port on the target block.",
                    block_name=endpoint.block_name,
                    port_name=endpoint.port_name,
                    endpoint=str(endpoint),
                    connection=connection_text,
                )
            )
            return None
        return _ResolvedInput(
            signal_spec=spec.signal_spec,
            targets=(_RelativeEndpoint(path=(endpoint.block_name,), port_name=endpoint.port_name),),
            subsystem_port=None,
            required=spec.required,
        )

    flattened = flattened_children[endpoint.block_name]
    exposed = flattened.exposed_inputs.get(endpoint.port_name)
    if exposed is None:
        diagnostics.append(
            _diagnostic(
                "UNKNOWN_SUBSYSTEM_INPUT",
                f"Connection {connection_text or endpoint} references missing subsystem input {endpoint.port_name!r}.",
                "Expose the requested subsystem input or correct the endpoint spelling.",
                block_name=endpoint.block_name,
                port_name=endpoint.port_name,
                endpoint=str(endpoint),
                connection=connection_text,
            )
        )
        return None
    return _ResolvedInput(
        signal_spec=exposed.signal_spec,
        targets=tuple(target.prefixed((endpoint.block_name,)) for target in exposed.targets),
        subsystem_port=(endpoint.block_name, endpoint.port_name),
        required=exposed.required,
    )


def _flatten_container(
    container: System | Subsystem,
    memo: dict[int, _FlattenedContainer],
) -> tuple[_FlattenedContainer, list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    flattened_children: dict[str, _FlattenedContainer] = {}
    ordered_blocks: list[_RelativeBlock] = []
    block_builders: dict[tuple[str, ...], set[str]] = {}
    connections: list[_RelativeConnection] = []
    bound_subsystem_inputs: set[tuple[str, str]] = set()

    if isinstance(container, Subsystem) and not any(True for _ in container.iter_components()):
        diagnostics.append(
            _diagnostic(
                "EMPTY_SUBSYSTEM",
                f"Subsystem {container.name!r} must contain at least one block or child subsystem.",
                "Add internal components before compiling or validating the subsystem.",
            )
        )

    for name, component in container.iter_components():
        if isinstance(component, Block):
            relative = _RelativeBlock(path=(name,), block=component, optional_inputs=frozenset())
            ordered_blocks.append(relative)
            block_builders[relative.path] = set()
            continue

        flattened_child = memo[id(component)]
        flattened_children[name] = flattened_child
        for child_block in flattened_child.blocks:
            relative = child_block.prefixed((name,))
            ordered_blocks.append(relative)
            block_builders[relative.path] = set(relative.optional_inputs)
        for child_connection in flattened_child.connections:
            connections.append(child_connection.prefixed((name,)))

    for connection in container.connections:
        connection_text = str(connection)
        resolved_source = _resolve_output(
            container=container,
            endpoint=connection.source,
            flattened_children=flattened_children,
            diagnostics=diagnostics,
            connection_text=connection_text,
        )
        resolved_target = _resolve_input(
            container=container,
            endpoint=connection.target,
            flattened_children=flattened_children,
            diagnostics=diagnostics,
            connection_text=connection_text,
        )
        if resolved_target is not None and resolved_target.subsystem_port is not None:
            bound_subsystem_inputs.add(resolved_target.subsystem_port)
        if resolved_source is None or resolved_target is None:
            continue
        signal_spec_diagnostics = _collect_signal_spec_diagnostics(
            source_spec=resolved_source.signal_spec,
            target_spec=resolved_target.signal_spec,
            connection_text=connection_text,
            block_name=connection.target.block_name,
            port_name=connection.target.port_name,
            endpoint=str(connection.target),
        )
        if signal_spec_diagnostics:
            diagnostics.extend(signal_spec_diagnostics)
            continue
        for target in resolved_target.targets:
            connections.append(_RelativeConnection(source=resolved_source.source, target=target))

    exposed_inputs: dict[str, _FlattenedInput] = {}
    exposed_outputs: dict[str, _FlattenedOutput] = {}
    if isinstance(container, Subsystem):
        for name, exposed in container.exposed_inputs.items():
            targets: list[_RelativeEndpoint] = []
            effective_spec = exposed.signal_spec
            endpoint_text = f"{container.name}.{name}"
            for target in exposed.targets:
                resolved_target = _resolve_input(
                    container=container,
                    endpoint=target,
                    flattened_children=flattened_children,
                    diagnostics=diagnostics,
                )
                if resolved_target is None:
                    continue
                if resolved_target.subsystem_port is not None:
                    bound_subsystem_inputs.add(resolved_target.subsystem_port)
                if (
                    resolved_target.subsystem_port is not None
                    and not exposed.required
                    and resolved_target.required
                ):
                    diagnostics.append(
                        _diagnostic(
                            "INCOMPATIBLE_SUBSYSTEM_REQUIREDNESS",
                            (
                                f"Subsystem input {name!r} is declared optional but feeds required target "
                                f"{target}."
                            ),
                            "Keep the parent subsystem input required=True or relax the downstream target contract.",
                            endpoint=f"{container.name}.{name}",
                            connection=f"{container.name}.{name} -> {target}",
                        )
                    )
                effective_spec = _merge_signal_specs(
                    existing=effective_spec,
                    candidate=resolved_target.signal_spec,
                    code_prefix="INCOMPATIBLE_SUBSYSTEM_INPUT",
                    endpoint=endpoint_text,
                    diagnostics=diagnostics,
                )
                for leaf_target in resolved_target.targets:
                    targets.append(leaf_target)
                    if leaf_target.path in block_builders:
                        block_builders[leaf_target.path].add(leaf_target.port_name)
            if targets:
                exposed_inputs[name] = _FlattenedInput(
                    name=name,
                    required=exposed.required,
                    signal_spec=effective_spec,
                    targets=tuple(targets),
                )

        for name, exposed in container.exposed_outputs.items():
            endpoint_text = f"{container.name}.{name}"
            resolved_source = _resolve_output(
                container=container,
                endpoint=exposed.source,
                flattened_children=flattened_children,
                diagnostics=diagnostics,
            )
            if resolved_source is None:
                continue
            effective_spec = _merge_signal_specs(
                existing=exposed.signal_spec,
                candidate=resolved_source.signal_spec,
                code_prefix="INCOMPATIBLE_SUBSYSTEM_OUTPUT",
                endpoint=endpoint_text,
                diagnostics=diagnostics,
            )
            exposed_outputs_item = _FlattenedOutput(
                name=name,
                signal_spec=effective_spec,
                source=resolved_source.source,
            )
            exposed_outputs[name] = exposed_outputs_item

    for child_name, child in container.iter_subsystems():
        flattened_child = memo[id(child)]
        for input_name, exposed_input in flattened_child.exposed_inputs.items():
            if exposed_input.required and (child_name, input_name) not in bound_subsystem_inputs:
                diagnostics.append(
                    _diagnostic(
                        "MISSING_REQUIRED_SUBSYSTEM_INPUT",
                        f"Required subsystem input {child_name}.{input_name} is not connected.",
                        "Connect the subsystem input or expose it through the parent subsystem boundary.",
                        block_name=child_name,
                        port_name=input_name,
                        endpoint=f"{child_name}.{input_name}",
                    )
                )

    finalized_blocks = tuple(
        _RelativeBlock(
            path=block.path,
            block=block.block,
            optional_inputs=frozenset(block_builders.get(block.path, block.optional_inputs)),
        )
        for block in ordered_blocks
    )
    return (
        _FlattenedContainer(
            blocks=finalized_blocks,
            connections=tuple(connections),
            exposed_inputs=exposed_inputs,
            exposed_outputs=exposed_outputs,
        ),
        diagnostics,
    )


def _build_hierarchy_summary(
    root: System,
    memo: dict[int, _FlattenedContainer],
) -> dict[str, object]:
    def build_container_node(container: System | Subsystem, name: str, path: tuple[str, ...]) -> dict[str, object]:
        node_kind = "system" if isinstance(container, System) else "subsystem"
        flattened = memo.get(id(container))
        exposed_inputs = []
        exposed_outputs = []
        if flattened is not None:
            exposed_inputs = [
                {
                    "name": item.name,
                    "required": item.required,
                    "signal_spec": item.signal_spec.summary(),
                }
                for item in flattened.exposed_inputs.values()
            ]
            exposed_outputs = [
                {
                    "name": item.name,
                    "signal_spec": item.signal_spec.summary(),
                }
                for item in flattened.exposed_outputs.values()
            ]

        children: list[dict[str, object]] = []
        for child_name, component in container.iter_components():
            child_path = path + (child_name,)
            if isinstance(component, Block):
                children.append(
                    {
                        "name": child_name,
                        "kind": "block",
                        "path": "/".join(child_path),
                        "class_name": component.__class__.__name__,
                        "inputs": [spec.name for spec in component.input_ports],
                        "outputs": [spec.name for spec in component.output_ports],
                    }
                )
                continue
            children.append(build_container_node(component, child_name, child_path))

        return {
            "name": name,
            "kind": node_kind,
            "path": "/".join(path),
            "exposed_inputs": exposed_inputs,
            "exposed_outputs": exposed_outputs,
            "children": children,
        }

    return build_container_node(root, root.name, ())


def flatten_system(system: System) -> FlattenResult:
    memo: dict[int, _FlattenedContainer] = {}
    diagnostics: list[Diagnostic] = []
    stack: list[tuple[System | Subsystem, bool]] = [(system, False)]
    visiting: set[int] = set()
    saw_cycle = False

    while stack:
        container, expanded = stack.pop()
        container_id = id(container)

        if expanded:
            visiting.discard(container_id)
            flattened, container_diagnostics = _flatten_container(container, memo)
            memo[container_id] = flattened
            diagnostics.extend(container_diagnostics)
            continue

        if container_id in memo:
            continue

        visiting.add(container_id)
        stack.append((container, True))
        for child_name, subsystem in reversed(list(container.iter_subsystems())):
            child_id = id(subsystem)
            if child_id in visiting:
                diagnostics.append(
                    _diagnostic(
                        "CYCLIC_SUBSYSTEM_REFERENCE",
                        f"Subsystem {child_name!r} creates a cyclic subsystem reference.",
                        "Break the subsystem containment cycle before compiling the model.",
                        block_name=child_name,
                    )
                )
                saw_cycle = True
                continue
            if child_id not in memo:
                stack.append((subsystem, False))

    hierarchy_summary = _build_hierarchy_summary(system, memo)
    if saw_cycle or id(system) not in memo:
        return FlattenResult(
            flat_system=None,
            diagnostics=tuple(diagnostics),
            hierarchy_summary=hierarchy_summary,
        )

    root_flat = memo[id(system)]
    flat_system = System(system.name)
    for block in root_flat.blocks:
        flat_system._add_flat_block(
            block.path[0] if len(block.path) == 1 else "/".join(block.path),
            _clone_block(block.block, block.optional_inputs),
        )
    for connection in root_flat.connections:
        flat_system.connect(
            f"{connection.source.block_name()}.{connection.source.port_name}",
            f"{connection.target.block_name()}.{connection.target.port_name}",
        )

    return FlattenResult(
        flat_system=flat_system,
        diagnostics=tuple(diagnostics),
        hierarchy_summary=hierarchy_summary,
    )
