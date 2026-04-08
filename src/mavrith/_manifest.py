from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import Block, ContinuousBlock, DiscreteBlock
from .errors import ModelValidationError, MavrithError
from .simulation import SimulationConfig, Simulator
from .system import SourceRef, Subsystem, System

_SCHEMA_VERSION = 1
_UNSUPPORTED = object()
_RESERVED_BLOCK_ATTRIBUTES = frozenset(
    {
        "input_ports",
        "output_ports",
        "direct_feedthrough",
        "parameters",
        "description",
        "sample_time",
        "offset",
        "priority",
    }
)


@dataclass(frozen=True, slots=True)
class ExportResult:
    graph_path: Path
    detail_index_path: Path | None
    detail_paths: tuple[Path, ...]
    sharded_detail: bool


@dataclass(frozen=True, slots=True)
class _ComponentNode:
    path: str
    name: str
    kind: str
    component: System | Subsystem | Block
    parent_path: str | None
    instance_source: SourceRef | None
    detail_ref: str


@dataclass(frozen=True, slots=True)
class _ManifestContext:
    system: System
    nodes: Mapping[str, _ComponentNode]


def _detail_ref_for_path(path: str) -> str:
    if not path:
        return "detail/system.yaml"
    return f"detail/{path}.yaml"


def _block_kind(block: Block) -> str:
    if isinstance(block, ContinuousBlock):
        return "continuous"
    if isinstance(block, DiscreteBlock):
        return "discrete"
    return "stateless"


def _source_ref_summary(source_ref: SourceRef | None) -> dict[str, object] | None:
    if source_ref is None:
        return None
    return source_ref.summary()


def _object_source_summary(obj: Any) -> dict[str, object] | None:
    try:
        filename = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    return {
        "file": str(Path(filename).resolve()),
        "line": int(line),
        "function": getattr(obj, "__qualname__", getattr(obj, "__name__", None)),
    }


def _stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _description_hash(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _port_summary(port) -> dict[str, Any]:
    return {
        "name": port.name,
        "direction": port.direction.value,
        "required": port.required,
        "signal_spec": port.signal_spec.summary(),
    }


def _normalize_manifest_value(value: Any, *, fallback_repr: bool) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {
            "real": value.real,
            "imag": value.imag,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _normalize_manifest_value(value.item(), fallback_repr=fallback_repr)
    if isinstance(value, np.ndarray):
        return _normalize_manifest_value(value.tolist(), fallback_repr=fallback_repr)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_manifest_value(dataclasses.asdict(value), fallback_repr=fallback_repr)
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            normalized = _normalize_manifest_value(value[key], fallback_repr=fallback_repr)
            if normalized is _UNSUPPORTED:
                return _UNSUPPORTED
            normalized_mapping[str(key)] = normalized
        return normalized_mapping
    if isinstance(value, (list, tuple)):
        normalized_items: list[Any] = []
        for item in value:
            normalized = _normalize_manifest_value(item, fallback_repr=fallback_repr)
            if normalized is _UNSUPPORTED:
                return _UNSUPPORTED
            normalized_items.append(normalized)
        return normalized_items
    if isinstance(value, (set, frozenset)):
        normalized_items = []
        for item in sorted(value, key=repr):
            normalized = _normalize_manifest_value(item, fallback_repr=fallback_repr)
            if normalized is _UNSUPPORTED:
                return _UNSUPPORTED
            normalized_items.append(normalized)
        return normalized_items
    if fallback_repr:
        value_type = type(value)
        return {
            "repr": repr(value),
            "type": f"{value_type.__module__}.{value_type.__qualname__}",
        }
    return _UNSUPPORTED


def _build_parameter_summary(block: Block) -> dict[str, Any]:
    declared = _normalize_manifest_value(dict(block.parameters), fallback_repr=True)
    inferred: dict[str, Any] = {}
    explicit_parameter_keys = set(block.parameters)
    for name, value in sorted(vars(block).items()):
        if name.startswith("_") or name in _RESERVED_BLOCK_ATTRIBUTES or name in explicit_parameter_keys:
            continue
        normalized = _normalize_manifest_value(value, fallback_repr=False)
        if normalized is _UNSUPPORTED:
            continue
        inferred[name] = normalized
    return {
        "declared": declared if declared is not _UNSUPPORTED else {},
        "inferred": inferred,
        "inferred_mode": "best_effort",
    }


def _effective_description(component: System | Subsystem | Block) -> tuple[str | None, str]:
    explicit = getattr(component, "description", None)
    if explicit:
        return explicit, "explicit"
    if isinstance(component, Block):
        raw_docstring = type(component).__dict__.get("__doc__")
        docstring = inspect.cleandoc(raw_docstring) if raw_docstring else None
        if docstring:
            return docstring, "docstring"
    return None, "missing"


def _block_auto_summary(block: Block) -> str:
    kind = _block_kind(block)
    summary = (
        f"{kind} block with {len(block.input_ports)} input port(s) "
        f"and {len(block.output_ports)} output port(s)."
    )
    if isinstance(block, DiscreteBlock):
        summary += (
            f" sample_time={block.sample_time}, offset={block.offset}, "
            f"priority={block.priority}."
        )
    return summary


def _container_auto_summary(container: System | Subsystem) -> str:
    component_count = sum(1 for _ in container.iter_components())
    connection_count = len(container.connections)
    kind = "system" if isinstance(container, System) else "subsystem"
    return f"{kind} with {component_count} child component(s) and {connection_count} local connection(s)."


def _callable_fingerprint_data(obj: Any) -> dict[str, Any]:
    source_ref = _object_source_summary(obj)
    try:
        source_text = inspect.getsource(obj)
    except (OSError, TypeError):
        source_text = None
    return {
        "source_ref": source_ref,
        "source_text": source_text,
    }


def _build_block_implementation_source(block: Block) -> dict[str, Any]:
    callbacks = ["output"]
    if isinstance(block, DiscreteBlock):
        callbacks.extend(["initial_discrete_state", "update_state"])
    if isinstance(block, ContinuousBlock):
        callbacks.extend(["initial_continuous_state", "derivative"])
    methods: dict[str, Any] = {}
    block_type = type(block)
    for name in callbacks:
        callback = getattr(block_type, name, None)
        if callable(callback):
            methods[name] = _object_source_summary(callback)
    return {
        "class": _object_source_summary(block_type),
        "methods": methods,
    }


def _build_block_fingerprint(block: Block, parameter_summary: Mapping[str, Any]) -> str:
    block_type = type(block)
    callbacks = {"output": _callable_fingerprint_data(block_type.output)}
    if isinstance(block, DiscreteBlock):
        callbacks["initial_discrete_state"] = _callable_fingerprint_data(
            block_type.initial_discrete_state
        )
        callbacks["update_state"] = _callable_fingerprint_data(block_type.update_state)
    if isinstance(block, ContinuousBlock):
        callbacks["initial_continuous_state"] = _callable_fingerprint_data(
            block_type.initial_continuous_state
        )
        callbacks["derivative"] = _callable_fingerprint_data(block_type.derivative)
    return _stable_hash(
        {
            "class_name": block_type.__name__,
            "module": block_type.__module__,
            "block_kind": _block_kind(block),
            "direct_feedthrough": block.direct_feedthrough,
            "timing": {
                "sample_time": getattr(block, "sample_time", None),
                "offset": getattr(block, "offset", None),
                "priority": getattr(block, "priority", None),
            },
            "ports": {
                "inputs": [_port_summary(port) for port in block.input_ports],
                "outputs": [_port_summary(port) for port in block.output_ports],
            },
            "parameter_keys": {
                "declared": sorted(parameter_summary["declared"]),
                "inferred": sorted(parameter_summary["inferred"]),
            },
            "callbacks": callbacks,
        }
    )


def _connection_summary(connection) -> dict[str, Any]:
    return {
        "source": str(connection.source),
        "target": str(connection.target),
        "instance_source": _source_ref_summary(connection.source_ref),
    }


def _child_path(parent_path: str, child_name: str) -> str:
    if not parent_path:
        return child_name
    return f"{parent_path}/{child_name}"


def _build_manifest_context(system: System) -> _ManifestContext:
    nodes: dict[str, _ComponentNode] = {}

    def visit_container(
        container: System | Subsystem,
        *,
        path: str,
        parent_path: str | None,
        instance_source: SourceRef | None,
        active_ids: set[int],
    ) -> None:
        container_id = id(container)
        if container_id in active_ids:
            raise ModelValidationError(
                f"Subsystem path {path or system.name!r} participates in a cyclic subsystem reference.",
                code="CYCLIC_SUBSYSTEM_REFERENCE",
                suggestion="Break the subsystem containment cycle before exporting manifests.",
            )
        if path in nodes:
            raise ModelValidationError(
                f"Component path {path!r} is already in use.",
                code="DUPLICATE_COMPONENT_PATH",
                suggestion="Ensure each block or subsystem resolves to a unique path.",
            )

        kind = "system" if isinstance(container, System) else "subsystem"
        node_name = container.name if kind == "system" else path.rsplit("/", maxsplit=1)[-1]
        nodes[path] = _ComponentNode(
            path=path,
            name=node_name,
            kind=kind,
            component=container,
            parent_path=parent_path,
            instance_source=instance_source,
            detail_ref=_detail_ref_for_path(path),
        )

        active_ids.add(container_id)
        try:
            for child_name, component in container.iter_components():
                child_path = _child_path(path, child_name)
                child_source = container.get_component_source_ref(child_name)
                if child_path in nodes:
                    raise ModelValidationError(
                        f"Component path {child_path!r} is already in use.",
                        code="DUPLICATE_COMPONENT_PATH",
                        suggestion="Ensure each block or subsystem resolves to a unique path.",
                    )
                if isinstance(component, Block):
                    nodes[child_path] = _ComponentNode(
                        path=child_path,
                        name=child_name,
                        kind="block",
                        component=component,
                        parent_path=path,
                        instance_source=child_source,
                        detail_ref=_detail_ref_for_path(child_path),
                    )
                    continue
                visit_container(
                    component,
                    path=child_path,
                    parent_path=path,
                    instance_source=child_source,
                    active_ids=active_ids,
                )
        finally:
            active_ids.remove(container_id)

    visit_container(system, path="", parent_path=None, instance_source=None, active_ids=set())
    return _ManifestContext(system=system, nodes=nodes)


def _build_graph_manifest_from_context(context: _ManifestContext) -> dict[str, Any]:
    containers: list[dict[str, Any]] = []
    for node in context.nodes.values():
        if node.kind not in {"system", "subsystem"}:
            continue
        container = node.component
        assert isinstance(container, (System, Subsystem))
        children = [
            {
                "name": child_name,
                "kind": "block" if isinstance(component, Block) else "subsystem",
                "path": _child_path(node.path, child_name),
                "detail_ref": _detail_ref_for_path(_child_path(node.path, child_name)),
            }
            for child_name, component in container.iter_components()
        ]
        connections = [
            {
                "source": str(connection.source),
                "target": str(connection.target),
            }
            for connection in container.connections
        ]
        containers.append(
            {
                "name": context.system.name if node.kind == "system" else node.name,
                "kind": node.kind,
                "path": node.path,
                "detail_ref": node.detail_ref,
                "children": children,
                "connections": connections,
            }
        )

    return {
        "schema_version": _SCHEMA_VERSION,
        "manifest_kind": "mavrith_graph",
        "system_name": context.system.name,
        "root": {
            "name": context.system.name,
            "kind": "system",
            "path": "",
            "detail_ref": _detail_ref_for_path(""),
        },
        "containers": containers,
    }


def _build_detail_index(context: _ManifestContext) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "manifest_kind": "mavrith_detail_index",
        "system_name": context.system.name,
        "root_detail_ref": _detail_ref_for_path(""),
        "entries": [
            {
                "name": context.system.name if node.kind == "system" else node.name,
                "kind": node.kind,
                "path": node.path,
                "detail_ref": node.detail_ref,
            }
            for node in context.nodes.values()
        ],
    }


def _block_detail(context: _ManifestContext, node: _ComponentNode) -> dict[str, Any]:
    block = node.component
    assert isinstance(block, Block)
    description, description_origin = _effective_description(block)
    parameter_summary = _build_parameter_summary(block)
    return {
        "schema_version": _SCHEMA_VERSION,
        "manifest_kind": "mavrith_detail",
        "system_name": context.system.name,
        "id": node.path,
        "path": node.path,
        "name": node.name,
        "kind": "block",
        "detail_ref": node.detail_ref,
        "class_name": type(block).__name__,
        "module": type(block).__module__,
        "block_kind": _block_kind(block),
        "direct_feedthrough": block.direct_feedthrough,
        "ports": {
            "inputs": [_port_summary(port) for port in block.input_ports],
            "outputs": [_port_summary(port) for port in block.output_ports],
        },
        "timing": {
            "sample_time": getattr(block, "sample_time", None),
            "offset": getattr(block, "offset", None),
            "priority": getattr(block, "priority", None),
        },
        "parameters": parameter_summary,
        "instance_source": _source_ref_summary(node.instance_source),
        "implementation_source": _build_block_implementation_source(block),
        "description": description,
        "description_origin": description_origin,
        "auto_summary": _block_auto_summary(block),
        "implementation_fingerprint": _build_block_fingerprint(block, parameter_summary),
        "description_text_hash": _description_hash(description),
        "description_status": "current",
    }


def _subsystem_port_summary(subsystem: Subsystem) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = [
        {
            "name": item.name,
            "required": item.required,
            "signal_spec": item.signal_spec.summary(),
            "targets": [
                {
                    "target": str(target),
                    "instance_source": _source_ref_summary(source_ref),
                }
                for target, source_ref in zip(item.targets, item.source_refs, strict=False)
            ],
        }
        for item in subsystem.exposed_inputs.values()
    ]
    outputs = [
        {
            "name": item.name,
            "signal_spec": item.signal_spec.summary(),
            "source": str(item.source),
            "instance_source": _source_ref_summary(item.source_ref),
        }
        for item in subsystem.exposed_outputs.values()
    ]
    return inputs, outputs


def _container_fingerprint(
    container: System | Subsystem,
    *,
    node: _ComponentNode,
) -> str:
    fingerprint_data: dict[str, Any] = {
        "kind": node.kind,
        "path": node.path,
        "children": [
            {
                "name": child_name,
                "kind": "block" if isinstance(component, Block) else "subsystem",
            }
            for child_name, component in container.iter_components()
        ],
        "connections": [
            {
                "source": str(connection.source),
                "target": str(connection.target),
            }
            for connection in container.connections
        ],
    }
    if isinstance(container, Subsystem):
        exposed_inputs, exposed_outputs = _subsystem_port_summary(container)
        fingerprint_data["exposed_inputs"] = exposed_inputs
        fingerprint_data["exposed_outputs"] = exposed_outputs
    return _stable_hash(fingerprint_data)


def _container_detail(
    context: _ManifestContext,
    node: _ComponentNode,
    *,
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    container = node.component
    assert isinstance(container, (System, Subsystem))
    description, description_origin = _effective_description(container)
    detail = {
        "schema_version": _SCHEMA_VERSION,
        "manifest_kind": "mavrith_detail",
        "system_name": context.system.name,
        "id": context.system.name if node.kind == "system" else node.path,
        "path": node.path,
        "name": context.system.name if node.kind == "system" else node.name,
        "kind": node.kind,
        "detail_ref": node.detail_ref,
        "instance_source": _source_ref_summary(node.instance_source),
        "children": [
            {
                "name": child_name,
                "kind": "block" if isinstance(component, Block) else "subsystem",
                "path": _child_path(node.path, child_name),
                "detail_ref": _detail_ref_for_path(_child_path(node.path, child_name)),
            }
            for child_name, component in container.iter_components()
        ],
        "connections": [_connection_summary(connection) for connection in container.connections],
        "description": description,
        "description_origin": description_origin,
        "auto_summary": _container_auto_summary(container),
        "implementation_fingerprint": _container_fingerprint(container, node=node),
        "description_text_hash": _description_hash(description),
        "description_status": "current",
    }
    if isinstance(container, Subsystem):
        exposed_inputs, exposed_outputs = _subsystem_port_summary(container)
        detail["exposed_inputs"] = exposed_inputs
        detail["exposed_outputs"] = exposed_outputs

    if node.kind == "system" and config is not None:
        summary = Simulator().validate(context.system, config).summary()
        detail["execution_order"] = summary["execution_order"]
        detail["rate_groups"] = summary["rate_groups"]
        detail["cross_rate_connections"] = summary["cross_rate_connections"]
        detail["time_grid_constraints"] = summary["time_grid_constraints"]
    return detail


def _build_detail_manifest_from_context(
    context: _ManifestContext,
    *,
    path: str | None = None,
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    normalized_path = path or ""
    node = context.nodes.get(normalized_path)
    if node is None:
        raise ModelValidationError(
            f"Unknown component path {normalized_path!r}.",
            code="UNKNOWN_COMPONENT_PATH",
            suggestion="Use build_graph_manifest() or the detail index to choose a valid component path.",
        )
    if node.kind == "block":
        return _block_detail(context, node)
    return _container_detail(context, node, config=config if not normalized_path else None)


def build_graph_manifest(system: System) -> dict[str, Any]:
    context = _build_manifest_context(system)
    return _build_graph_manifest_from_context(context)


def build_detail_manifest(
    system: System,
    *,
    path: str | None = None,
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    context = _build_manifest_context(system)
    return _build_detail_manifest_from_context(context, path=path, config=config)


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise MavrithError(
            "YAML export requires the optional PyYAML dependency.",
            code="YAML_SUPPORT_UNAVAILABLE",
            suggestion="Install mavrith[yaml] or add PyYAML to your environment.",
        ) from exc
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, data: Any) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise MavrithError(
            "YAML export requires the optional PyYAML dependency.",
            code="YAML_SUPPORT_UNAVAILABLE",
            suggestion="Install mavrith[yaml] or add PyYAML to your environment.",
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def _apply_description_status(detail: dict[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    updated = dict(detail)
    if (
        previous is not None
        and updated.get("description") is not None
        and previous.get("implementation_fingerprint") != updated["implementation_fingerprint"]
        and previous.get("description_text_hash") == updated["description_text_hash"]
    ):
        updated["description_status"] = "review_recommended"
        return updated
    updated["description_status"] = "current"
    return updated


def write_manifest_bundle(
    system: System,
    out_dir: str | Path,
    *,
    sharded_detail: bool = True,
) -> ExportResult:
    context = _build_manifest_context(system)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "graph.yaml"
    _write_yaml(graph_path, _build_graph_manifest_from_context(context))

    if sharded_detail:
        detail_index_path = output_dir / "detail" / "index.yaml"
        _write_yaml(detail_index_path, _build_detail_index(context))

        detail_paths: list[Path] = []
        for node in context.nodes.values():
            detail = _build_detail_manifest_from_context(context, path=node.path or None, config=None)
            detail_path = output_dir / Path(node.detail_ref)
            previous = _load_yaml(detail_path)
            _write_yaml(detail_path, _apply_description_status(detail, previous))
            detail_paths.append(detail_path)

        return ExportResult(
            graph_path=graph_path,
            detail_index_path=detail_index_path,
            detail_paths=tuple(detail_paths),
            sharded_detail=True,
        )

    detail_bundle_path = output_dir / "detail.yaml"
    previous_bundle = _load_yaml(detail_bundle_path) or {}
    previous_details = {
        item.get("path", ""): item
        for item in previous_bundle.get("details", [])
        if isinstance(item, Mapping)
    }
    details = []
    for node in context.nodes.values():
        detail = _build_detail_manifest_from_context(context, path=node.path or None, config=None)
        details.append(_apply_description_status(detail, previous_details.get(node.path)))

    _write_yaml(
        detail_bundle_path,
        {
            "schema_version": _SCHEMA_VERSION,
            "manifest_kind": "mavrith_detail_bundle",
            "system_name": context.system.name,
            "index": _build_detail_index(context),
            "details": details,
        },
    )
    return ExportResult(
        graph_path=graph_path,
        detail_index_path=None,
        detail_paths=(detail_bundle_path,),
        sharded_detail=False,
    )
