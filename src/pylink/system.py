from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .core import Block, SignalSpec
from .errors import ModelValidationError


@dataclass(frozen=True, slots=True)
class Endpoint:
    block_name: str
    port_name: str

    @classmethod
    def parse(cls, raw: str) -> "Endpoint":
        if raw.count(".") != 1:
            raise ModelValidationError(
                f"Endpoint {raw!r} must use the format '<block>.<port>'.",
                code="INVALID_ENDPOINT_FORMAT",
                suggestion="Use endpoint strings like 'block_name.port_name'.",
            )
        block_name, port_name = raw.split(".", maxsplit=1)
        if not block_name or not port_name:
            raise ModelValidationError(
                f"Endpoint {raw!r} must use the format '<block>.<port>'.",
                code="INVALID_ENDPOINT_FORMAT",
                suggestion="Use endpoint strings like 'block_name.port_name'.",
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


@dataclass(frozen=True, slots=True)
class ExposedInput:
    name: str
    required: bool
    signal_spec: SignalSpec
    targets: tuple[Endpoint, ...]


@dataclass(frozen=True, slots=True)
class ExposedOutput:
    name: str
    signal_spec: SignalSpec
    source: Endpoint


def _validate_component_name(name: str, *, allow_hierarchy_path: bool = False) -> None:
    if not name:
        raise ModelValidationError(
            "Component names must be non-empty.",
            code="INVALID_COMPONENT_NAME",
            suggestion="Choose a non-empty block or subsystem name before adding the component.",
        )
    if "." in name:
        raise ModelValidationError(
            "Component names cannot contain '.'.",
            code="INVALID_COMPONENT_NAME",
            suggestion="Remove '.' characters from the component name.",
        )
    if not allow_hierarchy_path and "/" in name:
        raise ModelValidationError(
            "Component names cannot contain '/'.",
            code="INVALID_COMPONENT_NAME",
            suggestion="Remove '/' characters from the component name.",
        )


def _validate_exposed_port_name(name: str) -> None:
    if not name:
        raise ModelValidationError(
            "Subsystem port names must be non-empty.",
            code="INVALID_SUBSYSTEM_PORT_NAME",
            suggestion="Choose a non-empty exposed input or output name.",
        )
    if "." in name or "/" in name:
        raise ModelValidationError(
            "Subsystem port names cannot contain '.' or '/'.",
            code="INVALID_SUBSYSTEM_PORT_NAME",
            suggestion="Use simple exposed port names without '.' or '/'.",
        )


class _ComponentContainer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blocks: dict[str, Block] = {}
        self.subsystems: dict[str, Subsystem] = {}
        self.connections: list[Connection] = []
        self._component_order: list[str] = []

    def add_block(self, name: str, block: Block) -> "_ComponentContainer":
        _validate_component_name(name)
        if name in self.blocks:
            raise ModelValidationError(
                f"Block name {name!r} is already in use.",
                code="DUPLICATE_BLOCK_NAME",
                suggestion="Choose a unique block name before adding the block.",
            )
        if name in self.subsystems:
            raise ModelValidationError(
                f"Component name {name!r} is already in use.",
                code="DUPLICATE_COMPONENT_NAME",
                suggestion="Choose a unique block or subsystem name before adding the component.",
            )
        self.blocks[name] = block
        self._component_order.append(name)
        return self

    def _add_flat_block(self, name: str, block: Block) -> "_ComponentContainer":
        _validate_component_name(name, allow_hierarchy_path=True)
        if name in self.blocks or name in self.subsystems:
            raise ModelValidationError(
                f"Component name {name!r} is already in use.",
                code="DUPLICATE_COMPONENT_NAME",
                suggestion="Choose a unique block or subsystem name before adding the component.",
            )
        self.blocks[name] = block
        self._component_order.append(name)
        return self

    def add_subsystem(self, name: str, subsystem: "Subsystem") -> "_ComponentContainer":
        _validate_component_name(name)
        if name in self.blocks or name in self.subsystems:
            raise ModelValidationError(
                f"Subsystem name {name!r} is already in use.",
                code="DUPLICATE_SUBSYSTEM_NAME",
                suggestion="Choose a unique subsystem name before adding the subsystem.",
            )
        self.subsystems[name] = subsystem
        self._component_order.append(name)
        return self

    def connect(self, source: str, target: str) -> "_ComponentContainer":
        self.connections.append(Connection(source=Endpoint.parse(source), target=Endpoint.parse(target)))
        return self

    def get_component(self, name: str) -> Block | "Subsystem" | None:
        if name in self.blocks:
            return self.blocks[name]
        return self.subsystems.get(name)

    def iter_components(self) -> Iterable[tuple[str, Block | "Subsystem"]]:
        for name in self._component_order:
            component = self.get_component(name)
            if component is not None:
                yield name, component

    def iter_subsystems(self) -> Iterable[tuple[str, "Subsystem"]]:
        for name in self._component_order:
            subsystem = self.subsystems.get(name)
            if subsystem is not None:
                yield name, subsystem

    def has_subsystems(self) -> bool:
        return bool(self.subsystems)


class System(_ComponentContainer):
    def __init__(self, name: str = "system") -> None:
        super().__init__(name=name)


class Subsystem(_ComponentContainer):
    def __init__(self, name: str = "subsystem") -> None:
        super().__init__(name=name)
        self._exposed_inputs: dict[str, ExposedInput] = {}
        self._exposed_outputs: dict[str, ExposedOutput] = {}

    @property
    def exposed_inputs(self) -> dict[str, ExposedInput]:
        return dict(self._exposed_inputs)

    @property
    def exposed_outputs(self) -> dict[str, ExposedOutput]:
        return dict(self._exposed_outputs)

    def expose_input(
        self,
        name: str,
        target: str,
        *,
        spec: SignalSpec | None = None,
        required: bool = True,
    ) -> "Subsystem":
        _validate_exposed_port_name(name)
        endpoint = Endpoint.parse(target)
        existing = self._exposed_inputs.get(name)
        signal_spec = spec or SignalSpec()
        if existing is None:
            self._exposed_inputs[name] = ExposedInput(
                name=name,
                required=required,
                signal_spec=signal_spec,
                targets=(endpoint,),
            )
            return self
        if existing.required != required:
            raise ModelValidationError(
                f"Subsystem input {name!r} was already declared with required={existing.required}.",
                code="INCONSISTENT_SUBSYSTEM_INPUT",
                suggestion="Reuse the same required= value for repeated expose_input() calls.",
            )
        if existing.signal_spec != signal_spec:
            raise ModelValidationError(
                f"Subsystem input {name!r} was already declared with a different SignalSpec.",
                code="INCONSISTENT_SUBSYSTEM_INPUT",
                suggestion="Reuse the same spec=SignalSpec(...) when exposing one logical input to multiple targets.",
            )
        if endpoint in existing.targets:
            raise ModelValidationError(
                f"Subsystem input {name!r} is already bound to target {endpoint}.",
                code="DUPLICATE_SUBSYSTEM_INPUT_TARGET",
                suggestion="Remove the duplicate exposed input target binding.",
            )
        self._exposed_inputs[name] = ExposedInput(
            name=name,
            required=existing.required,
            signal_spec=existing.signal_spec,
            targets=existing.targets + (endpoint,),
        )
        return self

    def expose_output(
        self,
        source: str,
        name: str,
        *,
        spec: SignalSpec | None = None,
    ) -> "Subsystem":
        _validate_exposed_port_name(name)
        if name in self._exposed_outputs:
            raise ModelValidationError(
                f"Subsystem output {name!r} is already defined.",
                code="DUPLICATE_SUBSYSTEM_OUTPUT",
                suggestion="Choose a unique exposed output name for each subsystem output.",
            )
        self._exposed_outputs[name] = ExposedOutput(
            name=name,
            signal_spec=spec or SignalSpec(),
            source=Endpoint.parse(source),
        )
        return self
