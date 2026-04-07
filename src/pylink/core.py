from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence


class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class PortSpec:
    name: str
    direction: PortDirection
    required: bool = True
    data_type: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Port names must be non-empty.")
        if "." in self.name:
            raise ValueError("Port names cannot contain '.'.")

    @classmethod
    def input(
        cls,
        name: str,
        *,
        required: bool = True,
        data_type: str | None = None,
    ) -> "PortSpec":
        return cls(name=name, direction=PortDirection.INPUT, required=required, data_type=data_type)

    @classmethod
    def output(
        cls,
        name: str,
        *,
        data_type: str | None = None,
    ) -> "PortSpec":
        return cls(name=name, direction=PortDirection.OUTPUT, required=True, data_type=data_type)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    block_name: str
    time: float
    step_index: int
    dt: float
    parameters: Mapping[str, Any]
    discrete_state: Any = None
    continuous_state: Any = None


def _normalize_port_specs(
    port_specs: Sequence[PortSpec],
    expected_direction: PortDirection,
) -> tuple[PortSpec, ...]:
    normalized: list[PortSpec] = []
    seen: set[str] = set()
    for spec in port_specs:
        if spec.direction is not expected_direction:
            raise ValueError(
                f"Port '{spec.name}' must have direction {expected_direction.value!r}, "
                f"got {spec.direction.value!r}."
            )
        if spec.name in seen:
            raise ValueError(f"Duplicate port name {spec.name!r}.")
        seen.add(spec.name)
        normalized.append(spec)
    return tuple(normalized)


class Block:
    """Base class for user-defined blocks."""

    inputs: ClassVar[Sequence[PortSpec]] = ()
    outputs: ClassVar[Sequence[PortSpec]] = ()

    def __init__(
        self,
        *,
        inputs: Sequence[PortSpec] | None = None,
        outputs: Sequence[PortSpec] | None = None,
        direct_feedthrough: bool = True,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.input_ports = _normalize_port_specs(inputs or self.inputs, PortDirection.INPUT)
        self.output_ports = _normalize_port_specs(outputs or self.outputs, PortDirection.OUTPUT)
        self.direct_feedthrough = direct_feedthrough
        self.parameters = MappingProxyType(dict(parameters or {}))

    def get_input_spec(self, port_name: str) -> PortSpec | None:
        return next((spec for spec in self.input_ports if spec.name == port_name), None)

    def get_output_spec(self, port_name: str) -> PortSpec | None:
        return next((spec for spec in self.output_ports if spec.name == port_name), None)

    def initial_discrete_state(self) -> Any:
        return None

    def initial_continuous_state(self) -> Any:
        return None

    def output(self, ctx: ExecutionContext, inputs: Mapping[str, Any]) -> Any:
        if self.output_ports:
            raise NotImplementedError(f"{self.__class__.__name__}.output() must be implemented.")
        return {}


class DiscreteBlock(Block):
    """Base class for sampled blocks with discrete state."""

    def __init__(
        self,
        *,
        sample_time: float = 1.0,
        inputs: Sequence[PortSpec] | None = None,
        outputs: Sequence[PortSpec] | None = None,
        direct_feedthrough: bool = False,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if sample_time <= 0:
            raise ValueError("sample_time must be positive.")
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            direct_feedthrough=direct_feedthrough,
            parameters=parameters,
        )
        self.sample_time = float(sample_time)

    def update_state(self, ctx: ExecutionContext, inputs: Mapping[str, Any], state: Any) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.update_state() must be implemented.")


class ContinuousBlock(Block):
    """Base class for blocks with continuous state."""

    def __init__(
        self,
        *,
        inputs: Sequence[PortSpec] | None = None,
        outputs: Sequence[PortSpec] | None = None,
        direct_feedthrough: bool = False,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            direct_feedthrough=direct_feedthrough,
            parameters=parameters,
        )

    def derivative(self, ctx: ExecutionContext, inputs: Mapping[str, Any], state: Any) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.derivative() must be implemented.")
