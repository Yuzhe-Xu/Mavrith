from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence


class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


_VALID_SIGNAL_DTYPES = frozenset({"bool", "int", "float", "complex", "object"})


@dataclass(frozen=True, slots=True)
class SignalSpec:
    dtype: str | None = None
    shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.dtype is not None and self.dtype not in _VALID_SIGNAL_DTYPES:
            valid = ", ".join(sorted(_VALID_SIGNAL_DTYPES))
            raise ValueError(f"Unsupported signal dtype {self.dtype!r}. Expected one of: {valid}.")

        normalized_shape = self.shape
        if normalized_shape is not None:
            normalized_shape = tuple(normalized_shape)
            if len(normalized_shape) > 2:
                raise ValueError("Signal shapes may describe only scalars, vectors, or matrices.")
            for dimension in normalized_shape:
                if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                    raise ValueError("Signal shape dimensions must be positive integers.")
            object.__setattr__(self, "shape", normalized_shape)

    @property
    def is_specified(self) -> bool:
        return self.dtype is not None or self.shape is not None

    def summary(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": self.shape,
        }


@dataclass(frozen=True, slots=True)
class PortSpec:
    name: str
    direction: PortDirection
    required: bool = True
    signal_spec: SignalSpec = field(default_factory=SignalSpec)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Port names must be non-empty.")
        if "." in self.name:
            raise ValueError("Port names cannot contain '.'.")
        if self.signal_spec is None:
            object.__setattr__(self, "signal_spec", SignalSpec())
        elif not isinstance(self.signal_spec, SignalSpec):
            raise TypeError("signal_spec must be a SignalSpec instance.")

    @classmethod
    def input(
        cls,
        name: str,
        *,
        required: bool = True,
        spec: SignalSpec | None = None,
        data_type: str | None = None,
    ) -> "PortSpec":
        return cls(
            name=name,
            direction=PortDirection.INPUT,
            required=required,
            signal_spec=_coerce_signal_spec(spec=spec, data_type=data_type),
        )

    @classmethod
    def output(
        cls,
        name: str,
        *,
        spec: SignalSpec | None = None,
        data_type: str | None = None,
    ) -> "PortSpec":
        return cls(
            name=name,
            direction=PortDirection.OUTPUT,
            required=True,
            signal_spec=_coerce_signal_spec(spec=spec, data_type=data_type),
        )

    @property
    def data_type(self) -> str | None:
        return self.signal_spec.dtype

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.signal_spec.shape


def _coerce_signal_spec(*, spec: SignalSpec | None, data_type: str | None) -> SignalSpec:
    if spec is not None and data_type is not None:
        raise ValueError("Use either spec=SignalSpec(...) or data_type=..., not both.")
    if spec is not None:
        if not isinstance(spec, SignalSpec):
            raise TypeError("spec must be a SignalSpec instance.")
        return spec
    if data_type is not None:
        return SignalSpec(dtype=data_type)
    return SignalSpec()


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
