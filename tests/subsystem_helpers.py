from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mavrith import (
    Block,
    ContinuousBlock,
    DiscreteBlock,
    PortSpec,
    SignalSpec,
    Subsystem,
    System,
)


FLOAT_SCALAR = SignalSpec(dtype="float", shape=())
INT_SCALAR = SignalSpec(dtype="int", shape=())
VECTOR2 = SignalSpec(dtype="float", shape=(2,))


class Constant(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return float(self.value)


class IntConstant(Block):
    outputs = (PortSpec.output("out", spec=INT_SCALAR),)

    def __init__(self, value: int) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return int(self.value)


class VectorConstant(Block):
    outputs = (PortSpec.output("out", spec=VECTOR2),)

    def __init__(self, value: tuple[float, float]) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = np.asarray(value, dtype=float)

    def output(self, ctx, inputs):
        return self.value.copy()


class Echo(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return float(inputs["u"])


class VectorEcho(Block):
    inputs = (PortSpec.input("u", spec=VECTOR2),)
    outputs = (PortSpec.output("y", spec=VECTOR2),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return np.asarray(inputs["u"], dtype=float)


class Gain(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, gain: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.gain = gain

    def output(self, ctx, inputs):
        return self.gain * float(inputs["u"])


class Sum(Block):
    inputs = (
        PortSpec.input("a", spec=FLOAT_SCALAR),
        PortSpec.input("b", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return float(inputs["a"]) + float(inputs["b"])


class Difference(Block):
    inputs = (
        PortSpec.input("a", spec=FLOAT_SCALAR),
        PortSpec.input("b", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return float(inputs["a"]) - float(inputs["b"])


class Hold(DiscreteBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, sample_time: float) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        return float(inputs["u"])


class Integrator(ContinuousBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("x", spec=FLOAT_SCALAR),)

    def __init__(self, initial: float = 0.0) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return float(ctx.continuous_state)

    def derivative(self, ctx, inputs, state):
        return float(inputs["u"])


class FirstOrderPlant(ContinuousBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, initial: float = 0.0) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return float(ctx.continuous_state)

    def derivative(self, ctx, inputs, state):
        return -float(state) + float(inputs["u"])


class Passthrough(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return float(inputs["u"])


class SimpleRecorder:
    def __init__(self) -> None:
        self.outputs: list[dict[str, dict[str, object]]] = []

    def on_step(self, snapshot) -> None:
        self.outputs.append(dict(snapshot.outputs))


def build_passthrough_subsystem() -> Subsystem:
    subsystem = Subsystem("passthrough")
    subsystem.add_block("echo", Passthrough())
    subsystem.expose_input("u", "echo.u", spec=FLOAT_SCALAR)
    subsystem.expose_output("echo.y", "y", spec=FLOAT_SCALAR)
    return subsystem


def build_gain_hold_subsystem(*, gain: float = 2.0, sample_time: float = 0.1) -> Subsystem:
    subsystem = Subsystem("gain_hold")
    subsystem.add_block("gain", Gain(gain))
    subsystem.add_block("hold", Hold(sample_time=sample_time))
    subsystem.connect("gain.y", "hold.u")
    subsystem.expose_input("u", "gain.u", spec=FLOAT_SCALAR)
    subsystem.expose_output("hold.y", "y", spec=FLOAT_SCALAR)
    return subsystem


def build_nested_gain_hold_subsystem(levels: int, *, gain: float = 1.5, sample_time: float = 0.1) -> Subsystem:
    if levels <= 1:
        return build_gain_hold_subsystem(gain=gain, sample_time=sample_time)

    subsystem = Subsystem(f"nested_{levels}")
    subsystem.add_subsystem("inner", build_nested_gain_hold_subsystem(levels - 1, gain=gain, sample_time=sample_time))
    subsystem.expose_input("u", "inner.u", spec=FLOAT_SCALAR)
    subsystem.expose_output("inner.y", "y", spec=FLOAT_SCALAR)
    return subsystem


def build_hierarchical_closed_loop() -> System:
    controller = Subsystem("controller")
    controller.add_block("error", Difference())
    controller.add_block("gain", Gain(2.0))
    controller.add_block("hold", Hold(sample_time=0.1))
    controller.connect("error.y", "gain.u")
    controller.connect("gain.y", "hold.u")
    controller.expose_input("reference", "error.a", spec=FLOAT_SCALAR)
    controller.expose_input("measurement", "error.b", spec=FLOAT_SCALAR)
    controller.expose_output("hold.y", "command", spec=FLOAT_SCALAR)

    system = System("hierarchical_closed_loop")
    system.add_block("reference", Constant(1.0))
    system.add_subsystem("controller", controller)
    system.add_block("plant", FirstOrderPlant(initial=0.0))
    system.connect("reference.out", "controller.reference")
    system.connect("plant.y", "controller.measurement")
    system.connect("controller.command", "plant.u")
    return system


def build_flat_closed_loop() -> System:
    system = System("hierarchical_closed_loop")
    system.add_block("reference", Constant(1.0))
    system._add_flat_block("controller/error", Difference())
    system._add_flat_block("controller/gain", Gain(2.0))
    system._add_flat_block("controller/hold", Hold(sample_time=0.1))
    system.add_block("plant", FirstOrderPlant(initial=0.0))
    system.connect("reference.out", "controller/error.a")
    system.connect("plant.y", "controller/error.b")
    system.connect("controller/error.y", "controller/gain.u")
    system.connect("controller/gain.y", "controller/hold.u")
    system.connect("controller/hold.y", "plant.u")
    return system


def build_series_subsystem(*, segments: int, child: Subsystem) -> Subsystem:
    subsystem = Subsystem(f"series_{segments}")
    previous_output = ""
    for index in range(segments):
        child_name = f"stage_{index}"
        subsystem.add_subsystem(child_name, child)
        if index == 0:
            subsystem.expose_input("u", f"{child_name}.u", spec=FLOAT_SCALAR)
        else:
            subsystem.connect(previous_output, f"{child_name}.u")
        previous_output = f"{child_name}.y"
    subsystem.expose_output(previous_output, "y", spec=FLOAT_SCALAR)
    return subsystem


def build_hierarchical_chain(*, levels: int, width: int, leaf_blocks: int) -> Subsystem:
    if levels <= 1:
        leaf = Subsystem(f"leaf_{leaf_blocks}")
        for index in range(leaf_blocks):
            block_name = f"gain_{index}"
            leaf.add_block(block_name, Passthrough())
            if index == 0:
                leaf.expose_input("u", f"{block_name}.u", spec=FLOAT_SCALAR)
            else:
                leaf.connect(f"gain_{index - 1}.y", f"{block_name}.u")
        leaf.expose_output(f"gain_{leaf_blocks - 1}.y", "y", spec=FLOAT_SCALAR)
        return leaf

    child = build_hierarchical_chain(levels=levels - 1, width=width, leaf_blocks=leaf_blocks)
    return build_series_subsystem(segments=width, child=child)


@dataclass(frozen=True, slots=True)
class PerformanceCase:
    levels: int
    width: int
    leaf_blocks: int

    @property
    def leaf_count(self) -> int:
        return self.leaf_blocks * (self.width ** (self.levels - 1))
