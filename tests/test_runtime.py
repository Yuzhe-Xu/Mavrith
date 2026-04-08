from __future__ import annotations

import math

import pytest

from pylink import (
    Block,
    ContinuousBlock,
    DiscreteBlock,
    ModelValidationError,
    PortSpec,
    SimulationConfig,
    Simulator,
    System,
)


class Constant(Block):
    outputs = (PortSpec.output("out"),)

    def __init__(self, value):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Echo(Block):
    inputs = (PortSpec.input("inp"),)
    outputs = (PortSpec.output("out"),)

    def __init__(self):
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["inp"]


class Counter(DiscreteBlock):
    inputs = (PortSpec.input("delta"),)
    outputs = (PortSpec.output("count"),)

    def __init__(self, sample_time):
        super().__init__(sample_time=sample_time, direct_feedthrough=False)

    def initial_discrete_state(self):
        return 0

    def output(self, ctx, inputs):
        return ctx.discrete_state

    def update_state(self, ctx, inputs, state):
        return state + inputs["delta"]


class Decay(ContinuousBlock):
    outputs = (PortSpec.output("x"),)

    def __init__(self, initial):
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return -state


class DrivenIntegrator(ContinuousBlock):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("x"),)

    def __init__(self, initial=0.0):
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return inputs["u"]


def test_stateless_signal_propagation():
    system = System()
    system.add_block("src", Constant(42))
    system.add_block("echo", Echo())
    system.connect("src.out", "echo.inp")

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=0.2, dt=0.1))

    assert result.final_outputs["echo"]["out"] == 42


def test_discrete_block_updates_on_sample_hits():
    system = System()
    system.add_block("delta", Constant(1))
    system.add_block("counter", Counter(sample_time=0.2))
    system.connect("delta.out", "counter.delta")

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=0.4, dt=0.1))

    assert result.time_points == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4))
    assert result.final_discrete_states["counter"] == 3
    assert result.final_outputs["counter"]["count"] == 3


def test_continuous_block_is_integrated_with_scipy():
    system = System()
    system.add_block("plant", Decay(initial=1.0))

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=1.0, dt=0.1))

    assert result.final_continuous_states["plant"] == pytest.approx(math.exp(-1.0), rel=1e-3)


def test_mixed_system_runs_with_discrete_and_continuous_blocks():
    system = System()
    system.add_block("delta", Constant(1.0))
    system.add_block("counter", Counter(sample_time=0.1))
    system.add_block("plant", DrivenIntegrator(initial=0.0))
    system.connect("delta.out", "counter.delta")
    system.connect("counter.count", "plant.u")

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=0.2, dt=0.1))

    assert result.final_discrete_states["counter"] == 3
    assert result.final_continuous_states["plant"] == pytest.approx(0.3, rel=1e-3)


def test_incompatible_sample_time_is_rejected():
    system = System()
    system.add_block("delta", Constant(1))
    system.add_block("counter", Counter(sample_time=0.15))
    system.connect("delta.out", "counter.delta")

    with pytest.raises(ModelValidationError):
        Simulator().run(system, SimulationConfig(start=0.0, stop=0.3, dt=0.1))


def test_repeat_runs_are_deterministic():
    system = System()
    system.add_block("plant", Decay(initial=1.0))
    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=0.5, dt=0.1)

    first = simulator.run(system, config)
    second = simulator.run(system, config)

    assert first.final_continuous_states == second.final_continuous_states
    assert first.final_outputs == second.final_outputs
