from __future__ import annotations

import pytest

from pylink import AlgebraicLoopError, Block, ContinuousBlock, PortSpec, Simulator, System


class Source(Block):
    outputs = (PortSpec.output("y"),)

    def __init__(self, value=1.0):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class PassThrough(Block):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("y"),)

    def __init__(self):
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["u"]


class Integrator(ContinuousBlock):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("x"),)

    def __init__(self):
        super().__init__(direct_feedthrough=False)

    def initial_continuous_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return inputs["u"]


def test_direct_feedthrough_order_is_topological():
    system = System()
    system.add_block("src", Source())
    system.add_block("a", PassThrough())
    system.add_block("b", PassThrough())
    system.add_block("c", Integrator())
    system.connect("src.y", "a.u")
    system.connect("a.y", "b.u")
    system.connect("b.y", "c.u")

    plan = Simulator().compile(system)

    assert plan.block_order.index("src") < plan.block_order.index("a")
    assert plan.block_order.index("a") < plan.block_order.index("b")


def test_algebraic_loop_is_rejected():
    system = System()
    system.add_block("a", PassThrough())
    system.add_block("b", PassThrough())
    system.connect("a.y", "b.u")
    system.connect("b.y", "a.u")

    with pytest.raises(AlgebraicLoopError):
        Simulator().compile(system)


def test_feedback_through_stateful_block_is_allowed():
    system = System()
    system.add_block("sum", PassThrough())
    system.add_block("plant", Integrator())
    system.connect("plant.x", "sum.u")
    system.connect("sum.y", "plant.u")

    plan = Simulator().compile(system)

    assert set(plan.block_order) == {"sum", "plant"}
