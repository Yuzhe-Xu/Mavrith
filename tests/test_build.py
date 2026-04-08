from __future__ import annotations

import pytest

from mavrith import Block, ModelValidationError, PortSpec, Simulator, System


class Source(Block):
    outputs = (PortSpec.output("out"),)

    def __init__(self, value):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Sink(Block):
    inputs = (PortSpec.input("inp"),)

    def __init__(self):
        super().__init__(outputs=(), direct_feedthrough=True)


def test_duplicate_block_names_are_rejected():
    system = System()
    system.add_block("src", Source(1))
    with pytest.raises(ModelValidationError):
        system.add_block("src", Source(2))


def test_duplicate_input_connection_is_rejected():
    system = System()
    system.add_block("src", Source(1))
    system.add_block("sink", Sink())
    system.connect("src.out", "sink.inp")
    system.connect("src.out", "sink.inp")

    with pytest.raises(ModelValidationError):
        Simulator().compile(system)


def test_missing_required_input_is_rejected():
    system = System()
    system.add_block("sink", Sink())

    with pytest.raises(ModelValidationError):
        Simulator().compile(system)


def test_output_fanout_is_allowed():
    system = System()
    system.add_block("src", Source(1))
    system.add_block("left", Sink())
    system.add_block("right", Sink())
    system.connect("src.out", "left.inp")
    system.connect("src.out", "right.inp")

    plan = Simulator().compile(system)

    assert len(plan.fanout[("src", "out")]) == 2
