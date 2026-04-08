from __future__ import annotations

import pytest

from mavrith import (
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


class Sink(Block):
    inputs = (PortSpec.input("inp"),)

    def __init__(self):
        super().__init__(outputs=(), direct_feedthrough=True)


class PassThrough(Block):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("y"),)

    def __init__(self):
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["u"]


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


class MissingStatePlant(ContinuousBlock):
    outputs = (PortSpec.output("x"),)

    def __init__(self):
        super().__init__(direct_feedthrough=False)

    def initial_continuous_state(self):
        return None

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return 0.0


class BrokenPair(Block):
    outputs = (
        PortSpec.output("left"),
        PortSpec.output("right"),
    )

    def __init__(self):
        super().__init__(direct_feedthrough=False)

    def output(self, ctx, inputs):
        return {"left": 1.0}


def test_validate_reports_endpoint_spelling_errors_with_context():
    system = System("endpoint_typos")
    system.add_block("src", Constant(1.0))
    system.add_block("sink", Sink())
    system.connect("src.missing", "sink.inp")

    report = Simulator().validate(system)

    assert report.is_valid is False
    assert [diagnostic.code for diagnostic in report.diagnostics] == [
        "UNKNOWN_SOURCE_PORT",
        "MISSING_REQUIRED_INPUT",
    ]
    assert report.diagnostics[0].endpoint == "src.missing"
    assert report.diagnostics[0].connection == "src.missing -> sink.inp"


def test_validate_reports_duplicate_input_connections():
    system = System("duplicate_input")
    system.add_block("left", Constant(1.0))
    system.add_block("right", Constant(2.0))
    system.add_block("sink", Sink())
    system.connect("left.out", "sink.inp")
    system.connect("right.out", "sink.inp")

    report = Simulator().validate(system)

    assert report.is_valid is False
    assert report.diagnostics[0].code == "DUPLICATE_INPUT_CONNECTION"
    assert report.diagnostics[0].block_name == "sink"
    assert report.diagnostics[0].port_name == "inp"


def test_validate_reports_incompatible_sample_time():
    system = System("sample_time")
    system.add_block("delta", Constant(1.0))
    system.add_block("counter", Counter(sample_time=0.15))
    system.connect("delta.out", "counter.delta")

    report = Simulator().validate(
        system,
        SimulationConfig(start=0.0, stop=0.3, dt=0.1),
    )

    assert report.is_valid is False
    assert report.diagnostics[0].code == "INCOMPATIBLE_SAMPLE_TIME"


def test_validate_reports_output_contract_mismatch():
    system = System("broken_outputs")
    system.add_block("broken", BrokenPair())

    report = Simulator().validate(system)

    assert report.is_valid is False
    assert report.diagnostics[0].code == "MISSING_DECLARED_OUTPUTS"
    assert report.diagnostics[0].block_name == "broken"


def test_compile_summary_is_deterministic():
    system = System("deterministic_compile")
    system.add_block("src", Constant(1.0))
    system.add_block("echo", PassThrough())
    system.connect("src.out", "echo.u")

    simulator = Simulator()
    first = simulator.compile(system)
    second = simulator.compile(system)

    assert first.summary() == second.summary()
    assert first.summary()["execution_order"] == ["src", "echo"]


def test_validate_report_is_deterministic():
    system = System("deterministic_validate")
    system.add_block("src", Constant(1.0))
    system.add_block("sink", Sink())
    system.connect("src.missing", "sink.inp")

    simulator = Simulator()
    first = simulator.validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))
    second = simulator.validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert first.to_dict() == second.to_dict()


def test_duplicate_block_name_error_exposes_stable_code():
    system = System()
    system.add_block("src", Constant(1.0))

    with pytest.raises(ModelValidationError) as exc_info:
        system.add_block("src", Constant(2.0))

    assert exc_info.value.code == "DUPLICATE_BLOCK_NAME"


def test_validate_reports_direct_feedthrough_loop():
    system = System("loop")
    system.add_block("a", PassThrough())
    system.add_block("b", PassThrough())
    system.connect("a.y", "b.u")
    system.connect("b.y", "a.u")

    report = Simulator().validate(system)

    assert report.is_valid is False
    assert report.diagnostics[-1].code == "ALGEBRAIC_LOOP"


def test_validate_reports_missing_continuous_initial_state():
    system = System("missing_state")
    system.add_block("plant", MissingStatePlant())

    report = Simulator().validate(system)

    assert report.is_valid is False
    assert report.diagnostics[0].code == "MISSING_CONTINUOUS_INITIAL_STATE"
