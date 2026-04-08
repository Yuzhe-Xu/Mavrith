from __future__ import annotations

import pytest

from mavrith import (
    Block,
    ModelValidationError,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    Simulator,
    System,
)


FLOAT_SCALAR = SignalSpec(dtype="float", shape=())
FLOAT_VECTOR2 = SignalSpec(dtype="float", shape=(2,))
INT_SCALAR = SignalSpec(dtype="int", shape=())


class Source(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class IntSource(Block):
    outputs = (PortSpec.output("out", spec=INT_SCALAR),)

    def __init__(self, value=1):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class UnspecifiedSource(Block):
    outputs = (PortSpec.output("out"),)

    def __init__(self, value):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Sink(Block):
    inputs = (PortSpec.input("inp", spec=FLOAT_SCALAR),)

    def __init__(self):
        super().__init__(outputs=(), direct_feedthrough=True)


class IntSink(Block):
    inputs = (PortSpec.input("inp", spec=INT_SCALAR),)

    def __init__(self):
        super().__init__(outputs=(), direct_feedthrough=True)


class VectorSink(Block):
    inputs = (PortSpec.input("inp", spec=FLOAT_VECTOR2),)

    def __init__(self):
        super().__init__(outputs=(), direct_feedthrough=True)


class WrongOutputType(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self):
        super().__init__(direct_feedthrough=False)

    def output(self, ctx, inputs):
        return True


class WrongOutputShape(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self):
        super().__init__(direct_feedthrough=False)

    def output(self, ctx, inputs):
        return [1.0, 2.0]


class WrongVectorOutputShape(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_VECTOR2),)

    def __init__(self):
        super().__init__(direct_feedthrough=False)

    def output(self, ctx, inputs):
        return [1.0, 2.0, 3.0]


def test_signal_spec_static_compatibility_accepts_matching_ports():
    system = System("matching")
    system.add_block("src", Source(1.5))
    system.add_block("sink", Sink())
    system.connect("src.out", "sink.inp")

    plan = Simulator().compile(system)

    assert plan.summary()["blocks"][0]["outputs"][0]["signal_spec"] == {
        "dtype": "float",
        "shape": (),
    }


def test_signal_spec_static_type_mismatch_is_rejected():
    system = System("type_mismatch")
    system.add_block("src", Source(1.5))
    system.add_block("sink", IntSink())
    system.connect("src.out", "sink.inp")

    with pytest.raises(ModelValidationError) as exc_info:
        Simulator().compile(system)

    assert exc_info.value.code == "INCOMPATIBLE_PORT_TYPE"


def test_signal_spec_static_shape_mismatch_is_rejected():
    system = System("shape_mismatch")
    system.add_block("src", Source(1.5))
    system.add_block("sink", VectorSink())
    system.connect("src.out", "sink.inp")

    with pytest.raises(ModelValidationError) as exc_info:
        Simulator().compile(system)

    assert exc_info.value.code == "INCOMPATIBLE_PORT_SHAPE"


def test_signal_spec_static_wildcard_allows_unspecified_ports():
    system = System("wildcard")
    system.add_block("src", UnspecifiedSource(1.5))
    system.add_block("sink", Sink())
    system.connect("src.out", "sink.inp")

    plan = Simulator().compile(system)

    assert plan.block_order == ("src", "sink")


def test_validate_reports_runtime_input_signal_mismatch_for_unspecified_source():
    system = System("runtime_input_mismatch")
    system.add_block("src", UnspecifiedSource([1, 2, 3]))
    system.add_block("sink", Sink())
    system.connect("src.out", "sink.inp")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "INPUT_TYPE_MISMATCH"
    assert report.diagnostics[0].connection == "src.out -> sink.inp"


def test_validate_reports_output_type_mismatch():
    system = System("output_type_mismatch")
    system.add_block("src", WrongOutputType())

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "OUTPUT_TYPE_MISMATCH"


def test_validate_reports_output_shape_mismatch_for_scalar_signal():
    system = System("output_shape_mismatch_scalar")
    system.add_block("src", WrongOutputShape())

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "OUTPUT_SHAPE_MISMATCH"


def test_validate_reports_output_shape_mismatch_for_vector_signal():
    system = System("output_shape_mismatch_vector")
    system.add_block("src", WrongVectorOutputShape())

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "OUTPUT_SHAPE_MISMATCH"


def test_signal_spec_summary_is_structured_and_deterministic():
    system = System("summary")
    system.add_block("src", Source(1.5))
    system.add_block("sink", Sink())
    system.connect("src.out", "sink.inp")
    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=0.1, dt=0.1)

    first_plan = simulator.compile(system)
    second_plan = simulator.compile(system)
    first_report = simulator.validate(system, config)
    second_report = simulator.validate(system, config)

    assert first_plan.summary() == second_plan.summary()
    assert first_report.summary() == second_report.summary()
    assert first_report.summary()["blocks"][1]["inputs"][0]["signal_spec"] == {
        "dtype": "float",
        "shape": (),
    }


def test_data_type_sugar_normalizes_to_signal_spec():
    spec = PortSpec.output("out", data_type="float")

    assert spec.signal_spec == SignalSpec(dtype="float")
    assert spec.data_type == "float"
