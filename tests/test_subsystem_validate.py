from __future__ import annotations

import pytest

from mavrith import AlgebraicLoopError, Block, PortSpec, SimulationConfig, Simulator, Subsystem, System

from subsystem_helpers import (
    FLOAT_SCALAR,
    VECTOR2,
    Constant,
    Echo,
    Hold,
    IntConstant,
    Passthrough,
    VectorConstant,
    VectorEcho,
    build_passthrough_subsystem,
)


class OptionalEcho(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR, required=False),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        value = inputs["u"]
        return 0.0 if value is None else float(value)


def test_missing_required_subsystem_input_is_reported_on_boundary():
    system = System("missing_subsystem_input")
    system.add_subsystem("inner", build_passthrough_subsystem())

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "MISSING_REQUIRED_SUBSYSTEM_INPUT"
    assert report.diagnostics[0].endpoint == "inner.u"


def test_optional_subsystem_input_can_be_left_unconnected():
    optional = Subsystem("optional")
    optional.add_block("echo", OptionalEcho())
    optional.expose_input("u", "echo.u", spec=FLOAT_SCALAR, required=False)
    optional.expose_output("echo.y", "y", spec=FLOAT_SCALAR)

    system = System("optional_subsystem_input")
    system.add_subsystem("inner", optional)

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is True


def test_unknown_subsystem_port_is_reported():
    system = System("unknown_subsystem_port")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("inner", build_passthrough_subsystem())
    system.connect("source.out", "inner.missing")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "UNKNOWN_SUBSYSTEM_INPUT"


def test_cross_boundary_signal_type_mismatch_is_rejected():
    system = System("boundary_type_mismatch")
    system.add_block("source", IntConstant(1))
    system.add_subsystem("inner", build_passthrough_subsystem())
    system.connect("source.out", "inner.u")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "INCOMPATIBLE_PORT_TYPE"


def test_cross_boundary_signal_shape_mismatch_is_rejected():
    vector_subsystem = Subsystem("vector")
    vector_subsystem.add_block("echo", VectorEcho())
    vector_subsystem.expose_input("u", "echo.u", spec=VECTOR2)
    vector_subsystem.expose_output("echo.y", "y", spec=VECTOR2)

    system = System("boundary_shape_mismatch")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("inner", vector_subsystem)
    system.connect("source.out", "inner.u")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "INCOMPATIBLE_PORT_SHAPE"


def test_duplicate_external_connections_to_subsystem_input_are_rejected():
    system = System("duplicate_subsystem_input")
    system.add_block("left", Constant(1.0))
    system.add_block("right", Constant(2.0))
    system.add_subsystem("inner", build_passthrough_subsystem())
    system.connect("left.out", "inner.u")
    system.connect("right.out", "inner.u")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert any(diagnostic.code == "DUPLICATE_INPUT_CONNECTION" for diagnostic in report.diagnostics)


def test_internal_connection_and_exposed_input_conflict_is_rejected():
    conflicting = Subsystem("conflicting")
    conflicting.add_block("src", Constant(1.0))
    conflicting.add_block("echo", Echo())
    conflicting.connect("src.out", "echo.u")
    conflicting.expose_input("u", "echo.u", spec=FLOAT_SCALAR)
    conflicting.expose_output("echo.y", "y", spec=FLOAT_SCALAR)

    system = System("conflict_root")
    system.add_block("external", Constant(2.0))
    system.add_subsystem("inner", conflicting)
    system.connect("external.out", "inner.u")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert any(diagnostic.code == "DUPLICATE_INPUT_CONNECTION" for diagnostic in report.diagnostics)


def test_cross_subsystem_algebraic_loop_is_detected():
    system = System("subsystem_loop")
    system.add_subsystem("left", build_passthrough_subsystem())
    system.add_subsystem("right", build_passthrough_subsystem())
    system.connect("left.y", "right.u")
    system.connect("right.y", "left.u")

    with pytest.raises(AlgebraicLoopError):
        Simulator().compile(system)


def test_nested_discrete_sample_time_constraint_is_enforced():
    sampled = Subsystem("sampled")
    sampled.add_block("hold", Hold(sample_time=0.15))
    sampled.expose_input("u", "hold.u", spec=FLOAT_SCALAR)
    sampled.expose_output("hold.y", "y", spec=FLOAT_SCALAR)

    system = System("sample_time_nested")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("controller", sampled)
    system.add_block("sink", Echo())
    system.connect("source.out", "controller.u")
    system.connect("controller.y", "sink.u")

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.3, dt=0.1))

    assert report.is_valid is False
    assert any(diagnostic.code == "INCOMPATIBLE_SAMPLE_TIME" for diagnostic in report.diagnostics)


def test_parent_cannot_weaken_required_child_boundary():
    child = build_passthrough_subsystem()
    parent = Subsystem("parent")
    parent.add_subsystem("child", child)
    parent.expose_input("u", "child.u", spec=FLOAT_SCALAR, required=False)
    parent.expose_output("child.y", "y", spec=FLOAT_SCALAR)

    system = System("requiredness")
    system.add_subsystem("parent", parent)

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert any(
        diagnostic.code == "INCOMPATIBLE_SUBSYSTEM_REQUIREDNESS" for diagnostic in report.diagnostics
    )
