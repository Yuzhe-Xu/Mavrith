from __future__ import annotations

import pytest

from pylink import ModelValidationError, SimulationConfig, Simulator, Subsystem, System

from subsystem_helpers import FLOAT_SCALAR, Constant, build_passthrough_subsystem


def test_subsystem_name_conflicts_are_rejected():
    system = System()
    system.add_subsystem("loop", build_passthrough_subsystem())

    with pytest.raises(ModelValidationError) as exc_info:
        system.add_subsystem("loop", build_passthrough_subsystem())

    assert exc_info.value.code == "DUPLICATE_SUBSYSTEM_NAME"


def test_component_names_cannot_use_hierarchy_separator():
    system = System()

    with pytest.raises(ModelValidationError) as exc_info:
        system.add_subsystem("bad/name", build_passthrough_subsystem())

    assert exc_info.value.code == "INVALID_COMPONENT_NAME"


def test_duplicate_subsystem_output_name_is_rejected():
    subsystem = Subsystem("dup_output")
    subsystem.add_block("src", Constant(1.0))
    subsystem.expose_output("src.out", "y", spec=FLOAT_SCALAR)

    with pytest.raises(ModelValidationError) as exc_info:
        subsystem.expose_output("src.out", "y", spec=FLOAT_SCALAR)

    assert exc_info.value.code == "DUPLICATE_SUBSYSTEM_OUTPUT"


def test_empty_subsystem_is_reported_during_validation():
    system = System("empty_subsystem")
    system.add_subsystem("empty", Subsystem("empty"))

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "EMPTY_SUBSYSTEM"


def test_invalid_exposed_input_target_is_reported():
    subsystem = Subsystem("bad_input")
    subsystem.expose_input("u", "missing.u", spec=FLOAT_SCALAR)
    subsystem.add_block("src", Constant(1.0))
    subsystem.expose_output("src.out", "y", spec=FLOAT_SCALAR)

    system = System("invalid_exposed_input")
    system.add_subsystem("inner", subsystem)

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert report.diagnostics[0].code == "UNKNOWN_TARGET_COMPONENT"


def test_invalid_exposed_output_source_is_reported():
    subsystem = Subsystem("bad_output")
    subsystem.add_block("src", Constant(1.0))
    subsystem.expose_output("missing.y", "y", spec=FLOAT_SCALAR)

    system = System("invalid_exposed_output")
    system.add_subsystem("inner", subsystem)

    report = Simulator().validate(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert report.is_valid is False
    assert any(diagnostic.code == "UNKNOWN_SOURCE_COMPONENT" for diagnostic in report.diagnostics)
