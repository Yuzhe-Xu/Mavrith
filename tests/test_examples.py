from __future__ import annotations

import pytest

from examples.closed_loop import build_system as build_closed_loop_system
from examples.cruise_control import CruiseRecorder, build_system as build_cruise_control_system
from examples.mass_spring_damper import MassSpringRecorder, build_system as build_mass_spring_system
from examples.vehicle_path_tracking import (
    TrajectoryRecorder,
    build_system as build_vehicle_path_tracking_system,
)
from examples.water_cooling import build_system as build_water_cooling_system
from pylink import SimulationConfig, Simulator


def test_closed_loop_example_is_valid_and_has_stable_summary():
    system = build_closed_loop_system()
    config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)

    report = Simulator().validate(system, config)

    assert report.is_valid is True
    assert report.summary()["execution_order"] == ["reference", "controller", "plant", "error"]
    assert [item["connection"] for item in report.summary()["connections"]] == [
        "reference.value -> error.setpoint",
        "plant.y -> error.measurement",
        "error.error -> controller.error",
        "controller.command -> plant.u",
    ]


def test_water_cooling_example_is_valid_and_has_stable_summary():
    system = build_water_cooling_system()
    config = SimulationConfig(start=0.0, stop=30.0, dt=0.01)

    report = Simulator().validate(system, config)

    assert report.is_valid is True
    assert report.summary()["execution_order"] == ["room", "cup", "delta"]
    assert report.summary()["stateful_blocks"] == {
        "discrete": [],
        "continuous": ["cup"],
    }


def test_vehicle_path_tracking_example_recovers_from_large_initial_error():
    system = build_vehicle_path_tracking_system()
    config = SimulationConfig(start=0.0, stop=25.0, dt=0.02)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = TrajectoryRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert recorder.path_errors[0] == pytest.approx(4.73863375370596, rel=1e-3)
    assert recorder.path_errors[-1] < 1.0
    assert recorder.path_errors[-1] < recorder.path_errors[0]
    assert recorder.speeds[-1] == pytest.approx(8.0, abs=0.35)
    assert max(recorder.steering_abs) <= 0.55 + 1e-9
    assert result.final_outputs["vehicle"]["path_error"] < 1.0


def test_cruise_control_example_handles_grade_disturbances_and_saturation():
    system = build_cruise_control_system()
    config = SimulationConfig(start=0.0, stop=25.0, dt=0.02)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = CruiseRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert min(recorder.speeds) >= -1e-9
    assert min(recorder.throttles) >= -1e-9
    assert max(recorder.throttles) <= 1.0 + 1e-9
    assert max(recorder.throttles) == pytest.approx(1.0, abs=1e-9)
    assert recorder.speeds[-1] == pytest.approx(recorder.references[-1], abs=0.8)
    assert result.final_outputs["vehicle"]["speed"] == pytest.approx(recorder.speeds[-1], abs=1e-9)


def test_mass_spring_damper_example_rejects_disturbance_and_respects_force_limit():
    system = build_mass_spring_system()
    config = SimulationConfig(start=0.0, stop=10.0, dt=0.005)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = MassSpringRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert max(abs(force) for force in recorder.forces) <= 6.0 + 1e-9
    assert max(recorder.positions) < 1.5
    assert recorder.positions[-1] == pytest.approx(1.0, abs=0.05)
    assert recorder.velocities[-1] == pytest.approx(0.0, abs=0.08)
    assert result.final_outputs["plant"]["position"] == pytest.approx(recorder.positions[-1], abs=1e-9)
