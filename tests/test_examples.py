from __future__ import annotations

import pytest

from examples.aircraft_pitch_digital import PitchRecorder, build_system as build_aircraft_pitch_system
from examples.closed_loop import build_system as build_closed_loop_system
from examples.cstr_temperature_control import CSTRRecorder, build_system as build_cstr_system
from examples.cruise_control import CruiseRecorder, build_system as build_cruise_control_system
from examples.inverted_pendulum_lqr import PendulumRecorder, build_system as build_inverted_pendulum_system
from examples.large_hierarchical_benchmark import BenchmarkCase, run_case
from examples.mass_spring_damper import MassSpringRecorder, build_system as build_mass_spring_system
from examples.quadruple_tank import QuadrupleTankRecorder, build_system as build_quadruple_tank_system
from examples.vehicle_path_tracking import (
    TrajectoryRecorder,
    build_system as build_vehicle_path_tracking_system,
)
from examples.water_cooling import build_system as build_water_cooling_system
from mavrith import SimulationConfig, Simulator


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


def test_aircraft_pitch_digital_example_tracks_pitch_reference():
    system = build_aircraft_pitch_system()
    config = SimulationConfig(start=0.0, stop=10.0, dt=0.01)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = PitchRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert recorder.theta[-1] == pytest.approx(0.2, abs=0.01)
    assert max(recorder.theta) < 0.23
    assert max(abs(value) for value in recorder.delta) <= 0.35 + 1e-9
    assert result.final_outputs["airframe/plant"]["theta"] == pytest.approx(recorder.theta[-1], abs=1e-9)


def test_inverted_pendulum_lqr_example_stabilizes_angle_and_cart():
    system = build_inverted_pendulum_system()
    config = SimulationConfig(start=0.0, stop=6.0, dt=0.01)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = PendulumRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert recorder.cart_positions[-1] == pytest.approx(0.2, abs=0.03)
    assert recorder.angles[-1] == pytest.approx(0.0, abs=0.01)
    assert max(abs(value) for value in recorder.angles) < 0.06
    assert max(abs(value) for value in recorder.forces) <= 10.0 + 1e-9
    assert result.final_outputs["plant/dynamics"]["cart_position"] == pytest.approx(
        recorder.cart_positions[-1],
        abs=1e-9,
    )


def test_cstr_temperature_control_example_cools_to_target_band():
    system = build_cstr_system()
    config = SimulationConfig(start=0.0, stop=5.0, dt=0.01)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = CSTRRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert recorder.temperatures[-1] == pytest.approx(305.0, abs=1.0)
    assert 0.9 <= recorder.concentrations[-1] <= 1.0
    assert min(recorder.coolant_temperatures) >= 250.0 - 1e-9
    assert max(recorder.coolant_temperatures) <= 320.0 + 1e-9
    assert result.final_outputs["reactor/plant"]["temperature"] == pytest.approx(
        recorder.temperatures[-1],
        abs=1e-9,
    )


def test_quadruple_tank_example_returns_lower_tanks_to_reference_window():
    system = build_quadruple_tank_system()
    config = SimulationConfig(start=0.0, stop=120.0, dt=0.5)
    simulator = Simulator()

    report = simulator.validate(system, config)

    assert report.is_valid is True

    recorder = QuadrupleTankRecorder()
    result = simulator.run(system, config, observer=recorder)

    assert recorder.h1[-1] == pytest.approx(12.4, abs=0.6)
    assert recorder.h2[-1] == pytest.approx(12.7, abs=0.6)
    assert min(recorder.v1) >= -1e-9
    assert min(recorder.v2) >= -1e-9
    assert max(recorder.v1) <= 10.0 + 1e-9
    assert max(recorder.v2) <= 10.0 + 1e-9
    assert result.final_outputs["plant/tanks"]["h1"] == pytest.approx(recorder.h1[-1], abs=1e-9)


@pytest.mark.performance
def test_large_hierarchical_benchmark_example_reports_scaling_metrics():
    small = run_case(BenchmarkCase(levels=3, width=5, leaf_blocks=4))
    medium = run_case(BenchmarkCase(levels=4, width=5, leaf_blocks=8))

    assert small["final_output"] == pytest.approx(1.0, abs=1e-12)
    assert medium["final_output"] == pytest.approx(1.0, abs=1e-12)
    assert small["compile_seconds"] > 0.0
    assert medium["compile_seconds"] > 0.0
    assert medium["flat_blocks"] > small["flat_blocks"]
