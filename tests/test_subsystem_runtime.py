from __future__ import annotations

import pytest

from mavrith import SimulationConfig, Simulator, Subsystem, System

from subsystem_helpers import (
    FLOAT_SCALAR,
    Constant,
    Echo,
    Integrator,
    SimpleRecorder,
    build_flat_closed_loop,
    build_gain_hold_subsystem,
    build_hierarchical_closed_loop,
    build_nested_gain_hold_subsystem,
)


def test_hierarchical_continuous_and_discrete_system_runs():
    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)

    report = simulator.validate(build_hierarchical_closed_loop(), config)
    result = simulator.run(build_hierarchical_closed_loop(), config)

    assert report.is_valid is True
    assert result.final_outputs["plant"]["y"] == pytest.approx(0.6435457229360751, rel=1e-6)


def test_multi_rate_nested_control_remains_valid():
    inner = build_gain_hold_subsystem(gain=1.0, sample_time=0.1)
    outer = Subsystem("outer")
    outer.add_subsystem("inner", inner)
    outer.expose_input("u", "inner.u", spec=FLOAT_SCALAR)
    outer.expose_output("inner.y", "y", spec=FLOAT_SCALAR)

    system = System("multi_rate")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("controller", outer)
    system.add_block("plant", Integrator(initial=0.0))
    system.connect("source.out", "controller.u")
    system.connect("controller.y", "plant.u")

    config = SimulationConfig(start=0.0, stop=0.3, dt=0.1)
    simulator = Simulator()
    report = simulator.validate(system, config)
    result = simulator.run(system, config)

    assert report.is_valid is True
    assert result.final_continuous_states["plant"] == pytest.approx(0.3, abs=1e-6)


def test_deeply_nested_runs_are_deterministic():
    system = System("deep_nested")
    system.add_block("source", Constant(2.0))
    system.add_subsystem("chain", build_nested_gain_hold_subsystem(5, gain=1.0, sample_time=0.1))
    system.add_block("plant", Integrator(initial=0.0))
    system.connect("source.out", "chain.u")
    system.connect("chain.y", "plant.u")

    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=0.2, dt=0.1)

    first = simulator.run(system, config)
    second = simulator.run(system, config)

    assert first.final_continuous_states == second.final_continuous_states
    assert first.final_outputs == second.final_outputs


def test_observer_receives_flat_leaf_paths_for_hierarchical_models():
    system = System("observer_hierarchy")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("controller", build_gain_hold_subsystem(gain=2.0, sample_time=0.1))
    system.add_block("sink", Echo())
    system.connect("source.out", "controller.u")
    system.connect("controller.y", "sink.u")

    recorder = SimpleRecorder()
    result = Simulator().run(
        system,
        SimulationConfig(start=0.0, stop=0.2, dt=0.1),
        observer=recorder,
    )

    assert "controller/hold" in recorder.outputs[-1]
    assert result.final_outputs["sink"]["y"] == pytest.approx(2.0, abs=1e-9)


def test_hierarchical_and_manual_flat_results_match():
    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)

    hierarchical = simulator.run(build_hierarchical_closed_loop(), config)
    flat = simulator.run(build_flat_closed_loop(), config)

    assert hierarchical.final_outputs == flat.final_outputs
    assert hierarchical.final_discrete_states == flat.final_discrete_states
    assert hierarchical.final_continuous_states == flat.final_continuous_states
