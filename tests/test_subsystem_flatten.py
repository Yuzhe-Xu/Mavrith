from __future__ import annotations

from pylink import SimulationConfig, Simulator, Subsystem, System

from subsystem_helpers import (
    FLOAT_SCALAR,
    Constant,
    Echo,
    Gain,
    Sum,
    build_flat_closed_loop,
    build_gain_hold_subsystem,
    build_hierarchical_closed_loop,
    build_nested_gain_hold_subsystem,
    build_passthrough_subsystem,
)


def test_single_layer_subsystem_flattens_to_leaf_paths():
    system = System("single_layer")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("controller", build_gain_hold_subsystem())
    system.add_block("sink", Echo())
    system.connect("source.out", "controller.u")
    system.connect("controller.y", "sink.u")

    plan = Simulator().compile(system)

    assert tuple(plan.system.blocks) == ("source", "controller/gain", "controller/hold", "sink")
    assert [item["connection"] for item in plan.summary()["connections"]] == [
        "controller/gain.y -> controller/hold.u",
        "source.out -> controller/gain.u",
        "controller/hold.y -> sink.u",
    ]


def test_three_level_nesting_has_deterministic_flat_names():
    system = System("three_levels")
    system.add_block("source", Constant(2.0))
    system.add_subsystem("wrapper", build_nested_gain_hold_subsystem(3))
    system.add_block("sink", Echo())
    system.connect("source.out", "wrapper.u")
    system.connect("wrapper.y", "sink.u")

    plan = Simulator().compile(system)

    assert tuple(plan.system.blocks) == (
        "source",
        "wrapper/inner/inner/gain",
        "wrapper/inner/inner/hold",
        "sink",
    )


def test_subsystem_input_fanout_expands_to_multiple_leaf_connections():
    fanout = Subsystem("fanout")
    fanout.add_block("left", Echo())
    fanout.add_block("right", Gain(3.0))
    fanout.add_block("sum", Sum())
    fanout.connect("left.y", "sum.a")
    fanout.connect("right.y", "sum.b")
    fanout.expose_input("u", "left.u", spec=FLOAT_SCALAR)
    fanout.expose_input("u", "right.u", spec=FLOAT_SCALAR)
    fanout.expose_output("sum.y", "y", spec=FLOAT_SCALAR)

    system = System("fanout_root")
    system.add_block("source", Constant(2.0))
    system.add_subsystem("fan", fanout)
    system.add_block("sink", Echo())
    system.connect("source.out", "fan.u")
    system.connect("fan.y", "sink.u")

    plan = Simulator().compile(system)
    connections = [item["connection"] for item in plan.summary()["connections"]]

    assert "source.out -> fan/left.u" in connections
    assert "source.out -> fan/right.u" in connections
    assert len(plan.fanout[("source", "out")]) == 2


def test_nested_subsystem_output_binding_routes_through_child_boundary():
    outer = Subsystem("outer")
    outer.add_subsystem("inner", build_passthrough_subsystem())
    outer.expose_input("u", "inner.u", spec=FLOAT_SCALAR)
    outer.expose_output("inner.y", "y", spec=FLOAT_SCALAR)

    system = System("nested_output")
    system.add_block("source", Constant(4.0))
    system.add_subsystem("outer", outer)
    system.add_block("sink", Echo())
    system.connect("source.out", "outer.u")
    system.connect("outer.y", "sink.u")

    plan = Simulator().compile(system)

    assert [item["connection"] for item in plan.summary()["connections"]] == [
        "source.out -> outer/inner/echo.u",
        "outer/inner/echo.y -> sink.u",
    ]


def test_hierarchical_and_manual_flat_models_compile_to_same_graph():
    hierarchical_plan = Simulator().compile(build_hierarchical_closed_loop())
    manual_plan = Simulator().compile(build_flat_closed_loop())

    assert [item["name"] for item in hierarchical_plan.summary()["blocks"]] == [
        item["name"] for item in manual_plan.summary()["blocks"]
    ]
    assert sorted(item["connection"] for item in hierarchical_plan.summary()["connections"]) == sorted(
        item["connection"] for item in manual_plan.summary()["connections"]
    )


def test_hierarchical_and_manual_flat_models_run_to_same_result():
    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)

    hierarchical = simulator.run(build_hierarchical_closed_loop(), config)
    manual = simulator.run(build_flat_closed_loop(), config)

    assert hierarchical.final_continuous_states == manual.final_continuous_states
    assert hierarchical.final_discrete_states == manual.final_discrete_states
    assert hierarchical.final_outputs == manual.final_outputs
