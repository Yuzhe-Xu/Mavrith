from __future__ import annotations

from mavrith import SimulationConfig, Simulator, System

from subsystem_helpers import Constant, Echo, build_gain_hold_subsystem, build_hierarchical_closed_loop


def _collect_block_paths(node: dict[str, object]) -> list[str]:
    paths: list[str] = []
    for child in node["children"]:
        assert isinstance(child, dict)
        if child["kind"] == "block":
            paths.append(str(child["path"]))
        else:
            paths.extend(_collect_block_paths(child))
    return paths


def test_summary_exposes_hierarchy_tree_and_flat_view():
    report = Simulator().validate(
        build_hierarchical_closed_loop(),
        SimulationConfig(start=0.0, stop=1.0, dt=0.1),
    )
    summary = report.summary()

    assert summary["hierarchy"]["kind"] == "system"
    assert [child["name"] for child in summary["hierarchy"]["children"]] == [
        "reference",
        "controller",
        "plant",
    ]
    controller_node = summary["hierarchy"]["children"][1]
    assert controller_node["kind"] == "subsystem"
    assert [item["name"] for item in controller_node["exposed_inputs"]] == [
        "reference",
        "measurement",
    ]
    assert [item["name"] for item in controller_node["exposed_outputs"]] == ["command"]
    assert "controller/error" in _collect_block_paths(summary["hierarchy"])


def test_summary_is_deterministic_for_compile_and_validate():
    simulator = Simulator()
    system = build_hierarchical_closed_loop()
    config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)

    first_plan = simulator.compile(system)
    second_plan = simulator.compile(system)
    first_report = simulator.validate(system, config)
    second_report = simulator.validate(system, config)

    assert first_plan.summary() == second_plan.summary()
    assert first_report.summary() == second_report.summary()


def test_flat_block_list_matches_hierarchy_leaf_paths():
    system = System("summary_consistency")
    system.add_block("source", Constant(1.0))
    system.add_subsystem("controller", build_gain_hold_subsystem())
    system.add_block("sink", Echo())
    system.connect("source.out", "controller.u")
    system.connect("controller.y", "sink.u")

    summary = Simulator().validate(
        system,
        SimulationConfig(start=0.0, stop=0.2, dt=0.1),
    ).summary()

    flat_names = [item["name"] for item in summary["blocks"]]
    hierarchy_names = _collect_block_paths(summary["hierarchy"])

    assert flat_names == hierarchy_names
