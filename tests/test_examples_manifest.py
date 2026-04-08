from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from examples.closed_loop import build_system as build_closed_loop_system
from examples.multirate_offset_priority import build_system as build_multirate_offset_priority_system
from examples.water_cooling import build_system as build_water_cooling_system
from pylink import (
    SimulationConfig,
    build_detail_manifest,
    build_graph_manifest,
    write_manifest_bundle,
)


@pytest.mark.parametrize(
    ("name", "builder", "config", "component_path", "expected_declared", "expected_execution_order"),
    [
        (
            "closed_loop",
            build_closed_loop_system,
            SimulationConfig(start=0.0, stop=1.0, dt=0.1),
            "controller",
            {"gain": 2.0},
            ["reference", "controller", "plant", "error"],
        ),
        (
            "water_cooling",
            build_water_cooling_system,
            SimulationConfig(start=0.0, stop=30.0, dt=0.01),
            "cup",
            {"initial_temperature": 80.0, "cooling_rate": 0.1},
            ["room", "cup", "delta"],
        ),
        (
            "multirate_offset_priority",
            build_multirate_offset_priority_system,
            SimulationConfig(start=0.0, stop=0.4, dt=0.1),
            "gain",
            {"gain": 10.0},
            ["source", "same_time_capture", "offset_capture", "plant", "gain"],
        ),
    ],
)
def test_example_manifests_build_in_memory(
    name,
    builder,
    config,
    component_path,
    expected_declared,
    expected_execution_order,
):
    system = builder()

    graph = build_graph_manifest(system)
    component_detail = build_detail_manifest(system, path=component_path)
    root_detail = build_detail_manifest(system, config=config)

    assert graph["manifest_kind"] == "pylink_graph"
    assert graph["system_name"] == name
    assert graph["root"]["detail_ref"] == "detail/system.yaml"
    assert graph["containers"][0]["path"] == ""

    assert component_detail["path"] == component_path
    assert component_detail["parameters"]["declared"] == expected_declared
    assert component_detail["description"] is not None
    assert component_detail["description_origin"] == "explicit"
    assert component_detail["description_status"] == "current"
    assert component_detail["instance_source"]["file"].endswith(".py")

    assert root_detail["kind"] == "system"
    assert root_detail["execution_order"] == expected_execution_order
    assert root_detail["time_grid_constraints"]["config"] == {
        "start": config.start,
        "stop": config.stop,
        "dt": config.dt,
    }
    if name == "multirate_offset_priority":
        assert len(root_detail["rate_groups"]) == 3
        assert {item["classification"] for item in root_detail["cross_rate_connections"]} == {
            "slow-to-fast",
            "same-period-different-offset",
            "fast-to-slow",
            "same-rate",
        }


@pytest.mark.parametrize(
    ("name", "builder", "component_path"),
    [
        ("closed_loop", build_closed_loop_system, "controller"),
        ("water_cooling", build_water_cooling_system, "cup"),
        ("multirate_offset_priority", build_multirate_offset_priority_system, "gain"),
    ],
)
def test_example_manifest_bundle_writes_expected_yaml(name, builder, component_path, tmp_path):
    system = builder()

    out_dir = tmp_path / name
    result = write_manifest_bundle(system, out_dir)

    assert result.graph_path == out_dir / "graph.yaml"
    assert result.detail_index_path == out_dir / "detail" / "index.yaml"

    graph_yaml = yaml.safe_load(result.graph_path.read_text(encoding="utf-8"))
    index_yaml = yaml.safe_load(result.detail_index_path.read_text(encoding="utf-8"))
    component_path_obj = out_dir / "detail" / Path(f"{component_path}.yaml")
    component_yaml = yaml.safe_load(component_path_obj.read_text(encoding="utf-8"))

    assert graph_yaml["system_name"] == name
    assert index_yaml["root_detail_ref"] == "detail/system.yaml"
    assert any(item["path"] == component_path for item in index_yaml["entries"])
    assert component_yaml["path"] == component_path
    assert component_yaml["description_status"] == "current"
