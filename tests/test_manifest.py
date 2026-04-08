from __future__ import annotations

from pathlib import Path

import yaml

from pylink import (
    Block,
    PortSpec,
    SimulationConfig,
    Simulator,
    Subsystem,
    System,
    build_detail_manifest,
    build_graph_manifest,
    write_manifest_bundle,
)

from subsystem_helpers import FLOAT_SCALAR, build_hierarchical_closed_loop


class ExplicitSource(Block):
    """This docstring should not win over the explicit description."""

    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False, description="Explicitly described source block.")
        self.value = value

    def output(self, ctx, inputs):
        return float(self.value)


class DocstringGain(Block):
    """Gain block documented by its class docstring."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, gain: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.gain = gain

    def output(self, ctx, inputs):
        return self.gain * float(inputs["u"])


class PlainEcho(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return float(inputs["u"])


class DocstringConstant(Block):
    """Constant source documented only by a docstring."""

    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return float(self.value)


def build_manifest_demo_system() -> System:
    controller = Subsystem("controller", description="Reusable controller subsystem.")
    controller.add_block("gain", DocstringGain(2.0))
    controller.add_block("echo", PlainEcho())
    controller.connect("gain.y", "echo.u")
    controller.expose_input("u", "gain.u", spec=FLOAT_SCALAR)
    controller.expose_output("echo.y", "y", spec=FLOAT_SCALAR)

    system = System("manifest_demo")
    system.add_block("source", ExplicitSource(1.5))
    system.add_subsystem("controller", controller)
    system.connect("source.out", "controller.u")
    return system


def build_runtime_system() -> System:
    system = System("runtime_no_export")
    system.add_block("source", DocstringConstant(1.0))
    system.add_block("echo", PlainEcho())
    system.connect("source.out", "echo.u")
    return system


def test_graph_manifest_is_deterministic_and_preserves_local_topology():
    system = build_hierarchical_closed_loop()

    first = build_graph_manifest(system)
    second = build_graph_manifest(system)

    assert first == second
    assert first["manifest_kind"] == "pylink_graph"
    assert first["root"]["detail_ref"] == "detail/system.yaml"

    root = first["containers"][0]
    assert root["path"] == ""
    assert [child["path"] for child in root["children"]] == [
        "reference",
        "controller",
        "plant",
    ]
    assert root["connections"] == [
        {
            "source": "reference.out",
            "target": "controller.reference",
        },
        {
            "source": "plant.y",
            "target": "controller.measurement",
        },
        {
            "source": "controller.command",
            "target": "plant.u",
        },
    ]

    controller = next(item for item in first["containers"] if item["path"] == "controller")
    assert controller["children"] == [
        {
            "name": "error",
            "kind": "block",
            "path": "controller/error",
            "detail_ref": "detail/controller/error.yaml",
        },
        {
            "name": "gain",
            "kind": "block",
            "path": "controller/gain",
            "detail_ref": "detail/controller/gain.yaml",
        },
        {
            "name": "hold",
            "kind": "block",
            "path": "controller/hold",
            "detail_ref": "detail/controller/hold.yaml",
        },
    ]
    assert controller["connections"] == [
        {
            "source": "error.y",
            "target": "gain.u",
        },
        {
            "source": "gain.y",
            "target": "hold.u",
        },
    ]


def test_detail_manifest_exposes_sources_parameters_and_root_analysis():
    system = build_manifest_demo_system()

    source_detail = build_detail_manifest(system, path="source")
    controller_detail = build_detail_manifest(system, path="controller")
    root_detail = build_detail_manifest(
        system,
        config=SimulationConfig(start=0.0, stop=0.1, dt=0.1),
    )
    echo_detail = build_detail_manifest(system, path="controller/echo")

    assert source_detail["description"] == "Explicitly described source block."
    assert source_detail["description_origin"] == "explicit"
    assert source_detail["parameters"]["declared"] == {}
    assert source_detail["parameters"]["inferred"] == {"value": 1.5}
    assert source_detail["instance_source"]["file"].endswith("test_manifest.py")
    assert source_detail["implementation_source"]["methods"]["output"]["function"].endswith(
        "ExplicitSource.output"
    )

    assert controller_detail["description"] == "Reusable controller subsystem."
    assert controller_detail["description_origin"] == "explicit"
    assert controller_detail["children"] == [
        {
            "name": "gain",
            "kind": "block",
            "path": "controller/gain",
            "detail_ref": "detail/controller/gain.yaml",
        },
        {
            "name": "echo",
            "kind": "block",
            "path": "controller/echo",
            "detail_ref": "detail/controller/echo.yaml",
        },
    ]
    assert controller_detail["connections"][0]["source"] == "gain.y"
    assert controller_detail["connections"][0]["target"] == "echo.u"
    assert controller_detail["connections"][0]["instance_source"]["file"].endswith("test_manifest.py")
    assert controller_detail["exposed_inputs"][0]["targets"][0]["target"] == "gain.u"
    assert controller_detail["exposed_outputs"][0]["source"] == "echo.y"

    assert root_detail["kind"] == "system"
    assert root_detail["execution_order"] == ["source", "controller/gain", "controller/echo"]
    assert root_detail["time_grid_constraints"]["config"] == {
        "start": 0.0,
        "stop": 0.1,
        "dt": 0.1,
    }

    assert echo_detail["description"] is None
    assert echo_detail["description_origin"] == "missing"
    assert "stateless block" in echo_detail["auto_summary"]


def test_write_manifest_bundle_writes_shards_and_flags_stale_descriptions(tmp_path, monkeypatch):
    system = System("write_manifest")
    system.add_block("source", DocstringConstant(1.0))

    result = write_manifest_bundle(system, tmp_path)

    assert result.graph_path == tmp_path / "graph.yaml"
    assert result.detail_index_path == tmp_path / "detail" / "index.yaml"
    assert (tmp_path / "detail" / "system.yaml").exists()
    assert (tmp_path / "detail" / "source.yaml").exists()

    index = yaml.safe_load((tmp_path / "detail" / "index.yaml").read_text(encoding="utf-8"))
    source_detail = yaml.safe_load((tmp_path / "detail" / "source.yaml").read_text(encoding="utf-8"))

    assert index["entries"] == [
        {
            "name": "write_manifest",
            "kind": "system",
            "path": "",
            "detail_ref": "detail/system.yaml",
        },
        {
            "name": "source",
            "kind": "block",
            "path": "source",
            "detail_ref": "detail/source.yaml",
        },
    ]
    assert source_detail["description_status"] == "current"

    def new_output(self, ctx, inputs):
        return float(self.value) + 1.0

    monkeypatch.setattr(DocstringConstant, "output", new_output)
    write_manifest_bundle(system, tmp_path)
    refreshed_detail = yaml.safe_load((tmp_path / "detail" / "source.yaml").read_text(encoding="utf-8"))

    assert refreshed_detail["description_status"] == "review_recommended"


def test_simulation_paths_do_not_write_manifest_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    system = build_runtime_system()
    config = SimulationConfig(start=0.0, stop=0.1, dt=0.1)
    simulator = Simulator()

    report = simulator.validate(system, config)
    plan = simulator.compile(system)
    result = simulator.run(system, config)

    assert report.is_valid is True
    assert tuple(plan.block_order) == ("source", "echo")
    assert result.final_outputs["echo"]["y"] == 1.0
    assert list(tmp_path.rglob("*.yaml")) == []
    assert list(tmp_path.rglob("detail")) == []
