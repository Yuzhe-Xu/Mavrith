from __future__ import annotations

import pytest

from pylink import (
    Block,
    ContinuousBlock,
    DiscreteBlock,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    Simulator,
    StepSnapshot,
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class Recorder:
    def __init__(self, block_name: str, port_name: str) -> None:
        self.block_name = block_name
        self.port_name = port_name
        self.samples: list[tuple[float, float]] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.samples.append((snapshot.time, float(snapshot.outputs[self.block_name][self.port_name])))


class Constant(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Gain(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, gain: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.gain = gain

    def output(self, ctx, inputs):
        return self.gain * float(inputs["u"])


class Incrementer(DiscreteBlock):
    outputs = (PortSpec.output("count", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        *,
        sample_time: float,
        offset: float = 0.0,
        priority: int | None = None,
    ) -> None:
        super().__init__(
            sample_time=sample_time,
            offset=offset,
            priority=priority,
            direct_feedthrough=False,
        )

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        return float(state) + 1.0


class Capture(DiscreteBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        *,
        sample_time: float,
        offset: float = 0.0,
        priority: int | None = None,
    ) -> None:
        super().__init__(
            sample_time=sample_time,
            offset=offset,
            priority=priority,
            direct_feedthrough=False,
        )

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        return float(inputs["u"])


class Integrator(ContinuousBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("x", spec=FLOAT_SCALAR),)

    def __init__(self, initial: float = 0.0) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return float(ctx.continuous_state)

    def derivative(self, ctx, inputs, state):
        return float(inputs["u"])


def test_discrete_block_validates_offset_and_priority():
    with pytest.raises(ValueError):
        Incrementer(sample_time=0.1, offset=-0.01)

    with pytest.raises(ValueError):
        Incrementer(sample_time=0.1, offset=0.1)

    with pytest.raises(TypeError):
        Incrementer(sample_time=0.1, priority=1.5)  # type: ignore[arg-type]


def test_offset_hits_follow_absolute_zero_time():
    system = System("offset_hits")
    system.add_block("counter", Incrementer(sample_time=0.1, offset=0.02))

    recorder = Recorder("counter", "count")
    result = Simulator().run(
        system,
        SimulationConfig(start=0.02, stop=0.22, dt=0.01),
        observer=recorder,
    )

    change_times: list[float] = []
    last_value: float | None = None
    for time, value in recorder.samples:
        if last_value != value:
            change_times.append(round(time, 2))
            last_value = value

    assert change_times == [0.02, 0.12, 0.22]
    assert result.final_discrete_states["counter"] == pytest.approx(3.0)


def test_same_period_different_offset_reads_same_cycle_update():
    system = System("offset_ordering")
    system.add_block("source", Incrementer(sample_time=0.1, offset=0.0))
    system.add_block("capture", Capture(sample_time=0.1, offset=0.05))
    system.connect("source.count", "capture.u")

    simulator = Simulator()
    plan = simulator.compile(system)
    result = simulator.run(system, SimulationConfig(start=0.0, stop=0.15, dt=0.05))

    assert result.final_discrete_states["source"] == pytest.approx(2.0)
    assert result.final_discrete_states["capture"] == pytest.approx(2.0)
    assert plan.summary()["cross_rate_connections"][0]["classification"] == "same-period-different-offset"


def test_explicit_priority_changes_same_time_task_visibility():
    def build_system(source_priority: int, capture_priority: int) -> System:
        system = System("priority_visibility")
        system.add_block(
            "source",
            Incrementer(sample_time=0.2, offset=0.0, priority=source_priority),
        )
        system.add_block(
            "capture",
            Capture(sample_time=0.1, offset=0.0, priority=capture_priority),
        )
        system.connect("source.count", "capture.u")
        return system

    high_first = Simulator().run(
        build_system(source_priority=0, capture_priority=1),
        SimulationConfig(start=0.0, stop=0.2, dt=0.1),
    )
    low_first = Simulator().run(
        build_system(source_priority=1, capture_priority=0),
        SimulationConfig(start=0.0, stop=0.2, dt=0.1),
    )

    assert high_first.final_discrete_states["capture"] == pytest.approx(2.0)
    assert low_first.final_discrete_states["capture"] == pytest.approx(1.0)


def test_compiler_auto_priority_prefers_faster_then_smaller_offset():
    system = System("auto_priority")
    system.add_block("fast", Incrementer(sample_time=0.1))
    system.add_block("slow_early", Incrementer(sample_time=0.2, offset=0.0))
    system.add_block("slow_late", Incrementer(sample_time=0.2, offset=0.1))

    plan = Simulator().compile(system)

    groups = plan.summary()["rate_groups"]
    assert [group["blocks"] for group in groups] == [["fast"], ["slow_early"], ["slow_late"]]
    assert [group["resolved_priority"] for group in groups] == [0, 1, 2]


def test_cross_rate_connections_emit_warnings_and_summary_classifications():
    system = System("cross_rate_summary")
    system.add_block("slow", Incrementer(sample_time=0.2))
    system.add_block("plant", Integrator(initial=0.0))
    system.add_block("fast_capture", Capture(sample_time=0.1))
    system.add_block("offset_capture", Capture(sample_time=0.2, offset=0.1))
    system.connect("slow.count", "plant.u")
    system.connect("plant.x", "fast_capture.u")
    system.connect("slow.count", "offset_capture.u")

    simulator = Simulator()
    report = simulator.validate(system, SimulationConfig(start=0.0, stop=0.2, dt=0.1))
    summary = report.summary()

    assert report.is_valid is True
    assert report.warning_count == 3
    assert [diagnostic.code for diagnostic in report.diagnostics if diagnostic.severity == "warning"] == [
        "CROSS_RATE_CONNECTION",
        "CROSS_RATE_CONNECTION",
        "CROSS_RATE_CONNECTION",
    ]
    assert {item["classification"] for item in summary["cross_rate_connections"]} == {
        "slow-to-fast",
        "fast-to-slow",
        "same-period-different-offset",
    }


def test_direct_feedthrough_chain_propagates_after_high_priority_commit():
    system = System("propagation_chain")
    system.add_block("source", Incrementer(sample_time=0.2, priority=0))
    system.add_block("gain", Gain(2.0))
    system.add_block("capture", Capture(sample_time=0.1, priority=1))
    system.connect("source.count", "gain.u")
    system.connect("gain.y", "capture.u")

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=0.2, dt=0.1))

    assert result.final_discrete_states["capture"] == pytest.approx(4.0)


def test_continuous_interval_uses_post_hit_discrete_output():
    system = System("post_hit_continuous")
    system.add_block("command", Incrementer(sample_time=0.1))
    system.add_block("plant", Integrator(initial=0.0))
    system.connect("command.count", "plant.u")

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=0.1, dt=0.1))

    assert result.final_continuous_states["plant"] == pytest.approx(0.1, abs=1e-9)
