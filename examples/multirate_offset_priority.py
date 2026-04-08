from __future__ import annotations

"""Minimal multirate example with offset and priority-aware scheduling."""

from mavrith import (
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


class Incrementer(DiscreteBlock):
    """Sampled source whose output is the stored discrete count."""

    outputs = (PortSpec.output("count", spec=FLOAT_SCALAR),)

    def __init__(self, *, sample_time: float, offset: float, priority: int) -> None:
        super().__init__(
            sample_time=sample_time,
            offset=offset,
            priority=priority,
            direct_feedthrough=False,
            description="Sampled counter source used to demonstrate priority-ordered rate groups.",
        )

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        return float(state) + 1.0


class Gain(Block):
    """Stateless algebraic block: output depends on current input."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, gain: float) -> None:
        super().__init__(
            direct_feedthrough=True,
            parameters={"gain": gain},
            description="Algebraic gain block shared by the discrete observers and continuous plant.",
        )
        self.gain = gain

    def output(self, ctx, inputs):
        return self.gain * float(inputs["u"])


class Capture(DiscreteBlock):
    """Sampled observer that stores the last visible value on its own clock."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, *, sample_time: float, offset: float, priority: int) -> None:
        super().__init__(
            sample_time=sample_time,
            offset=offset,
            priority=priority,
            direct_feedthrough=False,
            description="Sampled capture block that records the visible signal on its own schedule.",
        )

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        return float(inputs["u"])


class Integrator(ContinuousBlock):
    """Continuous plant that integrates the sampled command."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("x", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(
            direct_feedthrough=False,
            description="Continuous integrator used to show interaction between sampled and continuous logic.",
        )

    def initial_continuous_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.continuous_state)

    def derivative(self, ctx, inputs, state):
        return float(inputs["u"])


class Recorder:
    def __init__(self) -> None:
        self.rows: list[tuple[float, float, float, float]] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.rows.append(
            (
                snapshot.time,
                float(snapshot.outputs["source"]["count"]),
                float(snapshot.outputs["same_time_capture"]["y"]),
                float(snapshot.outputs["offset_capture"]["y"]),
            )
        )


def build_system() -> System:
    system = System("multirate_offset_priority")
    system.add_block("source", Incrementer(sample_time=0.2, offset=0.0, priority=0))
    system.add_block("gain", Gain(10.0))
    system.add_block("same_time_capture", Capture(sample_time=0.1, offset=0.0, priority=1))
    system.add_block("offset_capture", Capture(sample_time=0.2, offset=0.1, priority=2))
    system.add_block("plant", Integrator())

    system.connect("source.count", "gain.u")
    system.connect("gain.y", "same_time_capture.u")
    system.connect("source.count", "offset_capture.u")
    system.connect("gain.y", "plant.u")
    return system


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=0.4, dt=0.1)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.severity} {diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    print("rate groups =", report.summary()["rate_groups"])
    print("cross-rate connections =", report.summary()["cross_rate_connections"])

    recorder = Recorder()
    result = simulator.run(system, config, observer=recorder)
    print("time, source, same_time_capture, offset_capture")
    for row in recorder.rows:
        print(f"{row[0]:.1f}, {row[1]:.1f}, {row[2]:.1f}, {row[3]:.1f}")
    print("final plant state =", result.final_continuous_states["plant"])


if __name__ == "__main__":
    main()
