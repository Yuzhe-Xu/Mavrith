from __future__ import annotations

from mavrith import (
    Block,
    ContinuousBlock,
    DiscreteBlock,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    Simulator,
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class Setpoint(Block):
    """Stateless source block: the output is a fixed parameter."""

    outputs = (PortSpec.output("value", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(
            direct_feedthrough=False,
            parameters={"value": value},
            description="Constant reference source for the closed-loop example.",
        )
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class ErrorBlock(Block):
    """Stateless algebraic block: current error depends on current inputs."""

    inputs = (
        PortSpec.input("setpoint", spec=FLOAT_SCALAR),
        PortSpec.input("measurement", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("error", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(
            direct_feedthrough=True,
            description="Computes setpoint minus measurement for the feedback loop.",
        )

    def output(self, ctx, inputs):
        return inputs["setpoint"] - inputs["measurement"]


class ProportionalController(DiscreteBlock):
    """Sampled controller: output comes from stored discrete state."""

    inputs = (PortSpec.input("error", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("command", spec=FLOAT_SCALAR),)

    def __init__(self, gain: float, sample_time: float) -> None:
        super().__init__(
            sample_time=sample_time,
            direct_feedthrough=False,
            parameters={"gain": gain},
            description="Sampled proportional controller whose output is stored in discrete state.",
        )
        self.gain = gain

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return ctx.discrete_state

    def update_state(self, ctx, inputs, state):
        return self.gain * inputs["error"]


class FirstOrderPlant(ContinuousBlock):
    """Continuous plant: output is the stored continuous state."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, initial: float = 0.0) -> None:
        super().__init__(
            direct_feedthrough=False,
            parameters={"initial": initial},
            description="First-order plant driven by the controller command.",
        )
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return -state + inputs["u"]


def build_system(
    *,
    reference: float = 1.0,
    controller_gain: float = 2.0,
    controller_sample_time: float = 0.1,
    plant_initial: float = 0.0,
) -> System:
    system = System("closed_loop")
    system.add_block("reference", Setpoint(reference))
    system.add_block("error", ErrorBlock())
    system.add_block(
        "controller",
        ProportionalController(
            gain=controller_gain,
            sample_time=controller_sample_time,
        ),
    )
    system.add_block("plant", FirstOrderPlant(initial=plant_initial))

    system.connect("reference.value", "error.setpoint")
    system.connect("plant.y", "error.measurement")
    system.connect("error.error", "controller.error")
    system.connect("controller.command", "plant.u")
    return system


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    result = simulator.run(system, config)
    print("execution order =", report.summary()["execution_order"])
    print("final y =", result.final_continuous_states["plant"])


if __name__ == "__main__":
    main()
