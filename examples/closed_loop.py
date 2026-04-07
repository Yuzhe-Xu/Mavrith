from __future__ import annotations

from pylink import (
    Block,
    ContinuousBlock,
    DiscreteBlock,
    PortSpec,
    SimulationConfig,
    Simulator,
    System,
)


class Setpoint(Block):
    outputs = (PortSpec.output("value"),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class ProportionalController(DiscreteBlock):
    inputs = (PortSpec.input("error"),)
    outputs = (PortSpec.output("command"),)

    def __init__(self, gain: float, sample_time: float) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.gain = gain

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return ctx.discrete_state

    def update_state(self, ctx, inputs, state):
        return self.gain * inputs["error"]


class FirstOrderPlant(ContinuousBlock):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("y"),)

    def __init__(self, initial: float = 0.0) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return -state + inputs["u"]


class ErrorBlock(Block):
    inputs = (PortSpec.input("setpoint"), PortSpec.input("measurement"))
    outputs = (PortSpec.output("error"),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["setpoint"] - inputs["measurement"]


if __name__ == "__main__":
    system = System("closed_loop")
    system.add_block("reference", Setpoint(1.0))
    system.add_block("error", ErrorBlock())
    system.add_block("controller", ProportionalController(gain=2.0, sample_time=0.1))
    system.add_block("plant", FirstOrderPlant(initial=0.0))

    system.connect("reference.value", "error.setpoint")
    system.connect("plant.y", "error.measurement")
    system.connect("error.error", "controller.error")
    system.connect("controller.command", "plant.u")

    result = Simulator().run(system, SimulationConfig(start=0.0, stop=1.0, dt=0.1))
    print("final y =", result.final_continuous_states["plant"])
