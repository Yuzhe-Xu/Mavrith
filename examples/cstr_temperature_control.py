from __future__ import annotations

"""Nonlinear CSTR temperature regulation with hierarchical PI control.

Reference:
- https://apmonitor.com/wiki/index.php/Apps/StirredReactor
"""

import math

from mavrith import (
    Block,
    ContinuousBlock,
    DiscreteBlock,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    Simulator,
    StepSnapshot,
    Subsystem,
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())
STATE_VECTOR = SignalSpec(dtype="float", shape=(2,))


class TemperatureReference(Block):
    outputs = (PortSpec.output("temperature_ref", spec=FLOAT_SCALAR),)

    def __init__(self, temperature_ref: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.temperature_ref = temperature_ref

    def output(self, ctx, inputs):
        return self.temperature_ref


class ReactorTemperatureController(DiscreteBlock):
    inputs = (
        PortSpec.input("temperature_ref", spec=FLOAT_SCALAR),
        PortSpec.input("temperature", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("raw_coolant", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        *,
        sample_time: float,
        bias: float,
        kp: float,
        ki: float,
        min_coolant: float,
        max_coolant: float,
    ) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.bias = bias
        self.kp = kp
        self.ki = ki
        self.min_coolant = min_coolant
        self.max_coolant = max_coolant

    def initial_discrete_state(self):
        return {
            "integral": 0.0,
            "command": self.bias,
        }

    def output(self, ctx, inputs):
        return float(ctx.discrete_state["command"])

    def update_state(self, ctx, inputs, state):
        error = float(inputs["temperature_ref"]) - float(inputs["temperature"])
        integral = float(state["integral"]) + error * ctx.dt
        raw = self.bias + self.kp * error + self.ki * integral
        saturated = min(max(raw, self.min_coolant), self.max_coolant)
        if raw != saturated and raw * error > 0.0:
            integral = float(state["integral"])
            raw = self.bias + self.kp * error + self.ki * integral
        return {
            "integral": integral,
            "command": raw,
        }


class CoolantLimiter(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("coolant_temp", spec=FLOAT_SCALAR),)

    def __init__(self, min_coolant: float, max_coolant: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.min_coolant = min_coolant
        self.max_coolant = max_coolant

    def output(self, ctx, inputs):
        return min(max(float(inputs["u"]), self.min_coolant), self.max_coolant)


class ContinuousStirredTankReactor(ContinuousBlock):
    inputs = (PortSpec.input("coolant_temp", spec=FLOAT_SCALAR),)
    outputs = (
        PortSpec.output("state", spec=STATE_VECTOR),
        PortSpec.output("concentration", spec=FLOAT_SCALAR),
        PortSpec.output("temperature", spec=FLOAT_SCALAR),
    )

    def __init__(
        self,
        *,
        initial_state: tuple[float, float],
        q: float = 100.0,
        volume: float = 100.0,
        density: float = 1000.0,
        heat_capacity: float = 0.239,
        heat_of_reaction: float = 5.0e4,
        activation_over_r: float = 8750.0,
        pre_exponential: float = 7.2e10,
        heat_transfer: float = 5.0e4,
        feed_concentration: float = 1.0,
        feed_temperature: float = 350.0,
    ) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_state = initial_state
        self.q = q
        self.volume = volume
        self.density = density
        self.heat_capacity = heat_capacity
        self.heat_of_reaction = heat_of_reaction
        self.activation_over_r = activation_over_r
        self.pre_exponential = pre_exponential
        self.heat_transfer = heat_transfer
        self.feed_concentration = feed_concentration
        self.feed_temperature = feed_temperature

    def initial_continuous_state(self):
        return self.initial_state

    def output(self, ctx, inputs):
        concentration, temperature = ctx.continuous_state
        return {
            "state": (float(concentration), float(temperature)),
            "concentration": float(concentration),
            "temperature": float(temperature),
        }

    def derivative(self, ctx, inputs, state):
        concentration, temperature = (float(state[0]), float(state[1]))
        temperature = max(temperature, 1.0)
        coolant_temp = float(inputs["coolant_temp"])
        reaction_rate = self.pre_exponential * math.exp(-self.activation_over_r / temperature) * concentration
        d_concentration = (self.q / self.volume) * (self.feed_concentration - concentration) - reaction_rate
        d_temperature = (
            (self.q / self.volume) * (self.feed_temperature - temperature)
            + (self.heat_of_reaction / (self.density * self.heat_capacity)) * reaction_rate
            + (self.heat_transfer / (self.density * self.heat_capacity * self.volume))
            * (coolant_temp - temperature)
        )
        return (d_concentration, d_temperature)


class CSTRRecorder:
    def __init__(self) -> None:
        self.concentrations: list[float] = []
        self.temperatures: list[float] = []
        self.coolant_temperatures: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.concentrations.append(float(snapshot.outputs["reactor/plant"]["concentration"]))
        self.temperatures.append(float(snapshot.outputs["reactor/plant"]["temperature"]))
        self.coolant_temperatures.append(float(snapshot.outputs["controller/limiter"]["coolant_temp"]))


def build_controller_subsystem(
    *,
    sample_time: float,
    bias: float,
    kp: float,
    ki: float,
    min_coolant: float,
    max_coolant: float,
) -> Subsystem:
    controller = Subsystem("temperature_controller")
    controller.add_block(
        "pi",
        ReactorTemperatureController(
            sample_time=sample_time,
            bias=bias,
            kp=kp,
            ki=ki,
            min_coolant=min_coolant,
            max_coolant=max_coolant,
        ),
    )
    controller.add_block("limiter", CoolantLimiter(min_coolant=min_coolant, max_coolant=max_coolant))
    controller.connect("pi.raw_coolant", "limiter.u")
    controller.expose_input("temperature_ref", "pi.temperature_ref", spec=FLOAT_SCALAR)
    controller.expose_input("temperature", "pi.temperature", spec=FLOAT_SCALAR)
    controller.expose_output("limiter.coolant_temp", "coolant_temp", spec=FLOAT_SCALAR)
    return controller


def build_reactor_subsystem(*, initial_state: tuple[float, float]) -> Subsystem:
    reactor = Subsystem("reactor")
    reactor.add_block("plant", ContinuousStirredTankReactor(initial_state=initial_state))
    reactor.expose_input("coolant_temp", "plant.coolant_temp", spec=FLOAT_SCALAR)
    reactor.expose_output("plant.state", "state", spec=STATE_VECTOR)
    reactor.expose_output("plant.concentration", "concentration", spec=FLOAT_SCALAR)
    reactor.expose_output("plant.temperature", "temperature", spec=FLOAT_SCALAR)
    return reactor


def build_system(
    *,
    temperature_ref: float = 305.0,
    controller_sample_time: float = 0.05,
    initial_state: tuple[float, float] = (0.8, 315.0),
    controller_bias: float = 280.0,
    controller_kp: float = 3.0,
    controller_ki: float = 0.35,
    min_coolant: float = 250.0,
    max_coolant: float = 320.0,
) -> System:
    system = System("cstr_temperature_control")
    system.add_block("reference", TemperatureReference(temperature_ref))
    system.add_subsystem(
        "controller",
        build_controller_subsystem(
            sample_time=controller_sample_time,
            bias=controller_bias,
            kp=controller_kp,
            ki=controller_ki,
            min_coolant=min_coolant,
            max_coolant=max_coolant,
        ),
    )
    system.add_subsystem("reactor", build_reactor_subsystem(initial_state=initial_state))

    system.connect("reference.temperature_ref", "controller.temperature_ref")
    system.connect("reactor.temperature", "controller.temperature")
    system.connect("controller.coolant_temp", "reactor.coolant_temp")
    return system


def print_summary(recorder: CSTRRecorder) -> None:
    print("cstr temperature control example")
    print(f"final concentration = {recorder.concentrations[-1]:.4f} mol/L")
    print(f"final temperature = {recorder.temperatures[-1]:.3f} K")
    print(
        "coolant range = "
        f"[{min(recorder.coolant_temperatures):.3f}, {max(recorder.coolant_temperatures):.3f}] K"
    )


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=5.0, dt=0.01)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = CSTRRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
