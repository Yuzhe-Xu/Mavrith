from __future__ import annotations

"""Hierarchical control of the quadruple-tank benchmark process.

References:
- https://home.ufam.edu.br/iurybessa/Laborat%C3%B3rio%20de%20Sistemas%20de%20Controle/old/The%20Quadruple-Tank%20Process.pdf
- https://sim.cpsec.org/en/latest/_modules/cpsim/models/linear/quadruple_tank.html
"""

import math

import numpy as np

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
STATE_VECTOR = SignalSpec(dtype="float", shape=(4,))


class LevelReference(Block):
    outputs = (PortSpec.output("level_ref", spec=FLOAT_SCALAR),)

    def __init__(self, level_ref: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.level_ref = level_ref

    def output(self, ctx, inputs):
        return self.level_ref


class TankPIController(DiscreteBlock):
    inputs = (
        PortSpec.input("level_ref", spec=FLOAT_SCALAR),
        PortSpec.input("level", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("pump_voltage", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        *,
        sample_time: float,
        bias: float,
        kp: float,
        ki: float,
        min_voltage: float,
        max_voltage: float,
    ) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.bias = bias
        self.kp = kp
        self.ki = ki
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage

    def initial_discrete_state(self):
        return {
            "integral": 0.0,
            "command": self.bias,
        }

    def output(self, ctx, inputs):
        return float(ctx.discrete_state["command"])

    def update_state(self, ctx, inputs, state):
        error = float(inputs["level_ref"]) - float(inputs["level"])
        integral = float(state["integral"]) + error * ctx.dt
        raw = self.bias + self.kp * error + self.ki * integral
        saturated = min(max(raw, self.min_voltage), self.max_voltage)
        if raw != saturated and raw * error > 0.0:
            integral = float(state["integral"])
            raw = self.bias + self.kp * error + self.ki * integral
        return {
            "integral": integral,
            "command": saturated,
        }


class QuadrupleTankPlant(ContinuousBlock):
    inputs = (
        PortSpec.input("v1", spec=FLOAT_SCALAR),
        PortSpec.input("v2", spec=FLOAT_SCALAR),
    )
    outputs = (
        PortSpec.output("state", spec=STATE_VECTOR),
        PortSpec.output("h1", spec=FLOAT_SCALAR),
        PortSpec.output("h2", spec=FLOAT_SCALAR),
    )

    def __init__(
        self,
        *,
        initial_state: tuple[float, float, float, float],
        gamma1: float = 0.70,
        gamma2: float = 0.60,
        A1: float = 28.0,
        A2: float = 32.0,
        A3: float = 28.0,
        A4: float = 32.0,
        a1: float = 0.071,
        a2: float = 0.057,
        a3: float = 0.071,
        a4: float = 0.057,
        k1: float = 3.33,
        k2: float = 3.35,
        g: float = 981.0,
    ) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.k1 = k1
        self.k2 = k2
        self.g = g

    def initial_continuous_state(self):
        return self.initial_state.copy()

    def output(self, ctx, inputs):
        levels = np.asarray(ctx.continuous_state, dtype=float)
        return {
            "state": levels.copy(),
            "h1": float(levels[0]),
            "h2": float(levels[1]),
        }

    def derivative(self, ctx, inputs, state):
        h1, h2, h3, h4 = (max(float(level), 0.0) for level in state)
        v1 = max(float(inputs["v1"]), 0.0)
        v2 = max(float(inputs["v2"]), 0.0)
        root = lambda level: math.sqrt(max(2.0 * self.g * level, 0.0))
        dh1 = (-self.a1 * root(h1) + self.a3 * root(h3) + self.gamma1 * self.k1 * v1) / self.A1
        dh2 = (-self.a2 * root(h2) + self.a4 * root(h4) + self.gamma2 * self.k2 * v2) / self.A2
        dh3 = (-self.a3 * root(h3) + (1.0 - self.gamma2) * self.k2 * v2) / self.A3
        dh4 = (-self.a4 * root(h4) + (1.0 - self.gamma1) * self.k1 * v1) / self.A4
        return np.array([dh1, dh2, dh3, dh4], dtype=float)


class QuadrupleTankRecorder:
    def __init__(self) -> None:
        self.h1: list[float] = []
        self.h2: list[float] = []
        self.v1: list[float] = []
        self.v2: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.h1.append(float(snapshot.outputs["plant/tanks"]["h1"]))
        self.h2.append(float(snapshot.outputs["plant/tanks"]["h2"]))
        self.v1.append(float(snapshot.outputs["controller/loop_1/pi"]["pump_voltage"]))
        self.v2.append(float(snapshot.outputs["controller/loop_2/pi"]["pump_voltage"]))


def build_loop_subsystem(
    *,
    sample_time: float,
    bias: float,
    kp: float,
    ki: float,
    min_voltage: float,
    max_voltage: float,
) -> Subsystem:
    loop = Subsystem("level_loop")
    loop.add_block(
        "pi",
        TankPIController(
            sample_time=sample_time,
            bias=bias,
            kp=kp,
            ki=ki,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        ),
    )
    loop.expose_input("level_ref", "pi.level_ref", spec=FLOAT_SCALAR)
    loop.expose_input("level", "pi.level", spec=FLOAT_SCALAR)
    loop.expose_output("pi.pump_voltage", "pump_voltage", spec=FLOAT_SCALAR)
    return loop


def build_controller_subsystem(
    *,
    sample_time: float,
    bias: float,
    kp_1: float,
    ki_1: float,
    kp_2: float,
    ki_2: float,
    min_voltage: float,
    max_voltage: float,
) -> Subsystem:
    controller = Subsystem("quadruple_controller")
    controller.add_subsystem(
        "loop_1",
        build_loop_subsystem(
            sample_time=sample_time,
            bias=bias,
            kp=kp_1,
            ki=ki_1,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        ),
    )
    controller.add_subsystem(
        "loop_2",
        build_loop_subsystem(
            sample_time=sample_time,
            bias=bias,
            kp=kp_2,
            ki=ki_2,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        ),
    )
    controller.expose_input("h1_ref", "loop_1.level_ref", spec=FLOAT_SCALAR)
    controller.expose_input("h2_ref", "loop_2.level_ref", spec=FLOAT_SCALAR)
    controller.expose_input("h1", "loop_1.level", spec=FLOAT_SCALAR)
    controller.expose_input("h2", "loop_2.level", spec=FLOAT_SCALAR)
    controller.expose_output("loop_1.pump_voltage", "v1", spec=FLOAT_SCALAR)
    controller.expose_output("loop_2.pump_voltage", "v2", spec=FLOAT_SCALAR)
    return controller


def build_plant_subsystem(*, initial_state: tuple[float, float, float, float]) -> Subsystem:
    plant = Subsystem("quadruple_tank_plant")
    plant.add_block("tanks", QuadrupleTankPlant(initial_state=initial_state))
    plant.expose_input("v1", "tanks.v1", spec=FLOAT_SCALAR)
    plant.expose_input("v2", "tanks.v2", spec=FLOAT_SCALAR)
    plant.expose_output("tanks.state", "state", spec=STATE_VECTOR)
    plant.expose_output("tanks.h1", "h1", spec=FLOAT_SCALAR)
    plant.expose_output("tanks.h2", "h2", spec=FLOAT_SCALAR)
    return plant


def build_system(
    *,
    h1_ref: float = 12.4,
    h2_ref: float = 12.7,
    controller_sample_time: float = 0.5,
    initial_state: tuple[float, float, float, float] = (14.5, 11.0, 2.0, 1.0),
    controller_bias: float = 3.0,
    controller_kp_1: float = 1.2,
    controller_ki_1: float = 0.045,
    controller_kp_2: float = 1.15,
    controller_ki_2: float = 0.040,
    min_voltage: float = 0.0,
    max_voltage: float = 10.0,
) -> System:
    system = System("quadruple_tank")
    system.add_block("reference_1", LevelReference(h1_ref))
    system.add_block("reference_2", LevelReference(h2_ref))
    system.add_subsystem(
        "controller",
        build_controller_subsystem(
            sample_time=controller_sample_time,
            bias=controller_bias,
            kp_1=controller_kp_1,
            ki_1=controller_ki_1,
            kp_2=controller_kp_2,
            ki_2=controller_ki_2,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
        ),
    )
    system.add_subsystem("plant", build_plant_subsystem(initial_state=initial_state))

    system.connect("reference_1.level_ref", "controller.h1_ref")
    system.connect("reference_2.level_ref", "controller.h2_ref")
    system.connect("plant.h1", "controller.h1")
    system.connect("plant.h2", "controller.h2")
    system.connect("controller.v1", "plant.v1")
    system.connect("controller.v2", "plant.v2")
    return system


def print_summary(recorder: QuadrupleTankRecorder) -> None:
    print("quadruple tank example")
    print(f"final h1 = {recorder.h1[-1]:.3f} cm")
    print(f"final h2 = {recorder.h2[-1]:.3f} cm")
    print(f"voltage range v1 = [{min(recorder.v1):.3f}, {max(recorder.v1):.3f}] V")
    print(f"voltage range v2 = [{min(recorder.v2):.3f}, {max(recorder.v2):.3f}] V")


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=120.0, dt=0.5)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = QuadrupleTankRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
