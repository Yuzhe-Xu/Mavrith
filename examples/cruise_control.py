from __future__ import annotations

"""Cruise control with a PI loop, saturation, and road-grade disturbance.

Inspired by:
- the python-control cruise control example
- the standard longitudinal force balance used in cruise-control tutorials
"""

import math

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


class SpeedReference(Block):
    """Stateless source block: desired speed follows a simple time schedule."""

    outputs = (PortSpec.output("speed_ref", spec=FLOAT_SCALAR),)

    def __init__(self, low_speed: float, high_speed: float, switch_time: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.low_speed = low_speed
        self.high_speed = high_speed
        self.switch_time = switch_time

    def output(self, ctx, inputs):
        return self.low_speed if ctx.time < self.switch_time else self.high_speed


class GradeProfile(Block):
    """Stateless disturbance block: grade changes over time to stress the loop."""

    outputs = (PortSpec.output("grade_rad", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=False)

    def output(self, ctx, inputs):
        if ctx.time < 6.0:
            return 0.0
        if ctx.time < 13.0:
            return math.radians(5.0)
        if ctx.time < 18.0:
            return 0.0
        if ctx.time < 22.0:
            return math.radians(-3.0)
        return 0.0


class SpeedError(Block):
    """Stateless algebraic block: current error depends on current speed."""

    inputs = (
        PortSpec.input("speed_ref", spec=FLOAT_SCALAR),
        PortSpec.input("speed", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("error", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["speed_ref"] - inputs["speed"]


class PIThrottleController(DiscreteBlock):
    """Sampled PI controller with simple anti-windup via conditional integration."""

    inputs = (PortSpec.input("error", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("raw_throttle", spec=FLOAT_SCALAR),)

    def __init__(self, *, kp: float, ki: float, sample_time: float) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.kp = kp
        self.ki = ki

    def initial_discrete_state(self):
        return {
            "integral": 0.0,
            "command": 0.0,
        }

    def output(self, ctx, inputs):
        return float(ctx.discrete_state["command"])

    def update_state(self, ctx, inputs, state):
        error = float(inputs["error"])
        integral = float(state["integral"]) + error * ctx.dt
        raw = self.kp * error + self.ki * integral
        saturated = min(max(raw, 0.0), 1.0)

        if raw != saturated and error * raw > 0.0:
            integral = float(state["integral"])
            raw = self.kp * error + self.ki * integral

        return {
            "integral": integral,
            "command": raw,
        }


class ThrottleLimiter(Block):
    """Stateless saturation block: keeps throttle within [0, 1]."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("throttle", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return min(max(float(inputs["u"]), 0.0), 1.0)


class CruiseVehicle(ContinuousBlock):
    """Continuous plant: longitudinal speed with drag and road-grade disturbance."""

    inputs = (
        PortSpec.input("throttle", spec=FLOAT_SCALAR),
        PortSpec.input("grade_rad", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("speed", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        *,
        initial_speed: float,
        mass: float,
        max_force: float,
        rolling_drag: float,
        air_drag: float,
    ) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_speed = initial_speed
        self.mass = mass
        self.max_force = max_force
        self.rolling_drag = rolling_drag
        self.air_drag = air_drag

    def initial_continuous_state(self):
        return self.initial_speed

    def output(self, ctx, inputs):
        return max(float(ctx.continuous_state), 0.0)

    def derivative(self, ctx, inputs, state):
        speed = max(float(state), 0.0)
        throttle = min(max(float(inputs["throttle"]), 0.0), 1.0)
        grade_rad = float(inputs["grade_rad"])

        engine_force = self.max_force * throttle
        rolling_force = self.rolling_drag * speed
        aerodynamic_force = self.air_drag * speed * abs(speed)
        grade_force = self.mass * 9.81 * math.sin(grade_rad)
        acceleration = (
            engine_force - rolling_force - aerodynamic_force - grade_force
        ) / self.mass
        if speed <= 0.0 and acceleration < 0.0:
            return 0.0
        return acceleration


class CruiseRecorder:
    def __init__(self) -> None:
        self.speeds: list[float] = []
        self.throttles: list[float] = []
        self.references: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.speeds.append(float(snapshot.outputs["vehicle"]["speed"]))
        self.throttles.append(float(snapshot.outputs["limiter"]["throttle"]))
        self.references.append(float(snapshot.outputs["reference"]["speed_ref"]))


def build_system(
    *,
    low_speed: float = 12.0,
    high_speed: float = 22.0,
    switch_time: float = 2.0,
    controller_sample_time: float = 0.1,
    controller_kp: float = 0.25,
    controller_ki: float = 0.18,
    initial_speed: float = 0.0,
    mass: float = 1200.0,
    max_force: float = 5200.0,
    rolling_drag: float = 55.0,
    air_drag: float = 3.8,
) -> System:
    system = System("cruise_control")
    system.add_block("reference", SpeedReference(low_speed, high_speed, switch_time))
    system.add_block("grade", GradeProfile())
    system.add_block("error", SpeedError())
    system.add_block(
        "controller",
        PIThrottleController(
            kp=controller_kp,
            ki=controller_ki,
            sample_time=controller_sample_time,
        ),
    )
    system.add_block("limiter", ThrottleLimiter())
    system.add_block(
        "vehicle",
        CruiseVehicle(
            initial_speed=initial_speed,
            mass=mass,
            max_force=max_force,
            rolling_drag=rolling_drag,
            air_drag=air_drag,
        ),
    )

    system.connect("reference.speed_ref", "error.speed_ref")
    system.connect("vehicle.speed", "error.speed")
    system.connect("error.error", "controller.error")
    system.connect("controller.raw_throttle", "limiter.u")
    system.connect("limiter.throttle", "vehicle.throttle")
    system.connect("grade.grade_rad", "vehicle.grade_rad")
    return system


def print_summary(recorder: CruiseRecorder) -> None:
    print("cruise control example")
    print(f"final speed = {recorder.speeds[-1]:.3f} m/s")
    print(f"final reference = {recorder.references[-1]:.3f} m/s")
    print(f"max throttle = {max(recorder.throttles):.3f}")
    print(f"min throttle = {min(recorder.throttles):.3f}")


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=25.0, dt=0.02)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = CruiseRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
