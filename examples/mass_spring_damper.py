from __future__ import annotations

"""Mass-spring-damper position control with disturbance rejection.

Inspired by the standard second-order model used in the University of Michigan
Control Tutorials for MATLAB and Simulink (CTMS).
"""

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
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())
PLANT_STATE = SignalSpec(dtype="float", shape=(2,))


class PositionReference(Block):
    """Stateless source block: reference steps from 0 to 1 meter."""

    outputs = (PortSpec.output("position_ref", spec=FLOAT_SCALAR),)

    def __init__(self, step_time: float, target_position: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.step_time = step_time
        self.target_position = target_position

    def output(self, ctx, inputs):
        return 0.0 if ctx.time < self.step_time else self.target_position


class DisturbanceForce(Block):
    """Stateless disturbance block: short force pulse to test recovery."""

    outputs = (PortSpec.output("disturbance", spec=FLOAT_SCALAR),)

    def __init__(self, start_time: float, end_time: float, force: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.start_time = start_time
        self.end_time = end_time
        self.force = force

    def output(self, ctx, inputs):
        if self.start_time <= ctx.time < self.end_time:
            return self.force
        return 0.0


class PDController(DiscreteBlock):
    """Sampled controller: PD feedback plus spring-force feedforward."""

    inputs = (
        PortSpec.input("position_ref", spec=FLOAT_SCALAR),
        PortSpec.input("position", spec=FLOAT_SCALAR),
        PortSpec.input("velocity", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("raw_force", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        *,
        sample_time: float,
        kp: float,
        kd: float,
        reference_gain: float,
    ) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.kp = kp
        self.kd = kd
        self.reference_gain = reference_gain

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        position_ref = float(inputs["position_ref"])
        error = position_ref - float(inputs["position"])
        damping = float(inputs["velocity"])
        return self.kp * error - self.kd * damping + self.reference_gain * position_ref


class ForceLimiter(Block):
    """Stateless saturation block: limits the actuator force magnitude."""

    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("force", spec=FLOAT_SCALAR),)

    def __init__(self, max_force: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.max_force = max_force

    def output(self, ctx, inputs):
        force = float(inputs["u"])
        return min(max(force, -self.max_force), self.max_force)


class MassSpringDamper(ContinuousBlock):
    """Continuous plant: state is [position, velocity]."""

    inputs = (
        PortSpec.input("force", spec=FLOAT_SCALAR),
        PortSpec.input("disturbance", spec=FLOAT_SCALAR),
    )
    outputs = (
        PortSpec.output("position", spec=FLOAT_SCALAR),
        PortSpec.output("velocity", spec=FLOAT_SCALAR),
    )

    def __init__(
        self,
        *,
        mass: float,
        damping: float,
        stiffness: float,
        initial_state: tuple[float, float],
    ) -> None:
        super().__init__(direct_feedthrough=False)
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.initial_state = np.asarray(initial_state, dtype=float)

    def initial_continuous_state(self):
        return self.initial_state.copy()

    def output(self, ctx, inputs):
        position, velocity = np.asarray(ctx.continuous_state, dtype=float)
        return {
            "position": float(position),
            "velocity": float(velocity),
        }

    def derivative(self, ctx, inputs, state):
        position, velocity = np.asarray(state, dtype=float)
        force = float(inputs["force"])
        disturbance = float(inputs["disturbance"])
        acceleration = (
            force + disturbance - self.damping * velocity - self.stiffness * position
        ) / self.mass
        return np.array([velocity, acceleration], dtype=float)


class MassSpringRecorder:
    def __init__(self) -> None:
        self.positions: list[float] = []
        self.velocities: list[float] = []
        self.forces: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.positions.append(float(snapshot.outputs["plant"]["position"]))
        self.velocities.append(float(snapshot.outputs["plant"]["velocity"]))
        self.forces.append(float(snapshot.outputs["limiter"]["force"]))


def build_system(
    *,
    step_time: float = 0.5,
    target_position: float = 1.0,
    disturbance_start: float = 4.0,
    disturbance_end: float = 4.5,
    disturbance_force: float = 2.0,
    controller_sample_time: float = 0.02,
    controller_kp: float = 12.0,
    controller_kd: float = 4.8,
    max_force: float = 6.0,
    mass: float = 1.0,
    damping: float = 0.8,
    stiffness: float = 4.0,
    initial_state: tuple[float, float] = (0.0, 0.0),
) -> System:
    system = System("mass_spring_damper")
    system.add_block("reference", PositionReference(step_time, target_position))
    system.add_block(
        "disturbance",
        DisturbanceForce(
            start_time=disturbance_start,
            end_time=disturbance_end,
            force=disturbance_force,
        ),
    )
    system.add_block(
        "controller",
        PDController(
            sample_time=controller_sample_time,
            kp=controller_kp,
            kd=controller_kd,
            reference_gain=stiffness,
        ),
    )
    system.add_block("limiter", ForceLimiter(max_force=max_force))
    system.add_block(
        "plant",
        MassSpringDamper(
            mass=mass,
            damping=damping,
            stiffness=stiffness,
            initial_state=initial_state,
        ),
    )

    system.connect("reference.position_ref", "controller.position_ref")
    system.connect("plant.position", "controller.position")
    system.connect("plant.velocity", "controller.velocity")
    system.connect("controller.raw_force", "limiter.u")
    system.connect("limiter.force", "plant.force")
    system.connect("disturbance.disturbance", "plant.disturbance")
    return system


def print_summary(recorder: MassSpringRecorder) -> None:
    print("mass-spring-damper example")
    print(f"final position = {recorder.positions[-1]:.3f} m")
    print(f"final velocity = {recorder.velocities[-1]:.3f} m/s")
    print(f"peak absolute force = {max(abs(force) for force in recorder.forces):.3f} N")


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=10.0, dt=0.005)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = MassSpringRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
