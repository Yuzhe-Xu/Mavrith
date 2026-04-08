from __future__ import annotations

"""Digital aircraft pitch control with hierarchical state-feedback blocks.

Reference:
- https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlDigital
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
    Subsystem,
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())
STATE_VECTOR = SignalSpec(dtype="float", shape=(3,))

AIRCRAFT_A = np.array(
    [
        [-0.313, 56.7, 0.0],
        [-0.0139, -0.426, 0.0],
        [0.0, 56.7, 0.0],
    ],
    dtype=float,
)
AIRCRAFT_B = np.array([0.232, 0.0203, 0.0], dtype=float)
DLQR_K = np.array([-0.6436, 168.3611, 6.9555], dtype=float)
DLQR_NBAR = 6.95


class PitchReference(Block):
    outputs = (PortSpec.output("theta_ref", spec=FLOAT_SCALAR),)

    def __init__(self, theta_ref: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.theta_ref = theta_ref

    def output(self, ctx, inputs):
        return self.theta_ref


class DigitalStateFeedback(DiscreteBlock):
    inputs = (
        PortSpec.input("theta_ref", spec=FLOAT_SCALAR),
        PortSpec.input("state", spec=STATE_VECTOR),
    )
    outputs = (PortSpec.output("raw_delta", spec=FLOAT_SCALAR),)

    def __init__(self, sample_time: float, max_deflection: float) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.max_deflection = max_deflection

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        state_vector = np.asarray(inputs["state"], dtype=float)
        theta_ref = float(inputs["theta_ref"])
        raw_delta = float(DLQR_NBAR * theta_ref - float(DLQR_K @ state_vector))
        return float(np.clip(raw_delta, -self.max_deflection, self.max_deflection))


class DeflectionLimiter(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("delta", spec=FLOAT_SCALAR),)

    def __init__(self, limit: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.limit = limit

    def output(self, ctx, inputs):
        command = float(inputs["u"])
        return float(np.clip(command, -self.limit, self.limit))


class AircraftPitchPlant(ContinuousBlock):
    inputs = (PortSpec.input("delta", spec=FLOAT_SCALAR),)
    outputs = (
        PortSpec.output("state", spec=STATE_VECTOR),
        PortSpec.output("theta", spec=FLOAT_SCALAR),
    )

    def __init__(self, initial_state: tuple[float, float, float]) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_state = np.asarray(initial_state, dtype=float)

    def initial_continuous_state(self):
        return self.initial_state.copy()

    def output(self, ctx, inputs):
        state = np.asarray(ctx.continuous_state, dtype=float)
        return {
            "state": state.copy(),
            "theta": float(state[2]),
        }

    def derivative(self, ctx, inputs, state):
        state_vector = np.asarray(state, dtype=float)
        delta = float(inputs["delta"])
        return AIRCRAFT_A @ state_vector + AIRCRAFT_B * delta


class PitchRecorder:
    def __init__(self) -> None:
        self.theta: list[float] = []
        self.delta: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.theta.append(float(snapshot.outputs["airframe/plant"]["theta"]))
        self.delta.append(float(snapshot.outputs["controller/limit"]["delta"]))


def build_controller_subsystem(*, sample_time: float, max_deflection: float) -> Subsystem:
    controller = Subsystem("pitch_controller")
    controller.add_block("state_feedback", DigitalStateFeedback(sample_time, max_deflection))
    controller.add_block("limit", DeflectionLimiter(limit=max_deflection))
    controller.connect("state_feedback.raw_delta", "limit.u")
    controller.expose_input("theta_ref", "state_feedback.theta_ref", spec=FLOAT_SCALAR)
    controller.expose_input("state", "state_feedback.state", spec=STATE_VECTOR)
    controller.expose_output("limit.delta", "delta", spec=FLOAT_SCALAR)
    return controller


def build_airframe_subsystem(*, initial_state: tuple[float, float, float]) -> Subsystem:
    airframe = Subsystem("airframe")
    airframe.add_block("plant", AircraftPitchPlant(initial_state=initial_state))
    airframe.expose_input("delta", "plant.delta", spec=FLOAT_SCALAR)
    airframe.expose_output("plant.state", "state", spec=STATE_VECTOR)
    airframe.expose_output("plant.theta", "theta", spec=FLOAT_SCALAR)
    return airframe


def build_system(
    *,
    theta_ref: float = 0.2,
    controller_sample_time: float = 0.01,
    max_deflection: float = 0.35,
    initial_state: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> System:
    system = System("aircraft_pitch_digital")
    system.add_block("reference", PitchReference(theta_ref))
    system.add_subsystem(
        "controller",
        build_controller_subsystem(
            sample_time=controller_sample_time,
            max_deflection=max_deflection,
        ),
    )
    system.add_subsystem("airframe", build_airframe_subsystem(initial_state=initial_state))

    system.connect("reference.theta_ref", "controller.theta_ref")
    system.connect("airframe.state", "controller.state")
    system.connect("controller.delta", "airframe.delta")
    return system


def print_summary(recorder: PitchRecorder) -> None:
    theta_peak = max(recorder.theta)
    print("aircraft pitch digital example")
    print(f"final theta = {recorder.theta[-1]:.4f} rad")
    print(f"peak theta = {theta_peak:.4f} rad")
    print(f"max |delta| = {max(abs(value) for value in recorder.delta):.4f} rad")


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=10.0, dt=0.01)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = PitchRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
