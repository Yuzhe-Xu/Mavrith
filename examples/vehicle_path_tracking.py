from __future__ import annotations

"""Circle tracking with a kinematic bicycle and pure-pursuit-style steering.

Inspired by the pure pursuit formulation described in:
- Thomas Fermi, Algorithms for Automated Driving
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
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())
STATE_VECTOR = SignalSpec(dtype="float", shape=(4,))
COMMAND_VECTOR = SignalSpec(dtype="float", shape=(2,))


def _wrap_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


class SpeedSetpoint(Block):
    """Stateless source block: the desired speed is a fixed parameter."""

    outputs = (PortSpec.output("speed_ref", spec=FLOAT_SCALAR),)

    def __init__(self, speed_ref: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.speed_ref = speed_ref

    def output(self, ctx, inputs):
        return self.speed_ref


class PurePursuitController(DiscreteBlock):
    """Sampled controller: output is the stored acceleration/steering command."""

    inputs = (
        PortSpec.input("state", spec=STATE_VECTOR),
        PortSpec.input("speed_ref", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("command", spec=COMMAND_VECTOR),)

    def __init__(
        self,
        *,
        sample_time: float,
        wheel_base: float,
        path_radius: float,
        lookahead_gain: float,
        min_lookahead: float,
        max_lookahead: float,
        speed_gain: float,
        speed_feedforward_gain: float,
        max_accel: float,
        max_steer: float,
    ) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.wheel_base = wheel_base
        self.path_radius = path_radius
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.speed_gain = speed_gain
        self.speed_feedforward_gain = speed_feedforward_gain
        self.max_accel = max_accel
        self.max_steer = max_steer

    def initial_discrete_state(self):
        return np.array([0.0, 0.0], dtype=float)

    def output(self, ctx, inputs):
        return np.asarray(ctx.discrete_state, dtype=float)

    def update_state(self, ctx, inputs, state):
        x, y, yaw, speed = np.asarray(inputs["state"], dtype=float)
        speed_ref = float(inputs["speed_ref"])

        radial_distance = max(float(np.hypot(x, y)), 1e-6)
        path_angle = math.atan2(y, x)
        lookahead = float(
            np.clip(
                self.lookahead_gain * max(speed, 0.0) + self.min_lookahead,
                self.min_lookahead,
                self.max_lookahead,
            )
        )
        target_angle = path_angle + lookahead / self.path_radius
        target_x = self.path_radius * math.cos(target_angle)
        target_y = self.path_radius * math.sin(target_angle)

        alpha = _wrap_angle(math.atan2(target_y - y, target_x - x) - yaw)
        curvature_steer = math.atan2(2.0 * self.wheel_base * math.sin(alpha), lookahead)
        radial_error = radial_distance - self.path_radius
        corrective_steer = -0.06 * radial_error
        steer = float(
            np.clip(
                curvature_steer + corrective_steer,
                -self.max_steer,
                self.max_steer,
            )
        )
        accel = float(
            np.clip(
                self.speed_gain * (speed_ref - speed) + self.speed_feedforward_gain * speed_ref,
                -self.max_accel,
                self.max_accel,
            )
        )
        return np.array([accel, steer], dtype=float)


class KinematicBicycle(ContinuousBlock):
    """Continuous plant: state is [x, y, yaw, speed] on a constant-radius path."""

    inputs = (PortSpec.input("command", spec=COMMAND_VECTOR),)
    outputs = (
        PortSpec.output("state", spec=STATE_VECTOR),
        PortSpec.output("path_error", spec=FLOAT_SCALAR),
    )

    def __init__(
        self,
        *,
        initial_state: tuple[float, float, float, float],
        wheel_base: float,
        path_radius: float,
        speed_drag: float,
    ) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.wheel_base = wheel_base
        self.path_radius = path_radius
        self.speed_drag = speed_drag

    def initial_continuous_state(self):
        return self.initial_state.copy()

    def output(self, ctx, inputs):
        state = np.asarray(ctx.continuous_state, dtype=float)
        path_error = abs(float(np.hypot(state[0], state[1]) - self.path_radius))
        return {
            "state": state.copy(),
            "path_error": path_error,
        }

    def derivative(self, ctx, inputs, state):
        accel, steer = np.asarray(inputs["command"], dtype=float)
        x, y, yaw, speed = np.asarray(state, dtype=float)

        speed = max(speed, 0.0)
        dx = speed * math.cos(yaw)
        dy = speed * math.sin(yaw)
        dyaw = speed * math.tan(steer) / self.wheel_base
        dv = accel - self.speed_drag * speed
        if speed <= 0.0 and dv < 0.0:
            dv = 0.0
        return np.array([dx, dy, dyaw, dv], dtype=float)


class TrajectoryRecorder:
    def __init__(self) -> None:
        self.path_errors: list[float] = []
        self.speeds: list[float] = []
        self.steering_abs: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        state = np.asarray(snapshot.continuous_states["vehicle"], dtype=float)
        command = np.asarray(snapshot.outputs["controller"]["command"], dtype=float)
        self.path_errors.append(float(snapshot.outputs["vehicle"]["path_error"]))
        self.speeds.append(float(state[3]))
        self.steering_abs.append(abs(float(command[1])))


def build_system(
    *,
    path_radius: float = 20.0,
    speed_ref: float = 8.0,
    controller_sample_time: float = 0.04,
    wheel_base: float = 2.8,
    lookahead_gain: float = 0.6,
    min_lookahead: float = 4.0,
    max_lookahead: float = 12.0,
    speed_gain: float = 1.6,
    speed_feedforward_gain: float = 0.12,
    max_accel: float = 2.5,
    max_steer: float = 0.55,
    speed_drag: float = 0.12,
    initial_state: tuple[float, float, float, float] = (24.0, -6.0, 2.4, 0.0),
) -> System:
    system = System("vehicle_path_tracking")
    system.add_block("speed_ref", SpeedSetpoint(speed_ref))
    system.add_block(
        "controller",
        PurePursuitController(
            sample_time=controller_sample_time,
            wheel_base=wheel_base,
            path_radius=path_radius,
            lookahead_gain=lookahead_gain,
            min_lookahead=min_lookahead,
            max_lookahead=max_lookahead,
            speed_gain=speed_gain,
            speed_feedforward_gain=speed_feedforward_gain,
            max_accel=max_accel,
            max_steer=max_steer,
        ),
    )
    system.add_block(
        "vehicle",
        KinematicBicycle(
            initial_state=initial_state,
            wheel_base=wheel_base,
            path_radius=path_radius,
            speed_drag=speed_drag,
        ),
    )

    system.connect("speed_ref.speed_ref", "controller.speed_ref")
    system.connect("vehicle.state", "controller.state")
    system.connect("controller.command", "vehicle.command")
    return system


def print_summary(recorder: TrajectoryRecorder) -> None:
    print("vehicle path tracking example")
    print("controller: sampled pure-pursuit steering + speed hold")
    print(f"initial path error = {recorder.path_errors[0]:.3f} m")
    print(f"final path error = {recorder.path_errors[-1]:.3f} m")
    print(f"final speed = {recorder.speeds[-1]:.3f} m/s")
    print(f"max steering magnitude = {max(recorder.steering_abs):.3f} rad")


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

    recorder = TrajectoryRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
