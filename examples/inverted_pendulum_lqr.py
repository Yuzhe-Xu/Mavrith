from __future__ import annotations

"""Digital LQR stabilization of the CTMS inverted pendulum benchmark.

Reference:
- https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from pylink import (
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

M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3
p = I * (M + m) + M * m * l**2

PENDULUM_A = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -((I + m * l**2) * b) / p, (m**2 * g * l**2) / p, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -(m * l * b) / p, m * g * l * (M + m) / p, 0.0],
    ],
    dtype=float,
)
PENDULUM_B = np.array([0.0, (I + m * l**2) / p, 0.0, m * l / p], dtype=float)


def _discrete_lqr(sample_time: float) -> tuple[np.ndarray, float]:
    c = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    a_d, b_d, _, _, _ = cont2discrete(
        (PENDULUM_A, PENDULUM_B.reshape(-1, 1), c, np.zeros((1, 1), dtype=float)),
        sample_time,
    )
    q = np.diag([25.0, 1.5, 180.0, 15.0])
    r = np.array([[0.12]], dtype=float)
    p_matrix = solve_discrete_are(a_d, b_d, q, r)
    gain = np.linalg.solve(b_d.T @ p_matrix @ b_d + r, b_d.T @ p_matrix @ a_d)
    closed_loop = a_d - b_d @ gain
    prefilter = 1.0 / float((c @ np.linalg.solve(np.eye(a_d.shape[0]) - closed_loop, b_d)).item())
    return gain.reshape(-1), prefilter


class CartReference(Block):
    outputs = (PortSpec.output("cart_ref", spec=FLOAT_SCALAR),)

    def __init__(self, cart_ref: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.cart_ref = cart_ref

    def output(self, ctx, inputs):
        return self.cart_ref


class LQRController(DiscreteBlock):
    inputs = (
        PortSpec.input("cart_ref", spec=FLOAT_SCALAR),
        PortSpec.input("state", spec=STATE_VECTOR),
    )
    outputs = (PortSpec.output("raw_force", spec=FLOAT_SCALAR),)

    def __init__(self, *, sample_time: float, max_force: float) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)
        self.max_force = max_force
        self.gain, self.prefilter = _discrete_lqr(sample_time)

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return float(ctx.discrete_state)

    def update_state(self, ctx, inputs, state):
        state_vector = np.asarray(inputs["state"], dtype=float)
        cart_ref = float(inputs["cart_ref"])
        raw_force = float(self.prefilter * cart_ref - float(self.gain @ state_vector))
        return float(np.clip(raw_force, -self.max_force, self.max_force))


class ForceLimiter(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("force", spec=FLOAT_SCALAR),)

    def __init__(self, max_force: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.max_force = max_force

    def output(self, ctx, inputs):
        return float(np.clip(float(inputs["u"]), -self.max_force, self.max_force))


class LinearizedInvertedPendulum(ContinuousBlock):
    inputs = (PortSpec.input("force", spec=FLOAT_SCALAR),)
    outputs = (
        PortSpec.output("state", spec=STATE_VECTOR),
        PortSpec.output("cart_position", spec=FLOAT_SCALAR),
        PortSpec.output("angle", spec=FLOAT_SCALAR),
    )

    def __init__(self, initial_state: tuple[float, float, float, float]) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_state = np.asarray(initial_state, dtype=float)

    def initial_continuous_state(self):
        return self.initial_state.copy()

    def output(self, ctx, inputs):
        state = np.asarray(ctx.continuous_state, dtype=float)
        return {
            "state": state.copy(),
            "cart_position": float(state[0]),
            "angle": float(state[2]),
        }

    def derivative(self, ctx, inputs, state):
        state_vector = np.asarray(state, dtype=float)
        force = float(inputs["force"])
        return PENDULUM_A @ state_vector + PENDULUM_B * force


class PendulumRecorder:
    def __init__(self) -> None:
        self.cart_positions: list[float] = []
        self.angles: list[float] = []
        self.forces: list[float] = []

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.cart_positions.append(float(snapshot.outputs["plant/dynamics"]["cart_position"]))
        self.angles.append(float(snapshot.outputs["plant/dynamics"]["angle"]))
        self.forces.append(float(snapshot.outputs["controller/limit"]["force"]))


def build_controller_subsystem(*, sample_time: float, max_force: float) -> Subsystem:
    controller = Subsystem("lqr_controller")
    controller.add_block("lqr", LQRController(sample_time=sample_time, max_force=max_force))
    controller.add_block("limit", ForceLimiter(max_force=max_force))
    controller.connect("lqr.raw_force", "limit.u")
    controller.expose_input("cart_ref", "lqr.cart_ref", spec=FLOAT_SCALAR)
    controller.expose_input("state", "lqr.state", spec=STATE_VECTOR)
    controller.expose_output("limit.force", "force", spec=FLOAT_SCALAR)
    return controller


def build_plant_subsystem(*, initial_state: tuple[float, float, float, float]) -> Subsystem:
    plant = Subsystem("pendulum_plant")
    plant.add_block("dynamics", LinearizedInvertedPendulum(initial_state=initial_state))
    plant.expose_input("force", "dynamics.force", spec=FLOAT_SCALAR)
    plant.expose_output("dynamics.state", "state", spec=STATE_VECTOR)
    plant.expose_output("dynamics.cart_position", "cart_position", spec=FLOAT_SCALAR)
    plant.expose_output("dynamics.angle", "angle", spec=FLOAT_SCALAR)
    return plant


def build_system(
    *,
    cart_ref: float = 0.2,
    controller_sample_time: float = 0.02,
    max_force: float = 10.0,
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.0),
) -> System:
    system = System("inverted_pendulum_lqr")
    system.add_block("reference", CartReference(cart_ref))
    system.add_subsystem(
        "controller",
        build_controller_subsystem(sample_time=controller_sample_time, max_force=max_force),
    )
    system.add_subsystem("plant", build_plant_subsystem(initial_state=initial_state))

    system.connect("reference.cart_ref", "controller.cart_ref")
    system.connect("plant.state", "controller.state")
    system.connect("controller.force", "plant.force")
    return system


def print_summary(recorder: PendulumRecorder) -> None:
    print("inverted pendulum lqr example")
    print(f"final cart position = {recorder.cart_positions[-1]:.4f} m")
    print(f"final angle = {recorder.angles[-1]:.4f} rad")
    print(f"max |force| = {max(abs(value) for value in recorder.forces):.4f} N")


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=6.0, dt=0.01)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = PendulumRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
