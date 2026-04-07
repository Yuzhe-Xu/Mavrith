from __future__ import annotations

from pylink import (
    Block,
    ContinuousBlock,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    Simulator,
    StepSnapshot,
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class RoomTemperatureSource(Block):
    """Stateless source block: the room temperature is a fixed parameter."""

    outputs = (PortSpec.output("room_temp", spec=FLOAT_SCALAR),)

    def __init__(self, room_temperature: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.room_temperature = room_temperature

    def output(self, ctx, inputs):
        return self.room_temperature


class TemperatureDifference(Block):
    """Stateless algebraic block: delta depends on current temperatures."""

    inputs = (
        PortSpec.input("water_temp", spec=FLOAT_SCALAR),
        PortSpec.input("room_temp", spec=FLOAT_SCALAR),
    )
    outputs = (PortSpec.output("delta_t", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["water_temp"] - inputs["room_temp"]


class CoolingCup(ContinuousBlock):
    """Continuous block: the water temperature is the continuous state."""

    inputs = (PortSpec.input("delta_t", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("water_temp", spec=FLOAT_SCALAR),)

    def __init__(self, initial_temperature: float, cooling_rate: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def initial_continuous_state(self):
        return self.initial_temperature

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return -self.cooling_rate * inputs["delta_t"]


class TemperatureRecorder:
    def __init__(self) -> None:
        self.temperature_by_time: dict[float, float] = {}

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.temperature_by_time[round(snapshot.time, 6)] = snapshot.continuous_states["cup"]


def build_system(
    *,
    initial_water_temperature: float = 80.0,
    room_temperature: float = 20.0,
    cooling_rate: float = 0.10,
) -> System:
    system = System("water_cooling")
    system.add_block("room", RoomTemperatureSource(room_temperature))
    system.add_block(
        "cup",
        CoolingCup(
            initial_temperature=initial_water_temperature,
            cooling_rate=cooling_rate,
        ),
    )
    system.add_block("delta", TemperatureDifference())

    system.connect("room.room_temp", "delta.room_temp")
    system.connect("cup.water_temp", "delta.water_temp")
    system.connect("delta.delta_t", "cup.delta_t")
    return system


def print_summary(recorder: TemperatureRecorder) -> None:
    sample_times = [0.0, 5.0, 10.0, 20.0, 30.0]

    print("water cooling example")
    print("model: dT/dt = -k * (T - T_room)")
    print("selected temperatures:")
    for time_point in sample_times:
        temperature = recorder.temperature_by_time[round(time_point, 6)]
        print(f"  t = {time_point:>4.1f} min -> T = {temperature:>6.2f} C")

    ordered_times = sorted(recorder.temperature_by_time)
    ordered_temperatures = [recorder.temperature_by_time[time_point] for time_point in ordered_times]
    monotonic_cooling = all(
        next_temp <= current_temp + 1e-9
        for current_temp, next_temp in zip(ordered_temperatures, ordered_temperatures[1:])
    )

    print("monotonic cooling =", monotonic_cooling)
    print("final temperature =", ordered_temperatures[-1])


def main() -> None:
    system = build_system()
    config = SimulationConfig(start=0.0, stop=30.0, dt=0.01)
    simulator = Simulator()
    report = simulator.validate(system, config)

    if not report.is_valid:
        for diagnostic in report.diagnostics:
            print(f"{diagnostic.code}: {diagnostic.message}")
            print(f"  fix: {diagnostic.suggestion}")
        return

    recorder = TemperatureRecorder()
    simulator.run(system, config, observer=recorder)
    print("execution order =", report.summary()["execution_order"])
    print_summary(recorder)


if __name__ == "__main__":
    main()
