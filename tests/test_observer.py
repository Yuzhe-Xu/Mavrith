from __future__ import annotations

from pylink import Block, PortSpec, SimulationConfig, Simulator, StepSnapshot, System


class Constant(Block):
    outputs = (PortSpec.output("out"),)

    def __init__(self, value):
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Recorder:
    def __init__(self) -> None:
        self.started = False
        self.ended = False
        self.snapshots: list[StepSnapshot] = []

    def on_simulation_start(self, plan, config) -> None:
        self.started = True

    def on_step(self, snapshot: StepSnapshot) -> None:
        self.snapshots.append(snapshot)

    def on_simulation_end(self, result) -> None:
        self.ended = True


def test_observer_receives_step_snapshots():
    system = System()
    system.add_block("src", Constant(3))
    observer = Recorder()

    result = Simulator().run(
        system,
        SimulationConfig(start=0.0, stop=0.2, dt=0.1),
        observer=observer,
    )

    assert observer.started is True
    assert observer.ended is True
    assert [snapshot.time for snapshot in observer.snapshots] == [0.0, 0.1, 0.2]
    assert observer.snapshots[-1].outputs["src"]["out"] == 3
    assert result.final_outputs["src"]["out"] == 3
