from __future__ import annotations

"""Programmatic benchmark for large hierarchical subsystem expansion."""

from dataclasses import dataclass
from time import perf_counter

from pylink import Block, PortSpec, SignalSpec, SimulationConfig, Simulator, Subsystem, System

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class Constant(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Passthrough(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["u"]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    levels: int
    width: int
    leaf_blocks: int

    @property
    def leaf_count(self) -> int:
        return self.leaf_blocks * (self.width ** (self.levels - 1))


def build_leaf_chain(*, leaf_blocks: int) -> Subsystem:
    subsystem = Subsystem(f"leaf_chain_{leaf_blocks}")
    for index in range(leaf_blocks):
        name = f"node_{index}"
        subsystem.add_block(name, Passthrough())
        if index == 0:
            subsystem.expose_input("u", f"{name}.u", spec=FLOAT_SCALAR)
        else:
            subsystem.connect(f"node_{index - 1}.y", f"{name}.u")
    subsystem.expose_output(f"node_{leaf_blocks - 1}.y", "y", spec=FLOAT_SCALAR)
    return subsystem


def build_hierarchy(*, levels: int, width: int, leaf_blocks: int) -> Subsystem:
    if levels <= 1:
        return build_leaf_chain(leaf_blocks=leaf_blocks)

    child = build_hierarchy(levels=levels - 1, width=width, leaf_blocks=leaf_blocks)
    subsystem = Subsystem(f"hierarchy_{levels}")
    previous_output = ""
    for index in range(width):
        name = f"segment_{index}"
        subsystem.add_subsystem(name, child)
        if index == 0:
            subsystem.expose_input("u", f"{name}.u", spec=FLOAT_SCALAR)
        else:
            subsystem.connect(previous_output, f"{name}.u")
        previous_output = f"{name}.y"
    subsystem.expose_output(previous_output, "y", spec=FLOAT_SCALAR)
    return subsystem


def build_system(case: BenchmarkCase) -> System:
    system = System(f"large_hierarchy_{case.leaf_count}")
    system.add_block("source", Constant(1.0))
    system.add_subsystem(
        "network",
        build_hierarchy(levels=case.levels, width=case.width, leaf_blocks=case.leaf_blocks),
    )
    system.add_block("sink", Passthrough())
    system.connect("source.out", "network.u")
    system.connect("network.y", "sink.u")
    return system


def run_case(case: BenchmarkCase) -> dict[str, float]:
    simulator = Simulator()
    system = build_system(case)
    config = SimulationConfig(start=0.0, stop=0.1, dt=0.1)

    compile_start = perf_counter()
    plan = simulator.compile(system)
    compile_elapsed = perf_counter() - compile_start

    run_start = perf_counter()
    result = simulator.run(system, config)
    run_elapsed = perf_counter() - run_start

    return {
        "leaf_count": float(case.leaf_count),
        "flat_blocks": float(len(plan.system.blocks)),
        "compile_seconds": compile_elapsed,
        "run_seconds": run_elapsed,
        "final_output": float(result.final_outputs["sink"]["y"]),
    }


def main() -> None:
    cases = [
        BenchmarkCase(levels=3, width=5, leaf_blocks=4),
        BenchmarkCase(levels=4, width=5, leaf_blocks=8),
        BenchmarkCase(levels=5, width=5, leaf_blocks=8),
    ]
    print("large hierarchical benchmark")
    for case in cases:
        metrics = run_case(case)
        print(
            f"leafs={int(metrics['leaf_count'])} blocks={int(metrics['flat_blocks'])} "
            f"compile={metrics['compile_seconds']:.4f}s run={metrics['run_seconds']:.4f}s "
            f"final={metrics['final_output']:.3f}"
        )


if __name__ == "__main__":
    main()
