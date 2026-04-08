from __future__ import annotations

from time import perf_counter

import pytest

from mavrith import SimulationConfig, Simulator, System

from subsystem_helpers import Constant, Echo, PerformanceCase, build_hierarchical_chain


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    [
        PerformanceCase(levels=3, width=5, leaf_blocks=4),
        PerformanceCase(levels=4, width=5, leaf_blocks=8),
        PerformanceCase(levels=5, width=5, leaf_blocks=8),
    ],
    ids=lambda case: f"{case.leaf_count}_leaf_blocks",
)
def test_large_hierarchical_models_compile_and_run(case: PerformanceCase):
    system = System(f"perf_{case.leaf_count}")
    system.add_block("source", Constant(1.0))
    system.add_subsystem(
        "network",
        build_hierarchical_chain(
            levels=case.levels,
            width=case.width,
            leaf_blocks=case.leaf_blocks,
        ),
    )
    system.add_block("sink", Echo())
    system.connect("source.out", "network.u")
    system.connect("network.y", "sink.u")

    simulator = Simulator()
    config = SimulationConfig(start=0.0, stop=0.1, dt=0.1)

    compile_start = perf_counter()
    plan = simulator.compile(system)
    compile_elapsed = perf_counter() - compile_start

    run_start = perf_counter()
    result = simulator.run(system, config)
    run_elapsed = perf_counter() - run_start

    assert len(plan.system.blocks) == case.leaf_count + 2
    assert result.final_outputs["sink"]["y"] == pytest.approx(1.0, abs=1e-12)
    assert compile_elapsed > 0.0
    assert run_elapsed > 0.0
