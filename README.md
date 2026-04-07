# pylink

`pylink` is a lightweight Python framework for building block-based systems in code.
It focuses on graph orchestration, execution order, state management, fixed-step
simulation, and continuous-time solver integration through SciPy.

The framework does not ship with prebuilt control or signal-processing blocks.
Users define their own blocks by inheriting the provided base classes.

## Highlights

- Pure Python DSL for system construction
- Stateless, discrete-state, and continuous-state block base classes
- Connection validation and algebraic loop detection
- Deterministic fixed-step simulation loop
- SciPy-backed continuous solver interface
- Observer hooks for tracing and debugging

## Quick Example

```python
from pylink import (
    Block,
    ContinuousBlock,
    PortSpec,
    SimulationConfig,
    Simulator,
    System,
)


class Constant(Block):
    outputs = (PortSpec.output("out"),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Integrator(ContinuousBlock):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("x"),)

    def __init__(self, initial: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return inputs["u"]


system = System("demo")
system.add_block("source", Constant(1.0))
system.add_block("plant", Integrator(initial=0.0))
system.connect("source.out", "plant.u")

result = Simulator().run(
    system,
    SimulationConfig(start=0.0, stop=1.0, dt=0.1),
)

print(result.time_points[-1], result.final_continuous_states["plant"])
```

## Notes

- Continuous states must be numeric and SciPy-compatible.
- Discrete states can be any Python object.
- Algebraic loops made entirely of direct-feedthrough paths are rejected.
- Feedback through stateful blocks is supported.

## More Examples

- Closed-loop control example: `uv run python examples/closed_loop.py`
- Water cooling example based on Newton's law of cooling: `uv run python examples/water_cooling.py`
