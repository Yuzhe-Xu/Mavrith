# pylink

`pylink` is a pure Python framework for building block-based dynamic systems in code.
It is intentionally small: the library provides the simulation kernel, while users
define domain-specific blocks themselves.

The design goal is AI-native authoring. A Codex or Claude Code workflow should be
able to generate a readable model, validate it, and iterate on it without needing
an extra graphical tool or a large built-in block catalog.

## What It Does

- Pure Python DSL for building systems with blocks and connections
- Stateless, discrete-state, and continuous-state block base classes
- Port dtype/shape declarations with `SignalSpec`
- Connection validation and direct-feedthrough algebraic loop detection
- Static and initial-runtime signal compatibility checks
- Deterministic execution ordering
- Fixed-step simulation with SciPy-backed continuous integration
- Structured validation reports for AI-friendly diagnostics
- Observer hooks for tracing and lightweight result collection

## What It Does Not Do

- Ship a large built-in block library
- Provide a GUI or drag-and-drop editor
- Solve algebraic loops automatically
- Replace the need for domain-specific block logic

## Current Status

The current repository implements a usable simulation kernel rather than just a
prototype API. The core workflow is in place:

- build systems in pure Python with explicit ports and connections
- validate graph structure, algebraic-loop constraints, time-grid constraints,
  and signal dtype/shape compatibility
- simulate mixed continuous/discrete systems with deterministic fixed-step
  semantics
- inspect runs through observers, validation summaries, and runnable examples

The repository also includes a growing example suite that exercises several
control patterns:

- scalar feedback loops
- continuous process models
- vector-valued state feedback
- actuator saturation
- disturbance rejection
- off-nominal initial conditions

## Recommended Modeling Pattern

Use one consistent authoring style so humans and AI tools both have a stable target:

1. Declare ports as class attributes with `PortSpec`.
2. Use `Block` for stateless transforms and constant sources.
3. Use `DiscreteBlock` when output comes from sampled state.
4. Use `ContinuousBlock` when output comes from continuous state.
5. Keep one `build_system()` function per non-trivial example.
6. Call `Simulator.validate(...)` before `run(...)`.

Use `SignalSpec` when a port should declare its expected dtype and shape:

- `SignalSpec(dtype="float", shape=())` for a scalar float
- `SignalSpec(dtype="float", shape=(3,))` for a length-3 vector
- `SignalSpec(dtype="float", shape=(2, 2))` for a matrix

`SignalSpec` validation is intentionally strict in v1:

- when both ends declare `dtype`, they must match exactly
- when both ends declare `shape`, they must match exactly
- unspecified fields act as wildcards
- no implicit `int -> float` conversion, broadcasting, or shape promotion

The key modeling decision is `direct_feedthrough`:

- `True`: the current output depends on current inputs.
- `False`: the current output depends only on state or fixed parameters.

Feedback through stateful blocks is valid. Feedback through only direct-feedthrough
blocks is rejected as an algebraic loop.

## Quick Example

```python
from pylink import (
    Block,
    ContinuousBlock,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    Simulator,
    System,
)

FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class Constant(Block):
    outputs = (PortSpec.output("out", spec=FLOAT_SCALAR),)

    def __init__(self, value: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.value = value

    def output(self, ctx, inputs):
        return self.value


class Integrator(ContinuousBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("x", spec=FLOAT_SCALAR),)

    def __init__(self, initial: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return inputs["u"]


def build_system() -> System:
    system = System("demo")
    system.add_block("source", Constant(1.0))
    system.add_block("plant", Integrator(initial=0.0))
    system.connect("source.out", "plant.u")
    return system


system = build_system()
config = SimulationConfig(start=0.0, stop=1.0, dt=0.1)
simulator = Simulator()
report = simulator.validate(system, config)

if report.is_valid:
    result = simulator.run(system, config)
    print(result.time_points[-1], result.final_continuous_states["plant"])
else:
    for diagnostic in report.diagnostics:
        print(diagnostic.code, diagnostic.message)
```

Vector ports use the same API:

```python
from pylink import Block, PortSpec, SignalSpec

VECTOR3 = SignalSpec(dtype="float", shape=(3,))


class VectorGain(Block):
    inputs = (PortSpec.input("u", spec=VECTOR3),)
    outputs = (PortSpec.output("y", spec=VECTOR3),)

    def __init__(self, gain: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.gain = gain

    def output(self, ctx, inputs):
        return [self.gain * value for value in inputs["u"]]
```

## Running From The Repository

From the repo root, either install the package in editable mode or set
`PYTHONPATH=src` when running examples directly.

Examples:

```powershell
python -m pytest -q
$env:PYTHONPATH = "src"
python -m examples.water_cooling
python -m examples.vehicle_path_tracking
```

## Validation And Summaries

Use `Simulator.validate(system, config)` when you want a structured report without
starting the simulation. The report includes:

- Stable diagnostic codes
- Precise block / port / connection context
- Short repair suggestions
- A deterministic model summary with blocks, ports, connections, execution order,
  stateful blocks, time-grid constraints, and structured `signal_spec` metadata

Validation currently happens in two stages:

- compile-time checks for graph structure, algebraic loops, sample-time
  compatibility, and declared port `SignalSpec` compatibility
- initial runtime checks at `t=start` to catch outputs or inputs whose actual
  values do not match the declared `SignalSpec`

This makes it easier for AI tools to inspect, compare, and repair models.

## Cookbook

The repository examples follow the same style on purpose:

- `examples/closed_loop.py`
  - setpoint -> error -> controller -> plant
  - useful starting point for control and signal-flow models
- `examples/water_cooling.py`
  - source + algebraic transform + continuous plant
  - useful starting point for simple continuous-time process models
- `examples/vehicle_path_tracking.py`
  - sampled pure-pursuit-style controller + continuous kinematic bicycle
  - useful for vector signals, state feedback, steering saturation, and off-path recovery
- `examples/cruise_control.py`
  - PI loop + saturation + nonlinear drag + grade disturbance
  - useful for disturbance rejection and actuator limits
- `examples/mass_spring_damper.py`
  - sampled PD control + force saturation + disturbance pulse
  - useful for multi-output plants and settling behavior

Common modifications to these examples:

- Insert a noise source between plant output and measurement input
- Replace the controller block with a different sampled controller
- Add an observer that records selected signals
- Change `sample_time` and `dt` together, then re-run validation

## Notes

- Continuous states must be numeric and SciPy-compatible.
- Discrete states can be any Python object.
- `SignalSpec.dtype` currently supports `bool`, `int`, `float`, `complex`, and `object`.
- `SignalSpec.shape` currently supports scalars `()`, vectors `(n,)`, and matrices `(m, n)`.
- Port signal checks run statically when declarations are available and once at `t=start` as a runtime safeguard.
- `(stop - start)` must be an integer multiple of `dt`.
- Each `DiscreteBlock.sample_time` must be an integer multiple of `dt`.
