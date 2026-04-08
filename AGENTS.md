# pylink Guide for Coding Agents

This file is for LLM-based coding agents such as Codex. It explains how to use
`pylink` correctly when generating code that builds or simulates systems with
blocks and connections.

## What `pylink` is

`pylink` is a lightweight block-based simulation framework for Python.

It provides:

- block base classes
- port declarations
- signal dtype/shape declarations with `SignalSpec`
- graph construction
- connection validation
- signal compatibility validation
- execution ordering
- fixed-step simulation
- continuous-state integration through SciPy
- observer hooks for collecting results

It does **not** provide:

- built-in domain blocks such as gain, sum, constant, PID, integrator library
- GUI editing
- JSON/YAML model import
- plugin discovery
- algebraic loop solving
- plotting

When using this library, always assume that **domain logic must be implemented
by the user as custom blocks**.

## Public API to import

Import from the public package namespace:

```python
from pylink import (
    Block,
    ContinuousBlock,
    Diagnostic,
    DiscreteBlock,
    PortSpec,
    SignalSpec,
    SimulationConfig,
    SimulationObserver,
    Simulator,
    StepSnapshot,
    Subsystem,
    System,
    ValidationReport,
)
```

Do not import from internal modules such as `pylink.core` unless the caller
explicitly asks for internals.

## Core modeling rules

When generating a system, first decide what kind of block each component is.

### Use `Block` when

- the block is stateless
- the block output is a pure algebraic transform of current inputs
- the block output is a constant or parameter-driven value

Examples:

- constant source
- adder
- subtractor
- unit conversion
- signal routing

### Use `DiscreteBlock` when

- the block has sampled state
- the state updates only on discrete sample hits
- the output is based on stored discrete state
- the block should run on a periodic clock described by `sample_time`, optional `offset`, and optional `priority`

Examples:

- sampled controller
- counter
- zero-order-hold style stateful logic
- sampled estimator

### Use `ContinuousBlock` when

- the block has continuous state
- the model is governed by a differential equation
- the state evolves over time between sample hits

Examples:

- thermal state
- mass-spring state
- fluid level
- first-order plant

## Port declaration rules

Each block declares ports as class attributes using `PortSpec`.

```python
FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class MyBlock(Block):
    inputs = (
        PortSpec.input("u", spec=FLOAT_SCALAR),
        PortSpec.input("bias", required=False, spec=FLOAT_SCALAR),
    )
    outputs = (
        PortSpec.output("y", spec=FLOAT_SCALAR),
    )
```

Rules:

- port names must be non-empty
- port names must not contain `.`
- use `SignalSpec` when the port has a known dtype and shape
- prefer `spec=SignalSpec(...)` over the older `data_type=` convenience form
- `SignalSpec.dtype` may be `bool`, `int`, `float`, `complex`, `object`, or `None`
- `SignalSpec.shape` may be `()`, `(n,)`, `(m, n)`, or `None`
- when both ends declare `dtype` or `shape`, they must match exactly
- unspecified `dtype` or `shape` acts as a wildcard
- do not assume implicit casts, scalar-to-vector broadcasting, or shape promotion
- one input port can only have one upstream connection
- one output port can fan out to many downstream inputs
- if an input is required, it must be connected before simulation
- if multiple outputs are needed, declare multiple output ports explicitly

## `direct_feedthrough` is important

`direct_feedthrough` tells the compiler whether a block output depends on
current-step inputs.

### Set `direct_feedthrough=True` when

- `output()` reads current inputs to compute the current output
- the block is algebraic

Example:

```python
class Add(Block):
    inputs = (PortSpec.input("a"), PortSpec.input("b"))
    outputs = (PortSpec.output("sum"),)

    def __init__(self) -> None:
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["a"] + inputs["b"]
```

### Set `direct_feedthrough=False` when

- `output()` depends only on internal state or constants
- the block output should be available even in feedback loops
- the block is a stateful continuous or discrete block

Example:

```python
class Counter(DiscreteBlock):
    outputs = (PortSpec.output("count"),)

    def __init__(self) -> None:
        super().__init__(sample_time=1.0, offset=0.0, direct_feedthrough=False)

    def initial_discrete_state(self):
        return 0

    def output(self, ctx, inputs):
        return ctx.discrete_state
```

Important:

- feedback through only direct-feedthrough blocks creates an algebraic loop and
  is rejected
- feedback through a `DiscreteBlock` or `ContinuousBlock` is allowed because the
  output comes from stored state
- when `direct_feedthrough=False`, write `output()` so it does not require
  current-step inputs

## Block method contracts

### `Block.output(ctx, inputs) -> outputs`

This computes the block output at the current evaluation point.

Rules:

- if the block declares no outputs, return `None` or `{}`
- if the block declares exactly one output, you may return the raw value
- if the block declares multiple outputs, return a mapping with every declared
  port name
- do not return undeclared outputs
- do not omit declared outputs

Examples:

Single output:

```python
def output(self, ctx, inputs):
    return inputs["u"] * 2.0
```

Multiple outputs:

```python
def output(self, ctx, inputs):
    value = inputs["u"]
    return {
        "positive": max(value, 0.0),
        "negative": min(value, 0.0),
    }
```

### `DiscreteBlock.initial_discrete_state()`

Return the initial discrete state. This may be any Python object.

Examples:

- `0`
- `{"integral": 0.0, "last_error": 0.0}`
- a tuple or dataclass-like structure

### `DiscreteBlock.update_state(ctx, inputs, state) -> next_state`

This computes the next discrete state on sample hits.

Rules:

- it is called only when the block sample time hits the global simulation clock
- the `state` argument is the current committed discrete state
- the returned value becomes visible as soon as that discrete rate group commits at the current sample hit
- write it as a pure state transition when possible

## Discrete clock semantics

Each `DiscreteBlock` may define:

- `sample_time`: positive period
- `offset`: absolute offset with `0 <= offset < sample_time`
- `priority`: optional integer priority; lower values run first

Important:

- sample hits follow `t = n * sample_time + offset`
- when two rate groups hit at the same time, lower `priority` runs first
- if a rate group does not declare `priority`, the compiler assigns a stable one
- after one discrete rate group commits, downstream direct-feedthrough logic is re-evaluated before later groups at the same time run
- cross-rate connections are allowed, but `validate()` reports them as warnings with explicit classifications

### `ContinuousBlock.initial_continuous_state()`

Return the initial continuous state.

Rules:

- it must not be `None`
- it must be numeric and SciPy-compatible
- its shape must remain consistent across the simulation

Valid examples:

- `0.0`
- `1.5`
- `np.array([0.0, 1.0])`

### `ContinuousBlock.derivative(ctx, inputs, state) -> state_derivative`

This returns the time derivative of the continuous state.

Rules:

- the derivative shape must match the continuous state shape
- it should be side-effect free
- it may be called multiple times per fixed step by the solver
- do not perform logging, mutation, or irreversible actions here
- the derivative should only describe the system dynamics

Example:

```python
def derivative(self, ctx, inputs, state):
    return -0.1 * state + inputs["u"]
```

## `ExecutionContext`

All block callbacks receive an `ExecutionContext`.

Useful fields:

- `ctx.block_name`
- `ctx.time`
- `ctx.step_index`
- `ctx.dt`
- `ctx.parameters`
- `ctx.discrete_state`
- `ctx.continuous_state`

Use `ctx.discrete_state` and `ctx.continuous_state` rather than storing mutable
runtime state on the block instance.

## Building a system

Create a `System`, add blocks or `Subsystem` objects by unique names, then
connect ports using `"<component>.<port>"`.

```python
system = System("demo")
system.add_block("source", ConstantSource(1.0))
system.add_block("plant", Integrator(initial=0.0))
system.connect("source.out", "plant.u")
```

Rules:

- component names must be unique
- component names must not contain `.` or `/`
- `system.connect("src.out", "dst.in")` is directional
- a target input cannot have more than one source
- a source output may connect to many targets

## Building a subsystem

Use `Subsystem` when a reusable or nested fragment should expose a clean
boundary to the outside world.

```python
controller = Subsystem("controller")
controller.add_block("gain", Gain(2.0))
controller.expose_input("u", "gain.u", spec=FLOAT_SCALAR)
controller.expose_output("gain.y", "y", spec=FLOAT_SCALAR)
```

Rules:

- `expose_input(name, target, ...)` binds one external subsystem input to one or more internal input targets
- repeated `expose_input()` calls with the same name are allowed only when `required` and `spec` match
- `expose_output(source, name, ...)` binds one external subsystem output to exactly one internal output source
- subsystem ports must not contain `.` or `/`
- subsystem hierarchy is organizational only; it is flattened before compile and run

## Running a simulation

Use `Simulator().run(...)` with a `SimulationConfig`.

```python
result = Simulator().run(
    system,
    SimulationConfig(start=0.0, stop=10.0, dt=0.1),
)
```

`SimulationConfig` fields:

- `start`: simulation start time
- `stop`: simulation stop time
- `dt`: fixed global simulation step
- `solver`: optional continuous solver, defaults to `SciPySolver()`

Rules:

- `dt` must be positive
- `stop` must be greater than or equal to `start`
- `(stop - start)` must be an integer multiple of `dt`
- for each `DiscreteBlock`, `sample_time / dt` must be an integer
- for each `DiscreteBlock`, `offset / dt` must be an integer
- `start`, `dt`, and `offset` must align so sample hits land on the simulation grid
- if a discrete controller is used in an example, choose `sample_time`, `offset`, and `dt` together so the sample hits are exact

## Runtime semantics

The simulation loop is fixed-step and deterministic.

At a high level:

1. evaluate outputs from the current committed states at the current time
2. run all discrete rate groups that hit at that time, in priority order
3. after each rate-group commit, re-evaluate downstream direct-feedthrough outputs
4. record the final visible outputs for that time point
5. advance continuous states over one fixed step using those final visible discrete outputs
6. repeat at the next simulation time point

Important consequences:

- discrete state updates become visible immediately after their rate group commits
- a discrete block output typically reflects the stored state, not the just-read
  input
- lower-priority tasks at the same time can observe updates committed by higher-priority tasks
- continuous blocks are integrated by SciPy within each fixed step
- output and input `SignalSpec` values are checked once at `t=start` as a runtime safeguard
- because the solver may evaluate intermediate candidate times, `output()` and
  `derivative()` for continuous blocks should be side-effect free

## Observers

You can pass an observer object to `Simulator.run(...)`.

The object may implement any of these methods:

- `on_simulation_start(plan, config)`
- `on_step(snapshot)`
- `on_simulation_error(error)`
- `on_simulation_end(result)`

Example:

```python
class Recorder:
    def __init__(self) -> None:
        self.values = []

    def on_step(self, snapshot):
        self.values.append((snapshot.time, snapshot.outputs["src"]["out"]))
```

`StepSnapshot` contains:

- `time`
- `step_index`
- `outputs`
- `discrete_states`
- `continuous_states`

## `SimulationResult`

The result object contains:

- `time_points`
- `final_outputs`
- `final_discrete_states`
- `final_continuous_states`

Use it for end-state assertions and lightweight result inspection.

## `ValidationReport`

Call `Simulator.validate(system, config)` before `run(...)` when you want a
structured, non-executing validation pass.

The report contains:

- `is_valid`
- `diagnostics`
- a deterministic model summary with blocks, ports, connections, execution
  order, stateful blocks, time-grid constraints, resolved rate groups,
  cross-rate connections, and structured `signal_spec` summaries

Each `Diagnostic` includes:

- a stable `code`
- a human-readable `message`
- a short `suggestion`
- location fields such as block, port, endpoint, connection, and time when
  available

Common validation codes now include:

- graph issues such as missing ports, duplicate input connections, and algebraic loops
- time-grid issues such as incompatible discrete sample times
- time-grid issues such as incompatible discrete offsets
- `SignalSpec` issues such as `INCOMPATIBLE_PORT_TYPE`,
  `INCOMPATIBLE_PORT_SHAPE`, `OUTPUT_TYPE_MISMATCH`, and
  `INPUT_SHAPE_MISMATCH`

## Recommended coding pattern for agents

When asked to model a system with `pylink`, follow this order:

1. identify signals
2. identify stateful quantities
3. map each component to `Block`, `DiscreteBlock`, or `ContinuousBlock`
4. declare ports explicitly, including `SignalSpec` when known
5. decide `direct_feedthrough` correctly
6. choose whether repeated structure should become a `Subsystem`
7. build the `System` and any reusable `Subsystem` layers
8. connect using exact `"<component>.<port>"` strings
9. validate with `Simulator.validate(system, config)` and inspect diagnostics first
10. run with `SimulationConfig`
11. optionally add an observer for inspection

## Minimal templates

### Stateless transform

```python
FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class Gain(Block):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self, gain: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.gain = gain

    def output(self, ctx, inputs):
        return self.gain * inputs["u"]
```

### Discrete-state block

```python
FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class Accumulator(DiscreteBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("x", spec=FLOAT_SCALAR),)

    def __init__(
        self,
        sample_time: float,
        *,
        offset: float = 0.0,
        priority: int | None = None,
    ) -> None:
        super().__init__(
            sample_time=sample_time,
            offset=offset,
            priority=priority,
            direct_feedthrough=False,
        )

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return ctx.discrete_state

    def update_state(self, ctx, inputs, state):
        return state + inputs["u"]
```

### Continuous-state block

```python
FLOAT_SCALAR = SignalSpec(dtype="float", shape=())


class FirstOrderLag(ContinuousBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("x", spec=FLOAT_SCALAR),)

    def __init__(self, initial: float, tau: float) -> None:
        super().__init__(direct_feedthrough=False)
        self.initial = initial
        self.tau = tau

    def initial_continuous_state(self):
        return self.initial

    def output(self, ctx, inputs):
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        return (inputs["u"] - state) / self.tau
```

## Common mistakes to avoid

Do not:

- assume there are built-in blocks such as `Constant`, `Gain`, `Sum`, or `PID`
- connect two sources into one input port
- use undeclared port names
- set `direct_feedthrough=False` on a block whose `output()` reads current inputs
- set `direct_feedthrough=True` on a state-output block unless that output really
  depends on current inputs
- mutate block instance attributes as runtime state
- write side effects inside `ContinuousBlock.derivative()`
- expect plotting or GUI features from this library
- expect algebraic loops to be solved automatically

## Preferred style for generated examples

When generating example code for users:

- keep the domain logic inside clearly named custom blocks
- add short comments explaining why a block is stateless, discrete, or continuous
- prefer declaring `SignalSpec` on every externally meaningful signal
- keep one `build_system()` function when the example is non-trivial
- keep one `main()` entry point for runnable examples
- print a few interpretable outputs instead of dumping raw objects

## Reference examples in this repository

See these examples before inventing new patterns:

- `examples/closed_loop.py`
- `examples/water_cooling.py`
- `examples/vehicle_path_tracking.py`
- `examples/cruise_control.py`
- `examples/mass_spring_damper.py`
- `examples/multirate_offset_priority.py`

These show the intended style for:

- pure code DSL
- custom block authoring
- stateful feedback modeling
- continuous-time modeling
- vector-valued signals and multi-output plants
- actuator saturation and disturbance rejection
