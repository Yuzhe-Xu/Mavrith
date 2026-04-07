# pylink Guide for Coding Agents

This file is for LLM-based coding agents such as Codex. It explains how to use
`pylink` correctly when generating code that builds or simulates systems with
blocks and connections.

## What `pylink` is

`pylink` is a lightweight block-based simulation framework for Python.

It provides:

- block base classes
- port declarations
- graph construction
- connection validation
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
    DiscreteBlock,
    PortSpec,
    SimulationConfig,
    SimulationObserver,
    Simulator,
    StepSnapshot,
    System,
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
class MyBlock(Block):
    inputs = (
        PortSpec.input("u"),
        PortSpec.input("bias", required=False),
    )
    outputs = (
        PortSpec.output("y"),
    )
```

Rules:

- port names must be non-empty
- port names must not contain `.`
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
        super().__init__(sample_time=1.0, direct_feedthrough=False)

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
- the returned value becomes visible after the step commit
- write it as a pure state transition when possible

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

Create a `System`, add blocks by unique names, then connect ports using
`"<block>.<port>"`.

```python
system = System("demo")
system.add_block("source", ConstantSource(1.0))
system.add_block("plant", Integrator(initial=0.0))
system.connect("source.out", "plant.u")
```

Rules:

- block names must be unique
- block names must not contain `.`
- `system.connect("src.out", "dst.in")` is directional
- a target input cannot have more than one source
- a source output may connect to many targets

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

## Runtime semantics

The simulation loop is fixed-step and deterministic.

At a high level:

1. evaluate outputs at the current time
2. resolve current inputs from those outputs
3. advance continuous states over one step with the solver
4. update discrete states on sample hits
5. commit next states
6. evaluate outputs at the next time point

Important consequences:

- discrete state updates become visible after the step commit
- a discrete block output typically reflects the stored state, not the just-read
  input
- continuous blocks are integrated by SciPy within each fixed step
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

## Recommended coding pattern for agents

When asked to model a system with `pylink`, follow this order:

1. identify signals
2. identify stateful quantities
3. map each component to `Block`, `DiscreteBlock`, or `ContinuousBlock`
4. declare ports explicitly
5. decide `direct_feedthrough` correctly
6. build the `System`
7. connect using exact `"<block>.<port>"` strings
8. run with `SimulationConfig`
9. optionally add an observer for inspection

## Minimal templates

### Stateless transform

```python
class Gain(Block):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("y"),)

    def __init__(self, gain: float) -> None:
        super().__init__(direct_feedthrough=True)
        self.gain = gain

    def output(self, ctx, inputs):
        return self.gain * inputs["u"]
```

### Discrete-state block

```python
class Accumulator(DiscreteBlock):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("x"),)

    def __init__(self, sample_time: float) -> None:
        super().__init__(sample_time=sample_time, direct_feedthrough=False)

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return ctx.discrete_state

    def update_state(self, ctx, inputs, state):
        return state + inputs["u"]
```

### Continuous-state block

```python
class FirstOrderLag(ContinuousBlock):
    inputs = (PortSpec.input("u"),)
    outputs = (PortSpec.output("x"),)

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
- keep one `build_system()` function when the example is non-trivial
- keep one `main()` entry point for runnable examples
- print a few interpretable outputs instead of dumping raw objects

## Reference examples in this repository

See these examples before inventing new patterns:

- `examples/closed_loop.py`
- `examples/water_cooling.py`

These show the intended style for:

- pure code DSL
- custom block authoring
- stateful feedback modeling
- continuous-time modeling

