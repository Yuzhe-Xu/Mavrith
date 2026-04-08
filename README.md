# Mavrith

Mavrith is a pure Python framework for building block-based dynamic systems in code.
It is intentionally small: the library provides the simulation kernel, while users
define domain-specific blocks themselves.

The design goal is AI-native authoring. A Codex or Claude Code workflow should be
able to generate a readable model, validate it, and iterate on it without needing
an extra graphical tool or a large built-in block catalog.

## Installation

Install the released package from PyPI:

```bash
pip install mavrith
```

Install optional YAML export support:

```bash
pip install "mavrith[yaml]"
```

For local development from GitHub:

```bash
git clone https://github.com/Yuzhe-Xu/Mavrith.git
cd Mavrith
uv sync --extra dev --extra yaml
```

The `examples/` directory is kept in the repository as runnable documentation.
It is included in the source distribution for review and testing, but it is not
installed as part of the runtime wheel.

## What It Does

- Pure Python DSL for building systems with blocks and connections
- Hierarchical `Subsystem` composition with compile-time flattening
- Stateless, discrete-state, and continuous-state block base classes
- Multi-rate discrete execution with `sample_time`, `offset`, and `priority`
- Port dtype/shape declarations with `SignalSpec`
- Connection validation and direct-feedthrough algebraic loop detection
- Static and initial-runtime signal compatibility checks
- Deterministic execution ordering
- Fixed-step simulation with SciPy-backed continuous integration
- Structured validation reports for AI-friendly diagnostics
- Observer hooks for tracing and lightweight result collection
- AI-oriented graph/detail manifest export for large-model navigation

## What It Does Not Do

- Ship a large built-in block library
- Provide a GUI or drag-and-drop editor
- Solve algebraic loops automatically
- Replace the need for domain-specific block logic

## Current Status

The current repository implements a usable simulation kernel rather than just a
prototype API. The core workflow is in place:

- build systems in pure Python with explicit ports and connections
- group reusable model fragments with `Subsystem` and flatten them at compile time
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
5. Use `parameters=...` for stable exported tuning parameters and `description=...` for human-authored semantics when a block or subsystem is externally meaningful.
6. Keep one `build_system()` function per non-trivial example.
7. Call `Simulator.validate(...)` before `run(...)`.

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

## Discrete Rates, Offset, And Priority

`DiscreteBlock` now models a periodic sampled task with three timing fields:

- `sample_time`: the task period
- `offset`: an absolute time offset, interpreted as `t = n * sample_time + offset`
- `priority`: optional explicit task priority, where lower integers run first

Example:

```python
class SampledController(DiscreteBlock):
    inputs = (PortSpec.input("u", spec=FLOAT_SCALAR),)
    outputs = (PortSpec.output("y", spec=FLOAT_SCALAR),)

    def __init__(self) -> None:
        super().__init__(
            sample_time=0.1,
            offset=0.02,
            priority=1,
            direct_feedthrough=False,
        )

    def initial_discrete_state(self):
        return 0.0

    def output(self, ctx, inputs):
        return ctx.discrete_state

    def update_state(self, ctx, inputs, state):
        return inputs["u"]
```

When two discrete tasks hit at the same simulation time:

- lower `priority` runs first
- each committed task group becomes visible immediately
- downstream direct-feedthrough logic is re-evaluated before lower-priority tasks run

Cross-rate connections are allowed by default. `validate()` and `summary()` expose
them explicitly as `slow-to-fast`, `fast-to-slow`, or
`same-period-different-offset` so users and AI tools can inspect the implied
hold semantics.

## Quick Example

```python
from mavrith import (
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
from mavrith import Block, PortSpec, SignalSpec

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
python examples/export_water_cooling_manifest.py
python -m examples.vehicle_path_tracking
```

## Validation And Summaries

Use `Simulator.validate(system, config)` when you want a structured report without
starting the simulation. The report includes:

- Stable diagnostic codes
- Precise block / port / connection context
- Short repair suggestions
- A deterministic model summary with blocks, ports, connections, execution order,
  stateful blocks, time-grid constraints, resolved rate groups, cross-rate
  connections, and structured `signal_spec` metadata

Validation currently happens in two stages:

- compile-time checks for graph structure, algebraic loops, sample-time
  compatibility, and declared port `SignalSpec` compatibility
- initial runtime checks at `t=start` to catch outputs or inputs whose actual
  values do not match the declared `SignalSpec`

This makes it easier for AI tools to inspect, compare, and repair models.

## AI Manifest Export

`mavrith` can now export an AI-oriented graph/detail view without changing the
Python DSL or turning YAML into a second source of truth.

- `build_graph_manifest(system)` returns a topology-first graph manifest
- `build_detail_manifest(system, path=..., config=...)` returns one detail shard
- `write_manifest_bundle(system, out_dir)` writes:
  - `graph.yaml`
  - `detail/index.yaml`
  - one detail YAML per system/block/subsystem

Example:

```python
from mavrith import SimulationConfig, build_detail_manifest, build_graph_manifest, write_manifest_bundle

graph = build_graph_manifest(system)
root_detail = build_detail_manifest(
    system,
    config=SimulationConfig(start=0.0, stop=1.0, dt=0.1),
)
bundle = write_manifest_bundle(system, ".mavrith-ai")
```

If you also want config-aware execution and timing analysis in YAML, write a
separate root detail snapshot:

```python
import yaml

runtime_detail = build_detail_manifest(
    system,
    config=SimulationConfig(start=0.0, stop=1.0, dt=0.1),
)

with open(".mavrith-ai/detail/system.runtime.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(runtime_detail, handle, sort_keys=False, allow_unicode=False)
```

The manifest layer is designed for AI navigation:

- `graph` stays lightweight and only describes children plus local connections
- `detail` includes port contracts, parameters, source locations, descriptions,
  and implementation fingerprints
- repeated exports keep Python as the source of truth and flag human-written
  descriptions with `review_recommended` when implementation changes but the
  description text does not
- manifest generation is explicit; `validate()`, `compile()`, and `run()` never
  write YAML files

Recommended AI reading order when a manifest bundle exists:

1. `graph.yaml`
2. `detail/index.yaml`
3. the relevant `detail/<path>.yaml` shard(s)
4. the referenced Python source files

If a project also writes `detail/system.runtime.yaml`, read it after the
relevant detail shard when execution order, rate groups, or time-grid behavior
matter.

Recommended refresh loop when AI is writing the model:

1. update the Python source
2. re-export the manifest bundle explicitly
3. let the AI inspect `graph` first, then `detail`, then source
4. re-run validation or simulation if behavior changed

For block authors:

- `parameters=...` is the stable exported parameter contract
- `description=...` on `Block(...)` or `Subsystem(...)` is the preferred
  human-authored explanation field
- if no explicit description is provided, block class docstrings are used as a
  fallback

If you want YAML output support in a minimal environment, install the optional
extra:

```powershell
pip install -e ".[yaml]"
```

## Cookbook

The repository examples follow the same style on purpose:

- `examples/export_water_cooling_manifest.py`
  - standalone manifest export script for the water cooling example
  - writes `graph.yaml`, `detail/index.yaml`, component detail shards, and `detail/system.runtime.yaml`
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
- `examples/aircraft_pitch_digital.py`
  - sampled state-feedback controller for a continuous aircraft pitch plant
- `examples/inverted_pendulum_lqr.py`
  - hierarchical digital LQR for the classic cart-pendulum benchmark
- `examples/cstr_temperature_control.py`
  - nonlinear CSTR temperature regulation with a sampled PI controller
- `examples/quadruple_tank.py`
  - decentralized multivariable level control for the quadruple-tank benchmark
- `examples/large_hierarchical_benchmark.py`
  - generated deep subsystem hierarchy for compile and runtime scaling checks
- `examples/multirate_offset_priority.py`
  - sampled tasks with explicit `offset` and `priority`
  - useful for understanding same-time execution order and cross-rate summaries

Common modifications to these examples:

- Insert a noise source between plant output and measurement input
- Replace the controller block with a different sampled controller
- Add an observer that records selected signals
- Change `sample_time`, `offset`, and `dt` together, then re-run validation

## Notes

- Continuous states must be numeric and SciPy-compatible.
- Discrete states can be any Python object.
- `SignalSpec.dtype` currently supports `bool`, `int`, `float`, `complex`, and `object`.
- `SignalSpec.shape` currently supports scalars `()`, vectors `(n,)`, and matrices `(m, n)`.
- Port signal checks run statically when declarations are available and once at `t=start` as a runtime safeguard.
- `(stop - start)` must be an integer multiple of `dt`.
- Each `DiscreteBlock.sample_time` must be an integer multiple of `dt`.
- Each `DiscreteBlock.offset` must align with the simulation `dt` grid.
- Lower discrete `priority` values run first when multiple rate groups hit at the same time.
