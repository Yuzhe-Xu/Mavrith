from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from .compiler import ExecutionPlan, compile_system
from .core import Block, ContinuousBlock, DiscreteBlock, ExecutionContext
from .errors import ModelValidationError, SimulationError
from .solver import FloatVector, SciPySolver, Solver


class _UnresolvedInput:
    def __repr__(self) -> str:
        return "<UNRESOLVED_INPUT>"


UNRESOLVED_INPUT = _UnresolvedInput()


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    start: float = 0.0
    stop: float = 1.0
    dt: float = 0.1
    solver: Solver = field(default_factory=SciPySolver)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start.")


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    time: float
    step_index: int
    outputs: Mapping[str, Mapping[str, Any]]
    discrete_states: Mapping[str, Any]
    continuous_states: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    time_points: tuple[float, ...]
    final_outputs: Mapping[str, Mapping[str, Any]]
    final_discrete_states: Mapping[str, Any]
    final_continuous_states: Mapping[str, Any]


class SimulationObserver(Protocol):
    def on_simulation_start(self, plan: ExecutionPlan, config: SimulationConfig) -> None:
        ...

    def on_step(self, snapshot: StepSnapshot) -> None:
        ...

    def on_simulation_error(self, error: SimulationError) -> None:
        ...

    def on_simulation_end(self, result: SimulationResult) -> None:
        ...


@dataclass(frozen=True, slots=True)
class _StateSegment:
    block_name: str
    shape: tuple[int, ...]
    size: int


class _ContinuousStateCodec:
    def __init__(self, ordered_states: Mapping[str, Any]) -> None:
        segments: list[_StateSegment] = []
        total_size = 0
        for block_name, state in ordered_states.items():
            array = np.asarray(state, dtype=float)
            size = int(array.size)
            if size <= 0:
                raise ModelValidationError(
                    f"Continuous state for block {block_name!r} must be non-empty."
                )
            segments.append(_StateSegment(block_name=block_name, shape=array.shape, size=size))
            total_size += size
        self._segments = tuple(segments)
        self._total_size = total_size

    def pack(self, states: Mapping[str, Any]) -> FloatVector:
        vector = np.empty(self._total_size, dtype=float)
        cursor = 0
        for segment in self._segments:
            try:
                value = states[segment.block_name]
            except KeyError as exc:
                raise ModelValidationError(
                    f"Missing continuous state for block {segment.block_name!r}."
                ) from exc
            array = np.asarray(value, dtype=float)
            if array.size != segment.size or array.shape != segment.shape:
                raise ModelValidationError(
                    f"Continuous state shape mismatch for block {segment.block_name!r}: "
                    f"expected {segment.shape}, got {array.shape}."
                )
            vector[cursor : cursor + segment.size] = array.reshape(-1)
            cursor += segment.size
        return vector

    def unpack(self, vector: FloatVector) -> dict[str, Any]:
        state_map: dict[str, Any] = {}
        cursor = 0
        for segment in self._segments:
            raw = np.asarray(vector[cursor : cursor + segment.size], dtype=float)
            reshaped = raw.reshape(segment.shape)
            if segment.shape == ():
                state_map[segment.block_name] = float(reshaped)
            else:
                state_map[segment.block_name] = reshaped.copy()
            cursor += segment.size
        return state_map


class Simulator:
    def compile(self, system) -> ExecutionPlan:
        return compile_system(system)

    def run(
        self,
        system,
        config: SimulationConfig,
        observer: SimulationObserver | None = None,
    ) -> SimulationResult:
        plan = self.compile(system)
        self._validate_time_grid(plan, config)

        discrete_states = self._initialize_discrete_states(plan)
        continuous_states = self._initialize_continuous_states(plan)
        codec = (
            _ContinuousStateCodec({name: continuous_states[name] for name in plan.continuous_blocks})
            if plan.continuous_blocks
            else None
        )

        self._notify(observer, "on_simulation_start", plan, config)

        step_count = self._compute_step_count(config)
        current_time = float(config.start)
        current_outputs = self._evaluate_outputs(
            plan,
            time=current_time,
            step_index=0,
            dt=config.dt,
            discrete_states=discrete_states,
            continuous_states=continuous_states,
        )

        snapshots: list[StepSnapshot] = []
        snapshots.append(
            self._make_snapshot(
                time=current_time,
                step_index=0,
                outputs=current_outputs,
                discrete_states=discrete_states,
                continuous_states=continuous_states,
            )
        )
        self._notify(observer, "on_step", snapshots[-1])

        try:
            for step_index in range(step_count):
                full_inputs = self._resolve_all_inputs(plan, current_outputs, allow_unresolved=False)

                next_continuous_states = self._advance_continuous_states(
                    plan=plan,
                    config=config,
                    step_index=step_index,
                    time=current_time,
                    discrete_states=discrete_states,
                    continuous_states=continuous_states,
                    codec=codec,
                )
                next_discrete_states = self._advance_discrete_states(
                    plan=plan,
                    config=config,
                    step_index=step_index,
                    time=current_time,
                    discrete_states=discrete_states,
                    full_inputs=full_inputs,
                    continuous_states=continuous_states,
                )

                current_time = round(config.start + (step_index + 1) * config.dt, 12)
                continuous_states = next_continuous_states
                discrete_states = next_discrete_states
                current_outputs = self._evaluate_outputs(
                    plan,
                    time=current_time,
                    step_index=step_index + 1,
                    dt=config.dt,
                    discrete_states=discrete_states,
                    continuous_states=continuous_states,
                )
                snapshots.append(
                    self._make_snapshot(
                        time=current_time,
                        step_index=step_index + 1,
                        outputs=current_outputs,
                        discrete_states=discrete_states,
                        continuous_states=continuous_states,
                    )
                )
                self._notify(observer, "on_step", snapshots[-1])
        except SimulationError as error:
            self._notify(observer, "on_simulation_error", error)
            raise

        result = SimulationResult(
            time_points=tuple(snapshot.time for snapshot in snapshots),
            final_outputs=self._copy_output_mapping(current_outputs),
            final_discrete_states=dict(discrete_states),
            final_continuous_states=dict(continuous_states),
        )
        self._notify(observer, "on_simulation_end", result)
        return result

    def _validate_time_grid(self, plan: ExecutionPlan, config: SimulationConfig) -> None:
        total = config.stop - config.start
        step_count = total / config.dt
        rounded = round(step_count)
        if not np.isclose(step_count, rounded, atol=1e-9, rtol=0.0):
            raise ModelValidationError(
                "Simulation horizon must be an integer multiple of dt."
            )

        for block_name in plan.discrete_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, DiscreteBlock)
            ratio = block.sample_time / config.dt
            if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0.0):
                raise ModelValidationError(
                    f"Discrete block {block_name!r} sample_time={block.sample_time} "
                    f"is incompatible with dt={config.dt}."
                )

    def _compute_step_count(self, config: SimulationConfig) -> int:
        return int(round((config.stop - config.start) / config.dt))

    def _initialize_discrete_states(self, plan: ExecutionPlan) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for block_name in plan.discrete_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, DiscreteBlock)
            states[block_name] = block.initial_discrete_state()
        return states

    def _initialize_continuous_states(self, plan: ExecutionPlan) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for block_name in plan.continuous_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, ContinuousBlock)
            initial_state = block.initial_continuous_state()
            if initial_state is None:
                raise ModelValidationError(
                    f"Continuous block {block_name!r} must provide an initial state."
                )
            states[block_name] = initial_state
        return states

    def _evaluate_outputs(
        self,
        plan: ExecutionPlan,
        *,
        time: float,
        step_index: int,
        dt: float,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        for block_name in plan.block_order:
            block = plan.system.blocks[block_name]
            block_inputs = self._resolve_inputs_for_output(
                plan,
                block_name=block_name,
                outputs=outputs,
                allow_unresolved=not block.direct_feedthrough,
            )
            ctx = ExecutionContext(
                block_name=block_name,
                time=time,
                step_index=step_index,
                dt=dt,
                parameters=block.parameters,
                discrete_state=discrete_states.get(block_name),
                continuous_state=continuous_states.get(block_name),
            )
            try:
                raw_output = block.output(ctx, block_inputs)
            except SimulationError:
                raise
            except Exception as exc:
                raise SimulationError.from_exception(
                    "Block output evaluation failed.",
                    block_name=block_name,
                    time=time,
                    cause=exc,
                ) from exc
            outputs[block_name] = self._normalize_outputs(
                block=block,
                block_name=block_name,
                time=time,
                raw_output=raw_output,
            )
        return outputs

    def _resolve_inputs_for_output(
        self,
        plan: ExecutionPlan,
        *,
        block_name: str,
        outputs: Mapping[str, Mapping[str, Any]],
        allow_unresolved: bool,
    ) -> dict[str, Any]:
        block = plan.system.blocks[block_name]
        bindings = plan.input_bindings[block_name]
        values: dict[str, Any] = {}
        for spec in block.input_ports:
            endpoint = bindings.get(spec.name)
            if endpoint is None:
                values[spec.name] = None
                continue
            source_outputs = outputs.get(endpoint.block_name)
            if source_outputs is None or endpoint.port_name not in source_outputs:
                if allow_unresolved:
                    values[spec.name] = UNRESOLVED_INPUT
                    continue
                raise SimulationError(
                    "Required current-step input was unresolved during output evaluation.",
                    block_name=block_name,
                    port_name=spec.name,
                    connection=f"{endpoint} -> {block_name}.{spec.name}",
                )
            values[spec.name] = source_outputs[endpoint.port_name]
        return values

    def _resolve_all_inputs(
        self,
        plan: ExecutionPlan,
        outputs: Mapping[str, Mapping[str, Any]],
        *,
        allow_unresolved: bool,
    ) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for block_name in plan.system.blocks:
            bindings = plan.input_bindings[block_name]
            block = plan.system.blocks[block_name]
            values: dict[str, Any] = {}
            for spec in block.input_ports:
                endpoint = bindings.get(spec.name)
                if endpoint is None:
                    values[spec.name] = None
                    continue
                source_outputs = outputs.get(endpoint.block_name)
                if source_outputs is None or endpoint.port_name not in source_outputs:
                    if allow_unresolved:
                        values[spec.name] = UNRESOLVED_INPUT
                        continue
                    raise SimulationError(
                        "Connected input could not be resolved.",
                        block_name=block_name,
                        port_name=spec.name,
                        connection=f"{endpoint} -> {block_name}.{spec.name}",
                    )
                values[spec.name] = source_outputs[endpoint.port_name]
            resolved[block_name] = values
        return resolved

    def _normalize_outputs(
        self,
        *,
        block: Block,
        block_name: str,
        time: float,
        raw_output: Any,
    ) -> dict[str, Any]:
        output_specs = block.output_ports
        if not output_specs:
            if raw_output in (None, {}):
                return {}
            raise SimulationError(
                "Blocks without declared outputs must return None or {}.",
                block_name=block_name,
                time=time,
            )

        if len(output_specs) == 1 and not isinstance(raw_output, Mapping):
            return {output_specs[0].name: raw_output}

        if not isinstance(raw_output, Mapping):
            raise SimulationError(
                "Blocks with multiple outputs must return a mapping.",
                block_name=block_name,
                time=time,
            )

        normalized: dict[str, Any] = {}
        declared_names = {spec.name for spec in output_specs}
        missing = [name for name in declared_names if name not in raw_output]
        extra = [name for name in raw_output if name not in declared_names]
        if missing:
            raise SimulationError(
                f"Block is missing declared outputs: {missing}.",
                block_name=block_name,
                time=time,
            )
        if extra:
            raise SimulationError(
                f"Block returned undeclared outputs: {extra}.",
                block_name=block_name,
                time=time,
            )
        for spec in output_specs:
            normalized[spec.name] = raw_output[spec.name]
        return normalized

    def _advance_continuous_states(
        self,
        *,
        plan: ExecutionPlan,
        config: SimulationConfig,
        step_index: int,
        time: float,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
        codec: _ContinuousStateCodec | None,
    ) -> dict[str, Any]:
        if not plan.continuous_blocks or codec is None:
            return dict(continuous_states)

        def derivative_callback(candidate_time: float, state_vector: FloatVector) -> FloatVector:
            candidate_continuous_states = codec.unpack(state_vector)
            candidate_outputs = self._evaluate_outputs(
                plan,
                time=candidate_time,
                step_index=step_index,
                dt=config.dt,
                discrete_states=discrete_states,
                continuous_states=candidate_continuous_states,
            )
            candidate_inputs = self._resolve_all_inputs(
                plan,
                candidate_outputs,
                allow_unresolved=False,
            )
            derivatives: dict[str, Any] = {}
            for block_name in plan.continuous_blocks:
                block = plan.system.blocks[block_name]
                assert isinstance(block, ContinuousBlock)
                ctx = ExecutionContext(
                    block_name=block_name,
                    time=candidate_time,
                    step_index=step_index,
                    dt=config.dt,
                    parameters=block.parameters,
                    discrete_state=discrete_states.get(block_name),
                    continuous_state=candidate_continuous_states[block_name],
                )
                try:
                    derivatives[block_name] = block.derivative(
                        ctx,
                        candidate_inputs[block_name],
                        candidate_continuous_states[block_name],
                    )
                except SimulationError:
                    raise
                except Exception as exc:
                    raise SimulationError.from_exception(
                        "Continuous derivative evaluation failed.",
                        block_name=block_name,
                        time=candidate_time,
                        cause=exc,
                    ) from exc
            return codec.pack(derivatives)

        try:
            current_vector = codec.pack(
                {name: continuous_states[name] for name in plan.continuous_blocks}
            )
            next_vector = config.solver.step(
                t=time,
                dt=config.dt,
                state_vector=current_vector,
                derivative=derivative_callback,
            )
        except SimulationError:
            raise
        except Exception as exc:
            raise SimulationError.from_exception(
                "Continuous solver step failed.",
                time=time,
                cause=exc,
            ) from exc
        return codec.unpack(next_vector)

    def _advance_discrete_states(
        self,
        *,
        plan: ExecutionPlan,
        config: SimulationConfig,
        step_index: int,
        time: float,
        discrete_states: Mapping[str, Any],
        full_inputs: Mapping[str, Mapping[str, Any]],
        continuous_states: Mapping[str, Any],
    ) -> dict[str, Any]:
        next_states = dict(discrete_states)
        for block_name in plan.discrete_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, DiscreteBlock)
            if not self._is_sample_hit(
                time=time,
                start=config.start,
                sample_time=block.sample_time,
            ):
                continue
            ctx = ExecutionContext(
                block_name=block_name,
                time=time,
                step_index=step_index,
                dt=config.dt,
                parameters=block.parameters,
                discrete_state=discrete_states.get(block_name),
                continuous_state=continuous_states.get(block_name),
            )
            try:
                next_states[block_name] = block.update_state(
                    ctx,
                    full_inputs[block_name],
                    discrete_states.get(block_name),
                )
            except SimulationError:
                raise
            except Exception as exc:
                raise SimulationError.from_exception(
                    "Discrete state update failed.",
                    block_name=block_name,
                    time=time,
                    cause=exc,
                ) from exc
        return next_states

    def _is_sample_hit(self, *, time: float, start: float, sample_time: float) -> bool:
        offset = (time - start) / sample_time
        return bool(np.isclose(offset, round(offset), atol=1e-9, rtol=0.0))

    def _make_snapshot(
        self,
        *,
        time: float,
        step_index: int,
        outputs: Mapping[str, Mapping[str, Any]],
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
    ) -> StepSnapshot:
        return StepSnapshot(
            time=time,
            step_index=step_index,
            outputs=self._copy_output_mapping(outputs),
            discrete_states=dict(discrete_states),
            continuous_states=dict(continuous_states),
        )

    def _copy_output_mapping(
        self,
        outputs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return {block_name: dict(values) for block_name, values in outputs.items()}

    def _notify(self, observer: SimulationObserver | None, method_name: str, *args: Any) -> None:
        if observer is None:
            return
        method = getattr(observer, method_name, None)
        if method is not None:
            method(*args)
