from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ._hierarchy import flatten_system
from ._model import build_model_summary
from .compiler import (
    ExecutionPlan,
    ResolvedRateGroup,
    _analyze_system,
    _build_execution_plan,
    _error_diagnostics,
    compile_system,
)
from .core import Block, ContinuousBlock, DiscreteBlock, ExecutionContext, SignalSpec
from .diagnostics import Diagnostic, ValidationReport
from .errors import ModelValidationError, SimulationError
from .solver import FloatVector, SciPySolver, Solver


class _UnresolvedInput:
    def __repr__(self) -> str:
        return "<UNRESOLVED_INPUT>"


UNRESOLVED_INPUT = _UnresolvedInput()


def _signal_dtype_from_numpy_kind(kind: str) -> str:
    if kind == "b":
        return "bool"
    if kind in {"i", "u"}:
        return "int"
    if kind == "f":
        return "float"
    if kind == "c":
        return "complex"
    return "object"


def _infer_signal_value_signature(value: Any) -> tuple[str, tuple[int, ...]]:
    if isinstance(value, bool):
        return ("bool", ())
    if isinstance(value, int) and not isinstance(value, bool):
        return ("int", ())
    if isinstance(value, float):
        return ("float", ())
    if isinstance(value, complex):
        return ("complex", ())

    array = np.asarray(value)
    return (
        _signal_dtype_from_numpy_kind(array.dtype.kind),
        tuple(int(dimension) for dimension in array.shape),
    )


def _format_signal_shape(shape: tuple[int, ...]) -> str:
    if shape == ():
        return "()"
    return str(shape)


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
                    f"Continuous state for block {block_name!r} must be non-empty.",
                    code="EMPTY_CONTINUOUS_STATE",
                    suggestion="Return a non-empty numeric state from initial_continuous_state().",
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
                    f"Missing continuous state for block {segment.block_name!r}.",
                    code="MISSING_CONTINUOUS_STATE",
                    suggestion="Initialize every continuous block before packing the solver state vector.",
                ) from exc
            array = np.asarray(value, dtype=float)
            if array.size != segment.size or array.shape != segment.shape:
                raise ModelValidationError(
                    f"Continuous state shape mismatch for block {segment.block_name!r}: "
                    f"expected {segment.shape}, got {array.shape}.",
                    code="CONTINUOUS_STATE_SHAPE_MISMATCH",
                    suggestion="Keep the continuous state shape consistent across initial_continuous_state(), derivative(), and solver steps.",
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

    def validate(
        self,
        system,
        config: SimulationConfig | None = None,
    ) -> ValidationReport:
        flatten_result = flatten_system(system)
        diagnostics = list(flatten_result.diagnostics)

        if flatten_result.flat_system is None:
            empty_system = type(system)(system.name)
            analysis = _analyze_system(empty_system)
            summary = build_model_summary(
                analysis.model,
                block_order=None,
                config=config,
                hierarchy=flatten_result.hierarchy_summary,
                rate_groups=[],
                cross_rate_connections=[],
                execution_notes={},
            )
        else:
            analysis = _analyze_system(flatten_result.flat_system)
            diagnostics.extend(analysis.diagnostics)
            summary = build_model_summary(
                analysis.model,
                block_order=analysis.block_order,
                config=config,
                hierarchy=flatten_result.hierarchy_summary,
                rate_groups=[group.summary() for group in analysis.rate_groups],
                cross_rate_connections=[item.summary() for item in analysis.cross_rate_connections],
                execution_notes=dict(analysis.execution_notes),
            )

            flatten_errors = _error_diagnostics(flatten_result.diagnostics)
            analysis_errors = _error_diagnostics(analysis.diagnostics)

            if analysis.block_order is not None and not analysis_errors:
                plan = _build_execution_plan(
                    flatten_result.flat_system,
                    analysis,
                    hierarchy_summary=flatten_result.hierarchy_summary,
                )

                if config is not None:
                    diagnostics.extend(self._collect_time_grid_diagnostics(plan, config))

                discrete_states, discrete_state_diagnostics = self._collect_discrete_state_diagnostics(plan)
                continuous_states, continuous_state_diagnostics = self._collect_continuous_state_diagnostics(
                    plan
                )
                diagnostics.extend(discrete_state_diagnostics)
                diagnostics.extend(continuous_state_diagnostics)

                if (
                    not flatten_errors
                    and not analysis_errors
                    and not discrete_state_diagnostics
                    and not continuous_state_diagnostics
                ):
                    diagnostics.extend(
                        self._collect_initial_signal_diagnostics(
                            plan,
                            config=config,
                            discrete_states=discrete_states,
                            continuous_states=continuous_states,
                        )
                    )

        diagnostics.sort(key=lambda diagnostic: (not diagnostic.is_error,))

        return ValidationReport(
            system_name=system.name,
            diagnostics=tuple(diagnostics),
            _summary_data=summary,
        )

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
        snapshots: list[StepSnapshot] = []
        current_time = float(config.start)

        try:
            for step_index in range(step_count + 1):
                discrete_states, current_outputs, _ = self._evaluate_visible_outputs_at_time(
                    plan=plan,
                    time=current_time,
                    step_index=step_index,
                    dt=config.dt,
                    discrete_states=discrete_states,
                    continuous_states=continuous_states,
                    validate_signal_values=step_index == 0,
                )
                snapshots.append(
                    self._make_snapshot(
                        time=current_time,
                        step_index=step_index,
                        outputs=current_outputs,
                        discrete_states=discrete_states,
                        continuous_states=continuous_states,
                    )
                )
                self._notify(observer, "on_step", snapshots[-1])
                if step_index == step_count:
                    continue

                continuous_states = self._advance_continuous_states(
                    plan=plan,
                    config=config,
                    step_index=step_index,
                    time=current_time,
                    discrete_states=discrete_states,
                    continuous_states=continuous_states,
                    codec=codec,
                )
                current_time = round(config.start + (step_index + 1) * config.dt, 12)
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
        diagnostics = self._collect_time_grid_diagnostics(plan, config)
        if diagnostics:
            raise ModelValidationError.from_diagnostic(diagnostics[0])

    def _collect_time_grid_diagnostics(
        self,
        plan: ExecutionPlan,
        config: SimulationConfig,
    ) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        total = config.stop - config.start
        step_count = total / config.dt
        rounded = round(step_count)
        if not np.isclose(step_count, rounded, atol=1e-9, rtol=0.0):
            diagnostics.append(
                Diagnostic(
                    code="INVALID_TIME_GRID",
                    message="Simulation horizon must be an integer multiple of dt.",
                    suggestion="Choose start, stop, and dt so that (stop - start) / dt is an integer.",
                )
            )

        for block_name in plan.discrete_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, DiscreteBlock)
            ratio = block.sample_time / config.dt
            if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0.0):
                diagnostics.append(
                    Diagnostic(
                        code="INCOMPATIBLE_SAMPLE_TIME",
                        message=(
                            f"Discrete block {block_name!r} sample_time={block.sample_time} "
                            f"is incompatible with dt={config.dt}."
                        ),
                        suggestion="Choose a sample_time that is an integer multiple of dt.",
                        block_name=block_name,
                    )
                )
            offset_ratio = block.offset / config.dt
            if not np.isclose(offset_ratio, round(offset_ratio), atol=1e-9, rtol=0.0):
                diagnostics.append(
                    Diagnostic(
                        code="INCOMPATIBLE_SAMPLE_OFFSET",
                        message=(
                            f"Discrete block {block_name!r} offset={block.offset} "
                            f"is incompatible with dt={config.dt}."
                        ),
                        suggestion="Choose an offset that is an integer multiple of dt.",
                        block_name=block_name,
                    )
                )
            start_alignment = (config.start - block.offset) / config.dt
            if not np.isclose(start_alignment, round(start_alignment), atol=1e-9, rtol=0.0):
                diagnostics.append(
                    Diagnostic(
                        code="INCOMPATIBLE_SAMPLE_OFFSET",
                        message=(
                            f"Discrete block {block_name!r} offset={block.offset} does not align "
                            f"with start={config.start} on the dt={config.dt} simulation grid."
                        ),
                        suggestion="Choose start, dt, and offset so sample hits land on simulation time points.",
                        block_name=block_name,
                    )
                )
        return diagnostics

    def _compute_step_count(self, config: SimulationConfig) -> int:
        return int(round((config.stop - config.start) / config.dt))

    def _collect_discrete_state_diagnostics(
        self,
        plan: ExecutionPlan,
    ) -> tuple[dict[str, Any], list[Diagnostic]]:
        states: dict[str, Any] = {}
        diagnostics: list[Diagnostic] = []
        for block_name in plan.discrete_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, DiscreteBlock)
            try:
                states[block_name] = block.initial_discrete_state()
            except Exception as exc:
                diagnostics.append(
                    Diagnostic(
                        code="DISCRETE_INITIAL_STATE_ERROR",
                        message=(
                            f"Discrete block {block_name!r} failed to produce an initial discrete state: {exc!r}."
                        ),
                        suggestion="Make initial_discrete_state() return the starting sampled state without raising.",
                        block_name=block_name,
                    )
                )
        return states, diagnostics

    def _initialize_discrete_states(self, plan: ExecutionPlan) -> dict[str, Any]:
        states, diagnostics = self._collect_discrete_state_diagnostics(plan)
        if diagnostics:
            raise ModelValidationError.from_diagnostic(diagnostics[0])
        return states

    def _collect_continuous_state_diagnostics(
        self,
        plan: ExecutionPlan,
    ) -> tuple[dict[str, Any], list[Diagnostic]]:
        states: dict[str, Any] = {}
        diagnostics: list[Diagnostic] = []
        for block_name in plan.continuous_blocks:
            block = plan.system.blocks[block_name]
            assert isinstance(block, ContinuousBlock)
            try:
                initial_state = block.initial_continuous_state()
            except Exception as exc:
                diagnostics.append(
                    Diagnostic(
                        code="CONTINUOUS_INITIAL_STATE_ERROR",
                        message=(
                            f"Continuous block {block_name!r} failed to produce an initial continuous state: {exc!r}."
                        ),
                        suggestion="Make initial_continuous_state() return a numeric SciPy-compatible state without raising.",
                        block_name=block_name,
                    )
                )
                continue

            if initial_state is None:
                diagnostics.append(
                    Diagnostic(
                        code="MISSING_CONTINUOUS_INITIAL_STATE",
                        message=f"Continuous block {block_name!r} must provide an initial state.",
                        suggestion="Return a numeric initial state from initial_continuous_state().",
                        block_name=block_name,
                    )
                )
                continue

            states[block_name] = initial_state
        return states, diagnostics

    def _initialize_continuous_states(self, plan: ExecutionPlan) -> dict[str, Any]:
        states, diagnostics = self._collect_continuous_state_diagnostics(plan)
        if diagnostics:
            raise ModelValidationError.from_diagnostic(diagnostics[0])
        return states

    def _collect_initial_signal_diagnostics(
        self,
        plan: ExecutionPlan,
        *,
        config: SimulationConfig | None,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
    ) -> list[Diagnostic]:
        time = config.start if config is not None else 0.0
        dt = config.dt if config is not None else 1.0
        try:
            self._evaluate_visible_outputs_at_time(
                plan=plan,
                time=time,
                step_index=0,
                dt=dt,
                discrete_states=discrete_states,
                continuous_states=continuous_states,
                validate_signal_values=True,
            )
        except SimulationError as error:
            return [self._diagnostic_from_simulation_error(error)]
        return []

    def _evaluate_visible_outputs_at_time(
        self,
        plan: ExecutionPlan,
        *,
        time: float,
        step_index: int,
        dt: float,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
        validate_signal_values: bool = False,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        visible_discrete_states = dict(discrete_states)
        current_outputs = self._evaluate_outputs(
            plan,
            time=time,
            step_index=step_index,
            dt=dt,
            discrete_states=visible_discrete_states,
            continuous_states=continuous_states,
            validate_signal_values=False,
        )
        for group in self._build_hit_schedule_at_time(plan=plan, time=time):
            visible_discrete_states, current_outputs = self._apply_discrete_task_group(
                plan=plan,
                group=group,
                time=time,
                step_index=step_index,
                dt=dt,
                discrete_states=visible_discrete_states,
                continuous_states=continuous_states,
                current_outputs=current_outputs,
            )
        if validate_signal_values:
            current_outputs = self._evaluate_outputs(
                plan,
                time=time,
                step_index=step_index,
                dt=dt,
                discrete_states=visible_discrete_states,
                continuous_states=continuous_states,
                validate_signal_values=True,
            )
        full_inputs = self._resolve_all_inputs(plan, current_outputs, allow_unresolved=False)
        if validate_signal_values:
            self._validate_input_signal_values(
                plan,
                time=time,
                full_inputs=full_inputs,
            )
        return visible_discrete_states, current_outputs, full_inputs

    def _evaluate_outputs(
        self,
        plan: ExecutionPlan,
        *,
        time: float,
        step_index: int,
        dt: float,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
        validate_signal_values: bool = False,
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
                    code="BLOCK_OUTPUT_EXCEPTION",
                    suggestion="Update output() so it only uses declared inputs, state, and parameters.",
                ) from exc
            outputs[block_name] = self._normalize_outputs(
                block=block,
                block_name=block_name,
                time=time,
                raw_output=raw_output,
            )
            if validate_signal_values:
                self._validate_output_signal_values(
                    block=block,
                    block_name=block_name,
                    time=time,
                    outputs=outputs[block_name],
                )
        return outputs

    def _validate_output_signal_values(
        self,
        *,
        block: Block,
        block_name: str,
        time: float,
        outputs: Mapping[str, Any],
    ) -> None:
        for spec in block.output_ports:
            if not spec.signal_spec.is_specified:
                continue
            self._validate_signal_value(
                signal_spec=spec.signal_spec,
                value=outputs[spec.name],
                block_name=block_name,
                port_name=spec.name,
                time=time,
                code_for_dtype="OUTPUT_TYPE_MISMATCH",
                code_for_shape="OUTPUT_SHAPE_MISMATCH",
                message_prefix="Output",
                suggestion=(
                    "Return values that match the declared output SignalSpec or update the output port declaration."
                ),
            )

    def _validate_input_signal_values(
        self,
        plan: ExecutionPlan,
        *,
        time: float,
        full_inputs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for block_name, block in plan.system.blocks.items():
            bindings = plan.input_bindings[block_name]
            for spec in block.input_ports:
                if not spec.signal_spec.is_specified:
                    continue
                endpoint = bindings.get(spec.name)
                if endpoint is None:
                    continue
                self._validate_signal_value(
                    signal_spec=spec.signal_spec,
                    value=full_inputs[block_name][spec.name],
                    block_name=block_name,
                    port_name=spec.name,
                    time=time,
                    connection=f"{endpoint} -> {block_name}.{spec.name}",
                    code_for_dtype="INPUT_TYPE_MISMATCH",
                    code_for_shape="INPUT_SHAPE_MISMATCH",
                    message_prefix="Input",
                    suggestion=(
                        "Update the connected source value or the input SignalSpec so the runtime signal matches the declared dtype and shape."
                    ),
                )

    def _validate_signal_value(
        self,
        *,
        signal_spec: SignalSpec,
        value: Any,
        block_name: str,
        port_name: str,
        time: float,
        code_for_dtype: str,
        code_for_shape: str,
        message_prefix: str,
        suggestion: str,
        connection: str | None = None,
    ) -> None:
        if not signal_spec.is_specified:
            return

        actual_dtype, actual_shape = _infer_signal_value_signature(value)
        endpoint = f"{block_name}.{port_name}"

        if signal_spec.dtype is not None and actual_dtype != signal_spec.dtype:
            raise SimulationError(
                (
                    f"{message_prefix} {endpoint} resolved dtype {actual_dtype!r} "
                    f"but declared dtype is {signal_spec.dtype!r}."
                ),
                block_name=block_name,
                port_name=port_name,
                connection=connection,
                time=time,
                code=code_for_dtype,
                suggestion=suggestion,
            )

        if signal_spec.shape is not None and actual_shape != signal_spec.shape:
            raise SimulationError(
                (
                    f"{message_prefix} {endpoint} resolved shape {_format_signal_shape(actual_shape)} "
                    f"but declared shape is {_format_signal_shape(signal_spec.shape)}."
                ),
                block_name=block_name,
                port_name=port_name,
                connection=connection,
                time=time,
                code=code_for_shape,
                suggestion=suggestion,
            )

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
                    code="UNRESOLVED_CURRENT_INPUT",
                    suggestion="Ensure the execution order is valid and that feedback paths pass through stateful blocks.",
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
                        code="UNRESOLVED_CONNECTED_INPUT",
                        suggestion="Check the connected source port and make sure the model can be evaluated without an algebraic loop.",
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
                code="INVALID_EMPTY_OUTPUT_RETURN",
                suggestion="Return None or {} from blocks that declare no outputs.",
            )

        if len(output_specs) == 1 and not isinstance(raw_output, Mapping):
            return {output_specs[0].name: raw_output}

        if not isinstance(raw_output, Mapping):
            raise SimulationError(
                "Blocks with multiple outputs must return a mapping.",
                block_name=block_name,
                time=time,
                code="INVALID_MULTI_OUTPUT_RETURN",
                suggestion="Return a mapping that includes every declared output name.",
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
                code="MISSING_DECLARED_OUTPUTS",
                suggestion="Return every declared output port from output().",
            )
        if extra:
            raise SimulationError(
                f"Block returned undeclared outputs: {extra}.",
                block_name=block_name,
                time=time,
                code="UNDECLARED_OUTPUTS",
                suggestion="Only return output names that were declared on the block.",
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
                        code="CONTINUOUS_DERIVATIVE_EXCEPTION",
                        suggestion="Make derivative() pure, side-effect free, and compatible with the continuous state shape.",
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
                code="CONTINUOUS_SOLVER_FAILURE",
                suggestion="Check the derivative implementation and choose a solver configuration that matches the model dynamics.",
            ) from exc
        return codec.unpack(next_vector)

    def _build_hit_schedule_at_time(
        self,
        *,
        plan: ExecutionPlan,
        time: float,
    ) -> tuple[ResolvedRateGroup, ...]:
        return tuple(
            group
            for group in plan.rate_groups
            if self._is_sample_hit(time=time, sample_time=group.sample_time, offset=group.offset)
        )

    def _apply_discrete_task_group(
        self,
        *,
        plan: ExecutionPlan,
        group: ResolvedRateGroup,
        time: float,
        step_index: int,
        dt: float,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
        current_outputs: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        next_states = dict(discrete_states)
        full_inputs = self._resolve_all_inputs(plan, current_outputs, allow_unresolved=False)
        pending_updates: dict[str, Any] = {}
        for block_name in group.block_names:
            block = plan.system.blocks[block_name]
            assert isinstance(block, DiscreteBlock)
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
                pending_updates[block_name] = block.update_state(
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
                    code="DISCRETE_UPDATE_EXCEPTION",
                    suggestion="Make update_state() return the next sampled state without mutating shared runtime state.",
                ) from exc
        next_states.update(pending_updates)
        next_outputs = self._propagate_after_discrete_commit(
            plan=plan,
            time=time,
            step_index=step_index,
            dt=dt,
            discrete_states=next_states,
            continuous_states=continuous_states,
        )
        return next_states, next_outputs

    def _propagate_after_discrete_commit(
        self,
        *,
        plan: ExecutionPlan,
        time: float,
        step_index: int,
        dt: float,
        discrete_states: Mapping[str, Any],
        continuous_states: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        return self._evaluate_outputs(
            plan,
            time=time,
            step_index=step_index,
            dt=dt,
            discrete_states=discrete_states,
            continuous_states=continuous_states,
            validate_signal_values=False,
        )

    def _is_sample_hit(self, *, time: float, sample_time: float, offset: float) -> bool:
        if time + 1e-9 < offset:
            return False
        normalized = (time - offset) / sample_time
        return bool(np.isclose(normalized, round(normalized), atol=1e-9, rtol=0.0))

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

    def _diagnostic_from_simulation_error(self, error: SimulationError) -> Diagnostic:
        return Diagnostic(
            code=error.code or "SIMULATION_ERROR",
            message=getattr(error, "base_message", error.message),
            suggestion=error.suggestion or "Inspect the failing block callback and the connected signals.",
            block_name=error.block_name,
            port_name=error.port_name,
            connection=error.connection,
            time=error.time,
        )

    def _notify(self, observer: SimulationObserver | None, method_name: str, *args: Any) -> None:
        if observer is None:
            return
        method = getattr(observer, method_name, None)
        if method is not None:
            method(*args)
