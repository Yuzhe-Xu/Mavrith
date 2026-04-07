from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


FloatVector = NDArray[np.float64]
DerivativeFunction = Callable[[float, FloatVector], FloatVector]


class Solver(Protocol):
    def step(
        self,
        *,
        t: float,
        dt: float,
        state_vector: FloatVector,
        derivative: DerivativeFunction,
    ) -> FloatVector:
        ...


@dataclass(slots=True)
class SciPySolver:
    method: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-9

    def step(
        self,
        *,
        t: float,
        dt: float,
        state_vector: FloatVector,
        derivative: DerivativeFunction,
    ) -> FloatVector:
        if dt < 0:
            raise ValueError("dt must be non-negative.")
        if dt == 0 or state_vector.size == 0:
            return np.array(state_vector, dtype=float, copy=True)

        solution = solve_ivp(
            lambda tau, values: derivative(tau, np.asarray(values, dtype=float)),
            (t, t + dt),
            np.asarray(state_vector, dtype=float),
            method=self.method,
            t_eval=[t + dt],
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            raise RuntimeError(f"SciPy solver failed: {solution.message}")
        return np.asarray(solution.y[:, -1], dtype=float)
