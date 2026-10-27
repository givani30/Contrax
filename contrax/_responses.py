"""Standard single-input-channel LTI responses."""

from typing import Any

import jax.numpy as jnp
from jax import Array

from contrax.core import ContLTI, DiscLTI
from contrax.types import SimResult


def _unit_input(sys: DiscLTI | ContLTI, input_index: int) -> Array:
    m = sys.B.shape[1]
    if not 0 <= input_index < m:
        raise ValueError(f"input_index must be in [0, {m}), got {input_index}.")
    return jnp.zeros(m, dtype=sys.B.dtype).at[input_index].set(1.0)


def step_response(
    sys: DiscLTI | ContLTI,
    *,
    num_steps: int | None = None,
    duration: float | None = None,
    input_index: int = 0,
    x0: Array | None = None,
    dt: float | None = None,
    solver: Any = None,
    adjoint: Any = None,
    dt0: float | None = None,
    stepsize_controller: Any = None,
    max_steps: int = 100_000,
    throw: bool = False,
) -> SimResult:
    """Compute the unit-step response of an LTI system.

    Applies a unit step to one input channel and returns the output trajectory.

    For MIMO systems, select one input channel at a time via `input_index`.

    Gradients can flow through system parameters when step timing is fixed.
    The switching time is not a differentiable input to this response.

    Args:
        sys: Continuous or discrete-time system.
        num_steps: Discrete-only number of time steps.
        duration: Continuous-only response duration.
        input_index: Input channel receiving the unit step.
        x0: Optional initial state. Defaults to zeros.
        dt: Continuous-only output sample spacing hint.
        solver: Continuous-only Diffrax solver override.
        adjoint: Continuous-only Diffrax adjoint override.
        dt0: Continuous-only initial solver step size hint.
        stepsize_controller: Native Diffrax step-size controller.
        max_steps: Continuous integration step limit.
        throw: Raise on numerical failure.

    Returns:
        SimResult: Simulation result with output times, states, and outputs.
    """
    from contrax.sim import simulate

    n = sys.A.shape[0]
    x0 = jnp.zeros(n, dtype=sys.A.dtype) if x0 is None else x0
    u_step = _unit_input(sys, input_index)
    return simulate(
        sys,
        x0,
        lambda t, x: u_step,
        num_steps=num_steps,
        duration=duration,
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )


def impulse_response(
    sys: DiscLTI | ContLTI,
    *,
    num_steps: int | None = None,
    duration: float | None = None,
    input_index: int = 0,
    x0: Array | None = None,
    dt: float | None = None,
    solver: Any = None,
    adjoint: Any = None,
    dt0: float | None = None,
    stepsize_controller: Any = None,
    max_steps: int = 100_000,
    throw: bool = False,
) -> SimResult:
    """Compute the impulse response of an LTI system.

    For discrete systems this applies a unit pulse at `k=0`. For continuous
    systems this applies the equivalent state jump `x(0+) = x0 + B[:, i]` and
    then simulates with zero input. That means the returned sampled trajectory
    omits any singular direct-feedthrough `D delta(t)` spike at `t=0`.

    The continuous-time convention represents the impulse as a state jump;
    sampled outputs omit the singular direct-feedthrough term at `t=0`.

    Args:
        sys: Continuous or discrete-time system.
        num_steps: Discrete-only number of time steps.
        duration: Continuous-only response duration.
        input_index: Input channel receiving the impulse.
        x0: Optional initial state. Defaults to zeros.
        dt: Continuous-only output sample spacing hint.
        solver: Continuous-only Diffrax solver override.
        adjoint: Continuous-only Diffrax adjoint override.
        dt0: Continuous-only initial solver step size hint.
        stepsize_controller: Native Diffrax step-size controller.
        max_steps: Continuous integration step limit.
        throw: Raise on numerical failure.

    Returns:
        SimResult: Simulation result with output times, states, and outputs.
    """
    from contrax.sim import lsim, simulate

    n = sys.A.shape[0]
    x0 = jnp.zeros(n, dtype=sys.A.dtype) if x0 is None else x0
    u_impulse = _unit_input(sys, input_index)

    if isinstance(sys, DiscLTI):
        if (
            solver is not None
            or adjoint is not None
            or dt0 is not None
            or dt is not None
            or stepsize_controller is not None
            or max_steps != 100_000
        ):
            raise ValueError("Diffrax solver configuration is continuous-only")
        if duration is not None:
            raise ValueError(
                "impulse_response() for DiscLTI requires num_steps, not duration."
            )
        if num_steps is None:
            raise ValueError("impulse_response() for DiscLTI requires num_steps.")
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a nonnegative static Python integer")
        T_int = num_steps
        us = jnp.zeros((T_int, sys.B.shape[1]), dtype=sys.B.dtype)
        if T_int:
            us = us.at[0].set(u_impulse)
        return lsim(sys, us, x0=x0, throw=throw)

    if num_steps is not None:
        raise ValueError(
            "impulse_response() for ContLTI requires duration, not num_steps."
        )
    if duration is None:
        raise ValueError("impulse_response() for ContLTI requires duration.")
    x0_impulse = x0 + sys.B[:, input_index]
    return simulate(
        sys,
        x0_impulse,
        lambda t, x: jnp.zeros(sys.B.shape[1], dtype=sys.B.dtype),
        duration=float(duration),
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )


def initial_response(
    sys: DiscLTI | ContLTI,
    x0: Array,
    *,
    num_steps: int | None = None,
    duration: float | None = None,
    dt: float | None = None,
    solver: Any = None,
    adjoint: Any = None,
    dt0: float | None = None,
    stepsize_controller: Any = None,
    max_steps: int = 100_000,
    throw: bool = False,
) -> SimResult:
    """Compute the zero-input response from a nonzero initial state.

    Args:
        sys: Continuous or discrete-time system.
        x0: Initial state. Shape: `(n,)`.
        num_steps: Discrete-only number of time steps.
        duration: Continuous-only response duration.
        dt: Continuous-only output sample spacing hint.
        solver: Continuous-only Diffrax solver override.
        adjoint: Continuous-only Diffrax adjoint override.
        dt0: Continuous-only initial solver step size hint.
        stepsize_controller: Native Diffrax step-size controller.
        max_steps: Continuous integration step limit.
        throw: Raise on numerical failure.

    Returns:
        SimResult: Simulation result with output times, states, and outputs.
    """
    from contrax.sim import simulate

    return simulate(
        sys,
        x0,
        lambda t, x: jnp.zeros(sys.B.shape[1], dtype=sys.B.dtype),
        num_steps=num_steps,
        duration=duration,
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )
