"""contrax.sim — simulation primitives."""

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from contrax._integration import _diffrax_status
from contrax._lsim import lsim
from contrax._responses import impulse_response, initial_response, step_response
from contrax._rollout import foh_inputs, rollout
from contrax._sampling import sample_system
from contrax.core import ContLTI, DiscLTI
from contrax.nonlinear import NonlinearSystem
from contrax.phs import PHSSystem
from contrax.types import SimResult

__all__ = [
    "rollout",
    "foh_inputs",
    "sample_system",
    "lsim",
    "simulate",
    "step_response",
    "impulse_response",
    "initial_response",
    "as_ode_term",
]


def _simulate_discrete(
    sys: DiscLTI,
    x0: Array,
    policy: Callable[[float, Array], Array],
    T: int,
) -> tuple[Array, Array, Array]:
    ts = jnp.arange(T, dtype=x0.real.dtype) * jnp.asarray(sys.dt, dtype=x0.real.dtype)

    def step(x, t):
        u = policy(t, x)
        x_next = sys.A @ x + sys.B @ u
        y = sys.C @ x + sys.D @ u
        return x_next, (x_next, y)

    _, (xs, ys) = jax.lax.scan(step, x0, ts)
    xs_full = jnp.concatenate([x0[None], xs], axis=0)
    return ts, xs_full, ys


def _simulate_nonlinear_discrete(
    sys: NonlinearSystem | PHSSystem,
    x0: Array,
    policy: Callable[[float, Array], Array],
    T: int,
) -> SimResult:
    if sys.dt is None:
        raise ValueError("Discrete nonlinear simulation requires sys.dt.")
    ts = jnp.arange(T, dtype=x0.real.dtype) * jnp.asarray(sys.dt, dtype=x0.real.dtype)

    def step(carry, t):
        x, status, count = carry
        u = policy(t, x)
        y = sys.output(t, x, u)

        def advance():
            if isinstance(sys, NonlinearSystem):
                result = sys.step(t, x, u)
                code = jnp.where(
                    (result.status == 0) & ~result.successful, 3, result.status
                )
                return result.x, code, result.num_steps.astype(count.dtype)
            next_x = sys.dynamics(t, x, u)
            return (
                next_x,
                jnp.where(jnp.all(jnp.isfinite(next_x)), 0, 3),
                jnp.ones_like(count),
            )

        x_next, code, steps = jax.lax.cond(
            status == 0,
            advance,
            lambda: (x, status, jnp.zeros_like(count)),
        )
        return (x_next, code, count + steps), (x_next, y)

    (_, status, count), (xs, ys) = jax.lax.scan(
        step,
        (x0, jnp.asarray(0), jnp.asarray(0)),
        ts,
    )
    xs_full = jnp.concatenate([x0[None], xs], axis=0)
    return SimResult(ts, xs_full, ys, solver_status=status, num_steps=count)


def _build_continuous_save_grid(duration: float, dt: float | None, dtype: Any) -> Array:
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("Continuous simulate() requires finite duration >= 0.")
    if duration == 0:
        return jnp.zeros((1,), dtype=dtype)

    if dt is None:
        n_intervals = 200
    else:
        if dt <= 0.0:
            raise ValueError("Continuous simulate() requires dt > 0 when provided.")
        n_intervals = max(1, math.ceil(duration / dt))

    return jnp.linspace(0.0, duration, n_intervals + 1, dtype=dtype)


def _simulate_continuous(
    sys: ContLTI | NonlinearSystem | PHSSystem,
    x0: Array,
    policy: Callable[[float, Array], Array],
    T: float,
    *,
    dt: float | None,
    solver: Any,
    adjoint: Any,
    dt0: float | None,
    stepsize_controller: Any,
    max_steps: int,
    throw: bool,
) -> SimResult:
    try:
        import diffrax
    except ImportError:
        raise ImportError("diffrax is required for continuous simulate()")

    duration = float(T)
    ts = _build_continuous_save_grid(duration, dt, x0.dtype)
    if duration == 0:
        u0 = policy(ts[0], x0)
        y0 = (
            sys.C @ x0 + sys.D @ u0
            if isinstance(sys, ContLTI)
            else sys.output(ts[0], x0, u0)
        )
        result = SimResult(ts, x0[None], y0[None])
        return (
            eqx.error_if(result, ~result.successful, "Continuous simulation failed")
            if throw
            else result
        )
    t0 = jnp.asarray(0.0, dtype=x0.dtype)
    t1 = jnp.asarray(duration, dtype=x0.dtype)
    dt0 = jnp.asarray(dt0 if dt0 is not None else ts[1] - ts[0], dtype=x0.dtype)

    if solver is None:
        solver = diffrax.Tsit5()
    if adjoint is None:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    term = as_ode_term(sys, policy)
    saveat = diffrax.SaveAt(ts=ts)
    stepsize_controller = (
        diffrax.PIDController(rtol=1e-6, atol=1e-9)
        if stepsize_controller is None
        else stepsize_controller
    )
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=x0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        max_steps=max_steps,
        throw=False,
    )

    xs = solution.ys
    if isinstance(sys, ContLTI):
        ys = jax.vmap(lambda t, x: sys.C @ x + sys.D @ policy(t, x))(ts, xs)
    else:
        ys = jax.vmap(lambda t, x: sys.output(t, x, policy(t, x)))(ts, xs)
    result = SimResult(
        ts,
        xs,
        ys,
        solver_status=_diffrax_status(solution),
        num_steps=solution.stats["num_steps"],
    )
    return (
        eqx.error_if(result, ~result.successful, "Continuous simulation failed")
        if throw
        else result
    )


def simulate(
    sys: DiscLTI | ContLTI | NonlinearSystem | PHSSystem,
    x0: Array,  # (n,)
    policy: Callable[[float, Array], Array],  # (t, x) -> u
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
    """Simulate a system in closed loop under a control policy.

    Evaluates `policy(t, x)` along the system trajectory and dispatches to the
    appropriate simulation path for discrete or continuous LTI models.

    For [DiscLTI][contrax.systems.DiscLTI], this keeps the existing
    pure-`lax.scan` path and requires `num_steps`. For
    [ContLTI][contrax.systems.ContLTI], this uses Diffrax under the hood and
    requires `duration`. Continuous trajectories are sampled on a fixed uniform
    save grid so output shapes stay predictable.

    Continuous integration accepts native Diffrax solvers, adjoints and
    step-size controllers, plus an initial step and integration limit.
    The default continuous solver is
    `diffrax.Tsit5()` with `diffrax.RecursiveCheckpointAdjoint()`.

    Args:
        sys: Continuous or discrete-time system.
        x0: Initial state. Shape: `(n,)`.
        policy: Control policy mapping `(t, x)` to an input `u`.
        num_steps: Discrete-only number of time steps.
        duration: Continuous-only simulation duration.
        dt: Continuous-only output sample spacing hint. When omitted, the
            continuous path returns 201 samples including `t=0` and `t=T`.
        solver: Continuous-only Diffrax solver override.
        adjoint: Continuous-only Diffrax adjoint override.
        dt0: Continuous-only initial solver step size hint.
        stepsize_controller: Native Diffrax step-size controller. Defaults to
            PIDController(rtol=1e-6, atol=1e-9).
        max_steps: Continuous adaptive integration step limit.
        throw: Raise on numerical failure, otherwise return status diagnostics.

    Returns:
        SimResult: Simulation result with `.ts`, `.xs`, `.ys`. Supports tuple
        unpacking. For [DiscLTI][contrax.systems.DiscLTI], shapes are `(T,)`,
        `(T + 1, n)`, and `(T, p)`; for continuous systems, `(N,)`, `(N, n)`,
        and `(N, p)` where `N` is the number of saved samples on the output grid.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys_d = cx.dss(
        ...     jnp.array([[1.0, 0.05], [0.0, 1.0]]),
        ...     jnp.array([[0.0], [0.05]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ...     dt=0.05,
        ... )
        >>> K = cx.lqr(sys_d, jnp.eye(2), jnp.array([[1.0]])).K
        >>> ts, xs, ys = cx.simulate(
        ...     sys_d,
        ...     jnp.array([1.0, 0.0]),
        ...     lambda t, x: -K @ x,
        ...     num_steps=60,
        ... )
    """
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    discrete = isinstance(sys, DiscLTI) or (
        isinstance(sys, (NonlinearSystem, PHSSystem)) and sys.dt is not None
    )
    if discrete:
        if (
            solver is not None
            or adjoint is not None
            or dt0 is not None
            or dt is not None
            or stepsize_controller is not None
            or max_steps != 100_000
        ):
            raise ValueError("Diffrax solver configuration is continuous-only")
        if num_steps is not None and (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a nonnegative static Python integer")
    if isinstance(sys, DiscLTI):
        if duration is not None:
            raise ValueError("simulate() for DiscLTI requires num_steps, not duration.")
        if num_steps is None:
            raise ValueError("simulate() for DiscLTI requires num_steps.")
        result = SimResult(
            *_simulate_discrete(sys, x0, policy, num_steps),
            num_steps=jnp.asarray(num_steps),
        )
        return (
            eqx.error_if(result, ~result.successful, "Discrete simulation failed")
            if throw
            else result
        )
    if isinstance(sys, (NonlinearSystem, PHSSystem)) and sys.dt is not None:
        if duration is not None:
            raise ValueError(
                "simulate() for discrete nonlinear systems requires "
                "num_steps, not duration."
            )
        if num_steps is None:
            raise ValueError(
                "simulate() for discrete nonlinear systems requires num_steps."
            )
        result = _simulate_nonlinear_discrete(sys, x0, policy, num_steps)
        return (
            eqx.error_if(result, ~result.successful, "Discrete simulation failed")
            if throw
            else result
        )
    if num_steps is not None:
        if isinstance(sys, ContLTI):
            raise ValueError("simulate() for ContLTI requires duration, not num_steps.")
        raise ValueError(
            "simulate() for continuous nonlinear systems requires duration, "
            "not num_steps."
        )
    if duration is None:
        if isinstance(sys, ContLTI):
            raise ValueError("simulate() for ContLTI requires duration.")
        raise ValueError(
            "simulate() for continuous nonlinear systems requires duration."
        )
    return _simulate_continuous(
        sys,
        x0,
        policy,
        float(duration),
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=throw,
    )


def as_ode_term(
    sys: ContLTI | NonlinearSystem | PHSSystem,
    control_fn: Callable[[float, Array], Array],
) -> Any:
    """Wrap a continuous system as a Diffrax `ODETerm`.

    This is the lower-level bridge from
    [ContLTI][contrax.systems.ContLTI] into direct Diffrax workflows when the
    public `simulate()` surface is not the right level of control.

    This function does not simulate by itself. Its behavior under `jit`,
    `vmap`, and `grad` follows the downstream Diffrax solver and adjoint
    configuration used by the caller.

    Args:
        sys: Continuous-time system.
        control_fn: Control function mapping `(t, x)` to an input `u`.

    Returns:
        diffrax.ODETerm: An ODE term representing `x_dot = A @ x + B @ u(t, x)`.

    Raises:
        ImportError: If Diffrax is not installed.
    """
    try:
        import diffrax
    except ImportError:
        raise ImportError("diffrax is required for as_ode_term()")

    def vector_field(t, x, args):
        del args
        u = control_fn(t, x)
        if isinstance(sys, ContLTI):
            return sys.A @ x + sys.B @ u
        return sys.dynamics(t, x, u)

    return diffrax.ODETerm(vector_field)
