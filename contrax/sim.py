"""contrax.sim — simulation primitives."""

import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from contrax.core import ContLTI, DiscLTI
from contrax.nonlinear import NonlinearSystem
from contrax.phs import PHSSystem


def rollout(
    f: Callable[..., Array],
    x0: Array,
    us: Array,
    params: Any = None,
) -> Array:
    """Roll out arbitrary discrete-time dynamics over a fixed input sequence.

    This is the generic nonlinear trajectory primitive in Contrax. Unlike
    `lsim()` and `simulate()`, it does not require an LTI system object and does
    not compute outputs. It simply applies a transition function with
    `jax.lax.scan`.

    Args:
        f: Transition function. If `params` is omitted, `f(x, u)` is called.
            Otherwise, `f(x, u, params)` is called.
        x0: Initial state. Shape: `(n,)` or any fixed-shape state pytree.
        us: Input sequence. Shape: `(T, m)` or any fixed-leading-axis input
            pytree accepted by `jax.lax.scan`.
        params: Optional static or array-valued parameters passed to `f`.

    Returns:
        Array: State trajectory including the initial state. Shape:
        `(T + 1, ...)` for array states.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def f(x, u, gain):
        ...     return gain * x + u
        >>> xs = cx.rollout(f, jnp.array([1.0]), jnp.zeros((10, 1)), 0.9)
    """

    def step(x, u):
        x_next = f(x, u) if params is None else f(x, u, params)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None], xs], axis=0)


def foh_inputs(us: Array) -> Array:
    """Build first-order-hold endpoint pairs from a sample sequence.

    Given inputs `u_k`, returns a sequence with shape `(T, 2, m)` where each
    item contains `(u_k, u_{k+1})`, using `u_{T-1}` for the final right
    endpoint.
    """
    if us.ndim < 2:
        raise ValueError("foh_inputs() expects an input array with leading time axis.")
    return jnp.stack([us, jnp.concatenate([us[1:], us[-1:]], axis=0)], axis=1)


def lsim(
    sys: DiscLTI,
    us: Array,  # (T, m) input sequence
    x0: Array | None = None,  # (n,) initial state; defaults to zeros
) -> tuple[Array, Array, Array]:
    """Simulate a discrete system in open loop from an input sequence.

    Mirrors MATLAB-style `lsim()` for the current discrete state-space surface.

    This is the public open-loop simulation name in Contrax. The implementation
    is a pure `lax.scan`, so it composes naturally with `jit`, `vmap`, and
    differentiation on fixed-shape inputs.

    Args:
        sys: Discrete-time system.
        us: Input sequence. Shape: `(T, m)`.
        x0: Initial state. Shape: `(n,)`. Defaults to zeros.

    Returns:
        tuple[Array, Array, Array]: Simulation outputs `(ts, xs, ys)`. Shapes:
        `(T,)`, `(T + 1, n)`, and `(T, p)`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        ...     jnp.array([[0.0], [0.1]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ...     dt=0.1,
        ... )
        >>> ts, xs, ys = cx.lsim(sys, jnp.zeros((20, 1)))
    """
    n = sys.A.shape[0]
    if x0 is None:
        x0 = jnp.zeros(n, dtype=sys.A.dtype)

    T = us.shape[0]
    ts = jnp.arange(T, dtype=sys.A.dtype) * sys.dt

    def step(x, u):
        x_next = sys.A @ x + sys.B @ u
        y = sys.C @ x + sys.D @ u
        return x_next, (x_next, y)

    _, (xs, ys) = jax.lax.scan(step, x0, us)
    xs_full = jnp.concatenate([x0[None], xs], axis=0)
    return ts, xs_full, ys


def _simulate_discrete(
    sys: DiscLTI,
    x0: Array,
    policy: Callable[[float, Array], Array],
    T: int,
) -> tuple[Array, Array, Array]:
    ts = jnp.arange(T, dtype=x0.dtype) * sys.dt

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
) -> tuple[Array, Array, Array]:
    if sys.dt is None:
        raise ValueError("Discrete nonlinear simulation requires sys.dt.")
    ts = jnp.arange(T, dtype=x0.dtype) * sys.dt

    def step(x, t):
        u = policy(t, x)
        x_next = sys.dynamics(t, x, u)
        y = sys.output(t, x, u)
        return x_next, (x_next, y)

    _, (xs, ys) = jax.lax.scan(step, x0, ts)
    xs_full = jnp.concatenate([x0[None], xs], axis=0)
    return ts, xs_full, ys


def _build_continuous_save_grid(duration: float, dt: float | None, dtype: Any) -> Array:
    if duration <= 0.0:
        raise ValueError("Continuous simulate() requires duration > 0.")

    if dt is None:
        n_intervals = 200
    else:
        if dt <= 0.0:
            raise ValueError("Continuous simulate() requires dt > 0 when provided.")
        n_intervals = max(1, math.ceil(duration / dt))

    return jnp.linspace(0.0, duration, n_intervals + 1, dtype=dtype)


def _continuous_input_fn(
    u: Array,
    tau: Array,
    dt: Array,
    interpolation: str,
) -> Array:
    if interpolation == "zoh":
        return u
    if interpolation == "foh":
        u0 = u[0]
        u1 = u[1]
        weight = tau / dt
        return (1.0 - weight) * u0 + weight * u1
    raise ValueError("input_interpolation must be 'zoh' or 'foh'.")


def sample_system(
    sys: NonlinearSystem | PHSSystem,
    dt: float,
    *,
    input_interpolation: str = "zoh",
    solver: Any = None,
    adjoint: Any = None,
    dt0: float | None = None,
) -> NonlinearSystem:
    """Sample a continuous nonlinear system into a discrete transition model.

    This helper builds a discrete-time
    [NonlinearSystem][contrax.systems.NonlinearSystem] whose one-step dynamics
    are obtained by integrating the continuous model over one sample interval.

    With `input_interpolation="zoh"`, each discrete input has shape `(m,)` and
    is held constant over the interval. With `input_interpolation="foh"`, each
    discrete input has shape `(2, m)` and is interpreted as the pair
    `(u_k, u_{k+1})`; use `foh_inputs()` to build those pairs from a sampled
    input sequence.
    """
    if sys.dt is not None:
        raise ValueError("sample_system() expects a continuous system with dt=None.")
    if input_interpolation not in {"zoh", "foh"}:
        raise ValueError("input_interpolation must be 'zoh' or 'foh'.")

    try:
        import diffrax
    except ImportError:
        raise ImportError("diffrax is required for sample_system()")

    dt_arr = jnp.asarray(dt, dtype=float)
    if solver is None:
        solver = diffrax.Tsit5()
    if adjoint is None:
        adjoint = diffrax.DirectAdjoint()

    def dynamics(t: Array | float, x: Array, u: Array) -> Array:
        local_dt = dt_arr.astype(x.dtype)
        local_dt0 = local_dt if dt0 is None else jnp.asarray(dt0, dtype=x.dtype)

        def vf(tau, x_tau, args):
            base_t, u_step = args
            u_tau = _continuous_input_fn(u_step, tau, local_dt, input_interpolation)
            return sys.dynamics(base_t + tau, x_tau, u_tau)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vf),
            solver,
            t0=jnp.asarray(0.0, dtype=x.dtype),
            t1=local_dt,
            dt0=local_dt0,
            y0=x,
            args=(jnp.asarray(t, dtype=x.dtype), u),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            adjoint=adjoint,
            max_steps=100_000,
        )
        return solution.ys[0]

    def output(t: Array | float, x: Array, u: Array) -> Array:
        if input_interpolation == "foh":
            return sys.output(t, x, u[0])
        return sys.output(t, x, u)

    return NonlinearSystem(
        dynamics=dynamics,
        observation=output,
        dt=dt_arr,
        state_dim=sys.state_dim,
        input_dim=sys.input_dim,
        output_dim=sys.output_dim,
    )


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
) -> tuple[Array, Array, Array]:
    try:
        import diffrax
    except ImportError:
        raise ImportError("diffrax is required for continuous simulate()")

    duration = float(T)
    ts = _build_continuous_save_grid(duration, dt, x0.dtype)
    t0 = jnp.asarray(0.0, dtype=x0.dtype)
    t1 = jnp.asarray(duration, dtype=x0.dtype)
    dt0 = jnp.asarray(dt0 if dt0 is not None else ts[1] - ts[0], dtype=x0.dtype)

    if solver is None:
        solver = diffrax.Tsit5()
    if adjoint is None:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    term = as_ode_term(sys, policy)
    saveat = diffrax.SaveAt(ts=ts)
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)
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
        max_steps=100_000,
    )

    xs = solution.ys
    if isinstance(sys, ContLTI):
        ys = jax.vmap(lambda t, x: sys.C @ x + sys.D @ policy(t, x))(ts, xs)
    else:
        ys = jax.vmap(lambda t, x: sys.output(t, x, policy(t, x)))(ts, xs)
    return ts, xs, ys


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
) -> tuple[Array, Array, Array]:
    """Simulate a system in closed loop under a control policy.

    Evaluates `policy(t, x)` along the system trajectory and dispatches to the
    appropriate simulation path for discrete or continuous LTI models.

    For [DiscLTI][contrax.systems.DiscLTI], this keeps the existing
    pure-`lax.scan` path and requires `num_steps`. For
    [ContLTI][contrax.systems.ContLTI], this uses Diffrax under the hood and
    requires `duration`. Continuous trajectories are sampled on a fixed uniform
    save grid so output shapes stay predictable.

    The continuous path intentionally exposes only a small amount of Diffrax
    configuration: a sample spacing hint `dt`, plus optional `solver`,
    `adjoint`, and `dt0` overrides. The default continuous solver is
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

    Returns:
        tuple[Array, Array, Array]: Simulation outputs `(ts, xs, ys)`. Shapes:
        for [DiscLTI][contrax.systems.DiscLTI], `(T,)`, `(T + 1, n)`, and
        `(T, p)`; for [ContLTI][contrax.systems.ContLTI], `(N,)`, `(N, n)`,
        and `(N, p)` where `N` is the number of saved samples on the
        continuous output grid.

    Examples:
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
    if isinstance(sys, DiscLTI):
        if duration is not None:
            raise ValueError("simulate() for DiscLTI requires num_steps, not duration.")
        if num_steps is None:
            raise ValueError("simulate() for DiscLTI requires num_steps.")
        return _simulate_discrete(sys, x0, policy, int(num_steps))
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
        return _simulate_nonlinear_discrete(sys, x0, policy, int(num_steps))
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
    )


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
) -> tuple[Array, Array]:
    """Compute the unit-step response of an LTI system.

    This is the standard control sanity-check response: apply a unit step on
    one input channel and return the output trajectory.

    For MIMO systems, this first version follows a MATLAB-like workflow by
    selecting one input channel at a time via `input_index`.

    As an analysis helper, this is usually fine to differentiate with respect
    to system parameters when the step timing is fixed. It is not the right
    path for differentiating with respect to the discontinuity itself, such as
    a switching time or other hard event location.

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

    Returns:
        tuple[Array, Array]: Sample times and output trajectory `(ts, ys)`.
    """
    n = sys.A.shape[0]
    x0 = jnp.zeros(n, dtype=sys.A.dtype) if x0 is None else x0
    u_step = _unit_input(sys, input_index)
    ts, _, ys = simulate(
        sys,
        x0,
        lambda t, x: u_step,
        num_steps=num_steps,
        duration=duration,
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
    )
    return ts, ys


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
) -> tuple[Array, Array]:
    """Compute the impulse response of an LTI system.

    For discrete systems this applies a unit pulse at `k=0`. For continuous
    systems this applies the equivalent state jump `x(0+) = x0 + B[:, i]` and
    then simulates with zero input. That means the returned sampled trajectory
    omits any singular direct-feedthrough `D delta(t)` spike at `t=0`.

    This is an analysis helper rather than a smooth-input differentiation path.
    In particular, the continuous-time version uses the standard state-jump
    convention instead of representing a Dirac delta as an ordinary control
    signal.

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

    Returns:
        tuple[Array, Array]: Sample times and output trajectory `(ts, ys)`.
    """
    n = sys.A.shape[0]
    x0 = jnp.zeros(n, dtype=sys.A.dtype) if x0 is None else x0
    u_impulse = _unit_input(sys, input_index)

    if isinstance(sys, DiscLTI):
        if duration is not None:
            raise ValueError(
                "impulse_response() for DiscLTI requires num_steps, not duration."
            )
        if num_steps is None:
            raise ValueError("impulse_response() for DiscLTI requires num_steps.")
        T_int = int(num_steps)
        us = jnp.zeros((T_int, sys.B.shape[1]), dtype=sys.B.dtype)
        us = us.at[0].set(u_impulse)
        ts, _, ys = lsim(sys, us, x0=x0)
        return ts, ys

    if num_steps is not None:
        raise ValueError(
            "impulse_response() for ContLTI requires duration, not num_steps."
        )
    if duration is None:
        raise ValueError("impulse_response() for ContLTI requires duration.")
    x0_impulse = x0 + sys.B[:, input_index]
    ts, _, ys = simulate(
        sys,
        x0_impulse,
        lambda t, x: jnp.zeros(sys.B.shape[1], dtype=sys.B.dtype),
        duration=float(duration),
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
    )
    return ts, ys


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
) -> tuple[Array, Array]:
    """Compute the zero-input response from a nonzero initial state.

    This is the third standard inspection response alongside step and impulse.

    Args:
        sys: Continuous or discrete-time system.
        x0: Initial state. Shape: `(n,)`.
        num_steps: Discrete-only number of time steps.
        duration: Continuous-only response duration.
        dt: Continuous-only output sample spacing hint.
        solver: Continuous-only Diffrax solver override.
        adjoint: Continuous-only Diffrax adjoint override.
        dt0: Continuous-only initial solver step size hint.

    Returns:
        tuple[Array, Array]: Sample times and output trajectory `(ts, ys)`.
    """
    ts, _, ys = simulate(
        sys,
        x0,
        lambda t, x: jnp.zeros(sys.B.shape[1], dtype=sys.B.dtype),
        num_steps=num_steps,
        duration=duration,
        dt=dt,
        solver=solver,
        adjoint=adjoint,
        dt0=dt0,
    )
    return ts, ys


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
