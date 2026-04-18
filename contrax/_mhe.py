"""contrax._mhe — Moving Horizon Estimation objective and solver."""

from typing import Callable

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from contrax.types import MHEResult


def _call_transition(
    f: Callable[..., Array],
    x: Array,
    u: Array,
    params: object | None,
) -> Array:
    return f(x, u) if params is None else f(x, u, params)


def _call_observation(
    h: Callable[..., Array],
    x: Array,
    params: object | None,
) -> Array:
    return h(x) if params is None else h(x, params)


def _weighted_sum_squares(residuals: Array, covariance: Array) -> Array:
    flat = residuals.reshape((-1, residuals.shape[-1]))
    solved = jnp.linalg.solve(covariance, flat.T).T
    return jnp.sum(flat * solved)


def soft_quadratic_penalty(residuals: Array, weight: Array) -> Array:
    """Return a quadratic soft penalty with matrix or scalar weighting.

    This helper is intended for optional MHE soft costs such as state-envelope
    penalties or terminal-state regularization. It stays small on purpose:
    callers define the residuals they care about and use this helper for the
    weighted quadratic part.
    """
    residuals = jnp.asarray(residuals)
    weight = jnp.asarray(weight)
    if weight.ndim == 0:
        return weight * jnp.sum(residuals**2)
    return _weighted_sum_squares(residuals, weight)


def mhe_warm_start(
    xs: Array,
    *,
    transition: Callable[[Array, Array], Array] | None = None,
    terminal_input: Array | None = None,
) -> Array:
    """Shift an MHE trajectory guess forward by one step.

    The default behavior repeats the previous terminal state. If a transition
    and `terminal_input` are supplied, the terminal guess is propagated with
    that model instead. This is a generic rolling-window warm-start helper,
    not a full arrival-update policy.
    """
    xs = jnp.asarray(xs)
    if xs.ndim < 2:
        raise ValueError("mhe_warm_start() expects a state trajectory with time axis.")
    tail = xs[-1] if transition is None else transition(xs[-1], terminal_input)
    return jnp.concatenate([xs[1:], tail[None]], axis=0)


def mhe_objective(
    f: Callable[..., Array],
    h: Callable[..., Array],
    xs: Array,
    us: Array,
    ys: Array,
    x_prior: Array,
    P_prior: Array,
    Q_noise: Array,
    R_noise: Array,
    params: object | None = None,
    extra_cost: Callable[..., Array] | None = None,
) -> Array:
    """Evaluate a fixed-window moving-horizon-estimation objective.

    This is the first MHE primitive in Contrax: a pure cost function, not a
    solver. The candidate state trajectory is explicit in `xs`, so callers can
    optimize it with their preferred JAX-native solver while keeping fixed
    horizon shapes.

    The objective is a weighted nonlinear least-squares cost:

    - arrival cost on `xs[0] - x_prior`
    - process cost on `xs[k + 1] - f(xs[k], us[k], params)`
    - measurement cost on `ys[k] - h(xs[k], params)`
    - optional user `extra_cost(xs, us, ys, params)`

    Args:
        f: Transition function. Called as `f(x, u)` when `params=None`, and
            `f(x, u, params)` otherwise.
        h: Observation function. Called as `h(x)` when `params=None`, and
            `h(x, params)` otherwise.
        xs: Candidate state trajectory. Shape: `(T + 1, n)`.
        us: Input sequence. Shape: `(T, m)`.
        ys: Measurement sequence aligned with `xs`. Shape: `(T + 1, p)`.
        x_prior: Prior state for the start of the window. Shape: `(n,)`.
        P_prior: Arrival covariance. Shape: `(n, n)`.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        params: Optional parameters passed to `f`, `h`, and `extra_cost`.
        extra_cost: Optional callable adding additional smooth costs.
            Called as `extra_cost(xs, us, ys)` when `params=None`, and
            `extra_cost(xs, us, ys, params)` otherwise.

    Returns:
        Array: Scalar MHE cost.
    """
    arrival_residual = xs[0] - x_prior
    x_pred = jax.vmap(lambda x, u: _call_transition(f, x, u, params))(
        xs[:-1],
        us,
    )
    process_residuals = xs[1:] - x_pred
    y_pred = jax.vmap(lambda x: _call_observation(h, x, params))(xs)
    measurement_residuals = ys - y_pred

    cost = (
        _weighted_sum_squares(arrival_residual[None], P_prior)
        + _weighted_sum_squares(process_residuals, Q_noise)
        + _weighted_sum_squares(measurement_residuals, R_noise)
    )
    if extra_cost is not None:
        if params is None:
            cost = cost + extra_cost(xs, us, ys)
        else:
            cost = cost + extra_cost(xs, us, ys, params)
    return cost


def mhe(
    f: Callable[..., Array],
    h: Callable[..., Array],
    xs_init: Array,
    us: Array,
    ys: Array,
    x_prior: Array,
    P_prior: Array,
    Q_noise: Array,
    R_noise: Array,
    params: object | None = None,
    extra_cost: Callable[..., Array] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    max_steps: int = 256,
) -> MHEResult:
    """Solve a fixed-window moving-horizon-estimation problem.

    Minimises `mhe_objective` over the candidate state trajectory `xs_init`
    using an `optimistix` solver.  The default solver is
    `optx.LBFGS(rtol=1e-6, atol=1e-6)`, which is well-suited to the
    deterministic, smooth, fixed-horizon objectives that arise in MHE.

    The optimised terminal state `x_hat = xs[-1]` is the natural input to a
    downstream controller or observer.  For a linear-Gaussian model, the
    full trajectory `xs` matches the RTS smoother output within solver
    tolerance.

    Args:
        f: Transition function.  Called as `f(x, u)` when `params=None`,
            and `f(x, u, params)` otherwise.
        h: Observation function.  Called as `h(x)` when `params=None`,
            and `h(x, params)` otherwise.
        xs_init: Initial trajectory guess.  Shape: `(T, n)`.  Using the
            Kalman filtered means as a warm start is recommended.
        us: Input sequence.  Shape: `(T - 1, m)`.
        ys: Measurement sequence aligned with `xs`.  Shape: `(T, p)`.
        x_prior: Prior mean for the start of the window.  Shape: `(n,)`.
        P_prior: Arrival covariance.  Shape: `(n, n)`.
        Q_noise: Process noise covariance.  Shape: `(n, n)`.
        R_noise: Measurement noise covariance.  Shape: `(p, p)`.
        params: Optional parameters passed to `f`, `h`, and `extra_cost`.
        extra_cost: Optional callable adding additional smooth costs.
            Called as `extra_cost(xs, us, ys)` when `params=None`, and
            `extra_cost(xs, us, ys, params)` otherwise.
        solver: Optimistix minimiser.  Defaults to
            `optx.LBFGS(rtol=1e-6, atol=1e-6)`.
        max_steps: Maximum solver iterations.  Defaults to 256.

    Returns:
        [MHEResult][contrax.types.MHEResult]: Estimated trajectory, terminal
            state, final cost, and convergence flag.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[0.8]]), jnp.zeros((1, 1)),
        ...     jnp.array([[1.0]]), jnp.zeros((1, 1)), dt=1.0,
        ... )
        >>> f = lambda x, u: sys.A @ x
        >>> h = lambda x: sys.C @ x
        >>> ys = jnp.array([[0.1], [0.4], [0.6], [0.5], [0.4]])
        >>> result = cx.mhe(
        ...     f, h,
        ...     xs_init=jnp.zeros((5, 1)),
        ...     us=jnp.zeros((4, 1)),
        ...     ys=ys,
        ...     x_prior=jnp.zeros(1),
        ...     P_prior=jnp.eye(1),
        ...     Q_noise=0.05 * jnp.eye(1),
        ...     R_noise=0.2 * jnp.eye(1),
        ... )
    """
    if solver is None:
        solver = optx.LBFGS(rtol=1e-6, atol=1e-6)

    def fn(xs, args):
        us_, ys_, x_prior_, P_prior_, Q_noise_, R_noise_ = args
        return mhe_objective(
            f,
            h,
            xs,
            us_,
            ys_,
            x_prior_,
            P_prior_,
            Q_noise_,
            R_noise_,
            params,
            extra_cost,
        )

    args = (us, ys, x_prior, P_prior, Q_noise, R_noise)
    sol = optx.minimise(
        fn, solver, xs_init, args=args, max_steps=max_steps, throw=False
    )
    xs_opt = sol.value
    return MHEResult(
        xs=xs_opt,
        x_hat=xs_opt[-1],
        cost=fn(xs_opt, args),
        converged=sol.result == optx.RESULTS.successful,
    )
