"""Moving-horizon estimation objective and solver."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from contrax._mhe_model import model_predictions, validate_window
from contrax.core import DiscLTI
from contrax.nonlinear import SupportsNonlinearSystem
from contrax.solvers import SolverStatus
from contrax.types import MHEResult


class _BestSoFarMinimiser(optx.BestSoFarMinimiser):
    """Keep the accepted-cost accumulator in the objective's precision."""

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        state = super().init(fn, y, args, options, f_struct, aux_struct, tags)
        # Optimistix's best-so-far initializer uses the global default dtype.
        # A float32 objective needs a float32 accumulator in the while-loop carry.
        return eqx.tree_at(
            lambda s: s.best_loss, state, jnp.asarray(jnp.inf, dtype=f_struct.dtype)
        )


def _weighted_sum_squares(residuals: Array, covariance: Array) -> Array:
    flat = residuals.reshape((-1, residuals.shape[-1]))
    solved = jnp.linalg.solve(covariance, flat.T).T
    return jnp.sum(flat * solved)


def soft_quadratic_penalty(residuals: Array, weight: Array) -> Array:
    """Return a quadratic soft penalty with matrix or scalar weighting.

    A scalar weight multiplies the total squared residual norm; a matrix weight
    computes sum r.T @ weight @ r over the leading residual axes. Both represent
    The weight is a precision/cost weight. Choose a nonnegative scalar or a
    positive-semidefinite matrix to obtain a nonnegative penalty.

    The penalty is differentiable with respect to residuals and weights.

    Args:
        residuals: Real residual array, with final dimension n for matrix weights.
        weight: Scalar weight or square cost matrix. Matrix shape: `(n, n)`.

    Returns:
        Array: Scalar weighted quadratic cost.
    """
    residuals = jnp.asarray(residuals)
    weight = jnp.asarray(weight)
    if jnp.iscomplexobj(residuals) or jnp.iscomplexobj(weight):
        raise ValueError("soft_quadratic_penalty requires real residuals and weights")
    if weight.ndim == 0:
        return weight * jnp.sum(residuals**2)
    if (
        residuals.ndim < 1
        or weight.ndim != 2
        or weight.shape != (residuals.shape[-1], residuals.shape[-1])
    ):
        raise ValueError("matrix weight must match the final residual dimension")
    return jnp.einsum("...i,ij,...j->", residuals, weight, residuals)


def mhe_warm_start(
    xs: Array,
    *,
    transition: Callable[[Array, Array], Array] | None = None,
    terminal_input: Array | None = None,
) -> Array:
    """Shift an MHE trajectory guess forward by one step.

    By default, repeats the previous terminal state. Supply a transition and
    `terminal_input` to propagate that state; update the arrival prior
    separately when shifting the estimation window.
    """
    xs = jnp.asarray(xs)
    if xs.ndim < 2 or xs.shape[0] == 0:
        raise ValueError("mhe_warm_start() expects a state trajectory with time axis.")
    tail = xs[-1] if transition is None else transition(xs[-1], terminal_input)
    return jnp.concatenate([xs[1:], tail[None]], axis=0)


def mhe_objective(
    model_or_f: DiscLTI | SupportsNonlinearSystem | Callable[..., Array],
    xs: Array,
    us: Array,
    ys: Array,
    x_prior: Array,
    P_prior: Array,
    Q_noise: Array,
    R_noise: Array,
    *,
    observation: Callable[..., Array] | None = None,
    params: object | None = None,
    extra_cost: Callable[..., Array] | None = None,
    ts: Array | None = None,
    measurement_us: Array | None = None,
) -> Array:
    """Evaluate a fixed-window moving-horizon-estimation objective.

    The candidate state trajectory is explicit in `xs`, so callers can optimize
    it with a JAX optimizer. Accepts discrete LTI or nonlinear system models,
    or callable transitions with an explicit observation function. There are N
    states/measurements and N-1 transitions/inputs; traced inputs must retain
    fixed shapes.

    The objective is a weighted nonlinear least-squares cost:

    - arrival cost on `xs[0] - x_prior`
    - process cost on `xs[k + 1] - f(xs[k], us[k], params)`
    - measurement cost on `ys[k] - h(xs[k], params)`
    - optional user `extra_cost(xs, us, ys, params)`

    Args:
        model_or_f: Discrete system model, or transition callable `f(x, u)`
            (`f(x, u, params)` when params are supplied).
        xs: Candidate state trajectory. Shape: `(N, n)`, N >= 1.
        us: Transition input sequence. Shape: `(N-1, m)`.
        ys: Measurement sequence aligned with xs. Shape: `(N, p)`.
        x_prior: Prior state for the start of the window. Shape: `(n,)`.
        P_prior: Arrival covariance. Shape: `(n, n)`.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        observation: Required for callable transitions; `h(x)` or `h(x, params)`.
        params: Optional parameters passed to `f`, `h`, and `extra_cost`.
        extra_cost: Optional callable adding additional smooth costs.
            Called as `extra_cost(xs, us, ys)` when `params=None`, and
            `extra_cost(xs, us, ys, params)` otherwise.
        ts: Sample times for system-model callables. Shape: `(N,)`. Defaults to
            integer steps multiplied by the model sample period when available.
        measurement_us: Observation inputs. Shape: `(N, m)`. Defaults to us
            with the last input held at the terminal observation (zero for N=1).

    Returns:
        Array: Scalar MHE cost.
    """
    validate_window(xs, us, ys, x_prior, P_prior, Q_noise, R_noise)
    arrival_residual = xs[0] - x_prior
    x_pred, y_pred = model_predictions(
        model_or_f,
        observation,
        xs,
        us,
        params,
        ts,
        measurement_us,
    )
    process_residuals = xs[1:] - x_pred
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
    model_or_f: DiscLTI | SupportsNonlinearSystem | Callable[..., Array],
    xs_init: Array,
    us: Array,
    ys: Array,
    x_prior: Array,
    P_prior: Array,
    Q_noise: Array,
    R_noise: Array,
    *,
    observation: Callable[..., Array] | None = None,
    params: object | None = None,
    extra_cost: Callable[..., Array] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    max_steps: int = 256,
    ts: Array | None = None,
    measurement_us: Array | None = None,
    throw: bool = False,
) -> MHEResult:
    """Solve a fixed-window moving-horizon-estimation problem.

    Minimises `mhe_objective` over the candidate state trajectory `xs_init`
    using an `optimistix` solver.  The default solver is
    `optx.LBFGS(rtol=1e-6, atol=1e-6)`, which is well-suited to the
    deterministic, smooth, fixed-horizon objectives that arise in MHE.

    The optimised terminal state `final_state = xs[-1]` is the natural input to a
    downstream controller or observer.  For a linear-Gaussian model, the
    full trajectory `xs` matches the RTS smoother output within solver
    tolerance.

    The LBFGS implementation is a baseline fixed-window optimizer. The best
    finite accepted iterate is retained on iteration limits; inspect
    `solver_converged`, `status`, and `gradient_norm` when assessing a result.
    Implicit JVP/VJP derivatives require a converged local stationary point
    with a nonsingular objective Hessian.
    Model, noise and optional parameter data are explicit optimizer arguments.

    Args:
        model_or_f: Discrete system model or callable transition `f(x, u)`
            (`f(x, u, params)` when params are supplied).
        xs_init: Initial trajectory guess. Shape: `(N, n)`, N >= 1. Using the
            Kalman filtered means as a warm start is recommended.
        us: Input sequence. Shape: `(N - 1, m)`.
        ys: Measurement sequence aligned with xs. Shape: `(N, p)`.
        x_prior: Prior mean for the start of the window.  Shape: `(n,)`.
        P_prior: Arrival covariance.  Shape: `(n, n)`.
        Q_noise: Process noise covariance.  Shape: `(n, n)`.
        R_noise: Measurement noise covariance.  Shape: `(p, p)`.
        observation: Required for callable transitions; `h(x)` or `h(x, params)`.
        params: Optional parameters passed to `f`, `h`, and `extra_cost`.
        extra_cost: Optional callable adding additional smooth costs.
            Called as `extra_cost(xs, us, ys)` when `params=None`, and
            `extra_cost(xs, us, ys, params)` otherwise.
        solver: Optimistix minimiser.  Defaults to
            `optx.LBFGS(rtol=1e-6, atol=1e-6)`.
        max_steps: Maximum solver iterations.  Defaults to 256.
        ts: System-model sample times. Shape: `(N,)`.
        measurement_us: Observation inputs. Shape: `(N, m)`. Defaults to
            holding the final transition control for the final observation.
        throw: Raise on failed convergence; otherwise return status diagnostics.

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
        >>> ys = jnp.array([[0.1], [0.4], [0.6], [0.5], [0.4]])
        >>> result = cx.mhe(
        ...     sys,
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
    if not isinstance(solver, optx.AbstractMinimiser):
        raise TypeError("solver must be an Optimistix minimiser")
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    validate_window(xs_init, us, ys, x_prior, P_prior, Q_noise, R_noise)

    def fn(xs, args):
        (
            model,
            obs,
            us_,
            ys_,
            x_prior_,
            P_prior_,
            Q_noise_,
            R_noise_,
            params_,
            extra,
            ts_,
            measurement_us_,
        ) = args
        return mhe_objective(
            model,
            xs,
            us_,
            ys_,
            x_prior_,
            P_prior_,
            Q_noise_,
            R_noise_,
            observation=obs,
            params=params_,
            extra_cost=extra,
            ts=ts_,
            measurement_us=measurement_us_,
        )

    # All differentiable model/noise/parameter data is explicit optimizer data,
    # so the implicit optimality-system derivative includes those dependencies.
    args = (
        model_or_f,
        observation,
        us,
        ys,
        x_prior,
        P_prior,
        Q_noise,
        R_noise,
        params,
        extra_cost,
        ts,
        measurement_us,
    )
    initial_cost = fn(xs_init, args)
    try:
        sol = optx.minimise(
            fn,
            _BestSoFarMinimiser(solver),
            xs_init,
            args=args,
            max_steps=max_steps,
            throw=False,
        )
    except TypeError as error:
        # Optimistix LBFGS versions that allocate history without a dtype use
        # JAX's default precision, which can mismatch explicit float32 states.
        # Preserve the user's arrays and global precision configuration.
        if (
            type(solver) is optx.LBFGS
            and xs_init.dtype == jnp.float32
            and jax.config.jax_enable_x64
        ):
            raise ValueError(
                "Optimistix LBFGS could not initialize float32 MHE with x64 enabled. "
                "For a small window, pass "
                "solver=optimistix.BFGS(rtol=1e-6, atol=1e-6), "
                "or select a float32 JAX context explicitly. No precision was changed."
            ) from error
        raise
    xs_opt = sol.value
    final_cost = fn(xs_opt, args)
    finite = jnp.all(jnp.isfinite(xs_opt)) & jnp.isfinite(final_cost)
    converged = sol.result == optx.RESULTS.successful
    status = jnp.where(
        ~finite,
        int(SolverStatus.numerical_failure),
        jnp.where(
            converged,
            int(SolverStatus.successful),
            jnp.where(
                sol.result == optx.RESULTS.nonlinear_max_steps_reached,
                int(SolverStatus.max_steps_reached),
                int(SolverStatus.numerical_failure),
            ),
        ),
    )
    result = MHEResult(
        xs=xs_opt,
        final_cost=final_cost,
        solver_converged=converged,
        status=status,
        num_steps=sol.stats["num_steps"],
        gradient_norm=jnp.linalg.norm(jax.grad(fn)(xs_opt, args)),
        initial_cost=initial_cost,
    )
    return (
        eqx.error_if(
            result, status != 0, "MHE solve failed; inspect status and gradient_norm"
        )
        if throw
        else result
    )
