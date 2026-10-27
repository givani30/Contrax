"""contrax._ukf — Unscented Kalman filter."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

from contrax._filter_checks import checked, input_status
from contrax._filter_common import prepare_filter
from contrax._kalman import _symmetrize
from contrax.nonlinear import (
    DynamicsLike,
    ObservationFn,
    _coerce_dynamics,
    _coerce_observation,
    _is_system_model,
    _system_dt,
)
from contrax.solvers import UKF
from contrax.types import RTSResult, UKFResult


def _ukf_weights(
    n: int,
    alpha: float,
    beta: float,
    kappa: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    lambda_ = jnp.asarray(alpha**2 * (n + kappa) - n, dtype=dtype)
    c = jnp.asarray(n, dtype=dtype) + lambda_
    base = 1.0 / (2.0 * c)
    Wm = jnp.full((2 * n + 1,), base, dtype=dtype).at[0].set(lambda_ / c)
    Wc = Wm.at[0].set(lambda_ / c + (1.0 - alpha**2 + beta))
    return Wm, Wc, c


# Reference: Julier and Uhlmann (2004), "Unscented Filtering and Nonlinear
# Estimation", Proceedings of the IEEE.
def _sigma_points(
    mean: Array,
    cov: Array,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[Array, Array, Array]:
    n = mean.shape[0]
    cov = _symmetrize(cov)
    Wm, Wc, c = _ukf_weights(n, alpha, beta, kappa, cov.dtype)
    chol = jnp.linalg.cholesky(c * cov)
    offsets = jnp.swapaxes(chol, 0, 1)
    sigma = jnp.concatenate(
        [mean[None], mean[None] + offsets, mean[None] - offsets],
        axis=0,
    )
    return sigma, Wm, Wc


def _weighted_covariance(
    centered: Array,
    weights: Array,
) -> Array:
    return jnp.einsum("i,ij,ik->jk", weights, centered, centered)


def _cross_covariance(
    x_centered: Array,
    y_centered: Array,
    weights: Array,
) -> Array:
    return jnp.einsum("i,ij,ik->jk", weights, x_centered, y_centered)


def ukf(
    model_or_f: DynamicsLike,
    ys: Array,
    *,
    Q: Array,
    R: Array,
    us: Array | None = None,
    x0: Array | None = None,
    P0: Array | None = None,
    observation: ObservationFn | None = None,
    has_measurement: Array | None = None,
    solver: UKF | None = None,
    throw: bool = False,
) -> UKFResult:
    """Run an unscented Kalman filter for nonlinear dynamics and observations.

    Uses a deterministic sigma-point transform instead of local Jacobians to
    propagate means and covariances through the nonlinear transition and
    observation models.

    Assumes additive process and measurement noise and positive-definite
    sigma-point/innovation covariances. No jitter or covariance repair is added.
    Supports jit, vmap, JVP and VJP through JAX model callbacks. A single
    innovation Cholesky factor serves both the gain and Gaussian likelihood.

    Args:
        model_or_f: Nonlinear system model or dynamics function.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        ys: Measurement sequence. Shape: `(T, p)`.
        us: Optional controls, shape `(T, m)` or `(T, 2, m)` for sampled FOH.
            Defaults to zero inputs; declare model.input_dim or pass us for
            controlled nonlinear models.
        x0: Initial prior mean, shape `(n,)`. Defaults to zeros.
        P0: Initial prior covariance, shape `(n, n)`. Defaults to identity.
        observation: Observation function `h(x)` for plain dynamics callables.
            Omit this when passing a system model with its own output map.
        has_measurement: Whole-observation flags, shape `(T,)`. Missing samples
            may contain NaNs; their innovation, predicted measurement,
            innovation covariance and log-likelihood term are zero placeholders.
        solver: Sigma-point settings, default contrax.solvers.UKF().

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        [UKFResult][contrax.types.UKFResult]: A bundle containing filtered
            means/covariances plus prediction, innovation, and likelihood
            intermediates useful for diagnostics and smoothing.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def f(t, x, u):
        ...     return jnp.array([x[0] + 0.1 * x[1], x[1] + u[0]])
        >>> def h(x):
        ...     return jnp.array([x[0] ** 2])
        >>> result = cx.ukf(
        ...     f, jnp.zeros((10, 1)), Q=1e-3 * jnp.eye(2), R=1e-2 * jnp.eye(1),
        ...     us=jnp.zeros((10, 1)), x0=jnp.ones(2), observation=h,
        ... )
    """

    solver = UKF() if solver is None else solver
    if not isinstance(solver, UKF):
        raise TypeError("ukf solver must be contrax.solvers.UKF")
    ys, Q, R, us, x0, P0, flags = prepare_filter(
        model_or_f, ys, Q, R, us, x0, P0, has_measurement
    )
    alpha, beta, kappa = solver.alpha, solver.beta, solver.kappa
    if x0.shape[0] + kappa <= 0:
        raise ValueError("UKF requires n+kappa > 0")
    if _is_system_model(model_or_f):
        if observation is not None:
            raise ValueError("ukf(sys, ...) does not accept observation=.")
        obs = model_or_f
    else:
        if observation is None:
            raise ValueError(
                "ukf(f, ..., observation=...) requires an observation function."
            )
        obs = observation

    f = _coerce_dynamics(model_or_f)
    h = _coerce_observation(
        obs,
        None if _is_system_model(obs) else obs,
    )
    dt = _system_dt(model_or_f)
    ts = jnp.arange(ys.shape[0], dtype=x0.dtype)
    if dt is not None:
        ts = ts * jnp.asarray(dt, dtype=x0.dtype)

    # Update-then-predict: x0/P0 is the prior on x_0.
    def update_predict(carry, inputs):
        x_prior, P_prior, code = carry
        y, u, t, available = inputs

        def apply_update():
            sigma_obs, Wm_obs, Wc_obs = _sigma_points(
                x_prior, P_prior, alpha, beta, kappa
            )
            y_sigma = jax.vmap(lambda s: h(t, s, u))(sigma_obs)
            if y_sigma.shape != (sigma_obs.shape[0], y.shape[0]):
                raise ValueError("The observation map must return a vector matching y")
            y_pred = jnp.einsum("i,ij->j", Wm_obs, y_sigma)
            y_centered = y_sigma - y_pred
            S = _symmetrize(_weighted_covariance(y_centered, Wc_obs) + R)
            Pxy = _cross_covariance(sigma_obs - x_prior, y_centered, Wc_obs)
            observed = jnp.where(available, y, jnp.zeros_like(y))
            innov = observed - y_pred
            # Reuse the innovation factor for gain and likelihood. The shared
            # triangular solve whitens both cross covariance and residual.
            L = jnp.linalg.cholesky(S)
            whitened = solve_triangular(
                L, jnp.concatenate([Pxy.T, innov[:, None]], axis=1), lower=True
            )
            K = solve_triangular(L.T, whitened[:, :-1], lower=False).T
            x_post = x_prior + K @ innov
            P_post = _symmetrize(P_prior - whitened[:, :-1].T @ whitened[:, :-1])
            logdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
            normalizer = y.shape[0] * jnp.log(jnp.asarray(2 * jnp.pi, dtype=S.dtype))
            log_likelihood = -0.5 * (
                jnp.sum(whitened[:, -1] ** 2) + logdet + normalizer
            )
            return x_post, P_post, innov, y_pred, S, log_likelihood

        def skip_update():
            return (
                x_prior,
                P_prior,
                jnp.zeros_like(y),
                jnp.zeros_like(y),
                jnp.zeros_like(R),
                jnp.zeros((), dtype=x_prior.dtype),
            )

        x_post, P_post, innov, y_pred, S, log_likelihood = jax.lax.cond(
            available, apply_update, skip_update
        )

        # Predict: unscented transform through f from the posterior.
        sigma, Wm, Wc = _sigma_points(x_post, P_post, alpha, beta, kappa)

        def transition(state):
            if callable(getattr(model_or_f, "step", None)):
                step = model_or_f.step(t, state, u)
                value = jnp.where(
                    step.successful, step.x, jnp.full_like(step.x, jnp.nan)
                )
                return value, step.status
            value = f(t, state, u)
            return value, jnp.where(jnp.all(jnp.isfinite(value)), 0, 3)

        sigma_pred, codes = jax.vmap(transition)(sigma)
        next_code = codes[jnp.argmax(codes != 0)]
        code = jnp.where(code != 0, code, next_code)
        x_next = jnp.einsum("i,ij->j", Wm, sigma_pred)
        sigma_pred_centered = sigma_pred - x_next
        P_next = _weighted_covariance(sigma_pred_centered, Wc) + Q
        P_next = _symmetrize(P_next)
        P_cross = _cross_covariance(sigma - x_post, sigma_pred_centered, Wc)

        return (x_next, P_next, code), (
            x_post,
            P_post,
            innov,
            y_pred,
            S,
            log_likelihood,
            x_next,
            P_next,
            P_cross,
        )

    (
        (_, _, code),
        (
            x_hats,
            Ps,
            innovations,
            predicted_measurements,
            innovation_covariances,
            log_likelihood_terms,
            predicted_state_means,
            predicted_state_covariances,
            transition_cross_covariances,
        ),
    ) = jax.lax.scan(
        update_predict,
        (x0, _symmetrize(P0), input_status(P0, Q, R)),
        (ys, us, ts, flags),
    )
    result = UKFResult(
        xs=x_hats,
        Ps=Ps,
        innovations=innovations,
        has_measurement=flags,
        predicted_measurements=predicted_measurements,
        innovation_covariances=innovation_covariances,
        log_likelihood_terms=log_likelihood_terms,
        predicted_state_means=predicted_state_means,
        predicted_state_covariances=predicted_state_covariances,
        transition_cross_covariances=transition_cross_covariances,
        solver_status=code,
    )
    return checked(result, throw, "UKF filter failed; inspect status")


def uks(
    result: UKFResult,
    *,
    throw: bool = False,
) -> RTSResult:
    """Run an unscented Rauch-Tung-Striebel-style smoother.

    Uses the predicted means, covariances and transition cross-covariances
    retained by ukf. No model evaluation or repeated sigma-point transform is
    needed. Supports jit, vmap, JVP and VJP. Empty passes return empty results.
    """
    x_hats = result.xs
    Ps = result.Ps
    if x_hats.shape[0] == 0:
        return checked(
            RTSResult(xs=x_hats, Ps=Ps, solver_status=result.status),
            throw,
            "UKS smoother failed; inspect status",
        )

    def smooth_step(carry, inputs):
        x_s, P_s = carry
        x_f, P_f, x_pred, P_pred, P_cross = inputs
        G = jnp.linalg.solve(P_pred.T, P_cross.T).T
        x_s_new = x_f + G @ (x_s - x_pred)
        P_s_new = _symmetrize(P_f + G @ (P_s - P_pred) @ G.T)
        return (x_s_new, P_s_new), (x_s_new, P_s_new)

    init = (x_hats[-1], Ps[-1])
    _, (x_smooth_rev, P_smooth_rev) = jax.lax.scan(
        smooth_step,
        init,
        (
            x_hats[:-1][::-1],
            Ps[:-1][::-1],
            result.predicted_state_means[:-1][::-1],
            result.predicted_state_covariances[:-1][::-1],
            result.transition_cross_covariances[:-1][::-1],
        ),
    )
    x_smooth = jnp.concatenate([x_smooth_rev[::-1], x_hats[-1:]], axis=0)
    P_smooth = jnp.concatenate([P_smooth_rev[::-1], Ps[-1:]], axis=0)
    smoothed = RTSResult(xs=x_smooth, Ps=P_smooth, solver_status=result.status)
    return checked(smoothed, throw, "UKS smoother failed; inspect status")
