"""contrax._ukf — Unscented Kalman filter."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._kalman import _symmetrize
from contrax.nonlinear import (
    DynamicsLike,
    ObservationFn,
    _coerce_dynamics,
    _coerce_observation,
    _is_system_model,
    _system_dt,
)
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
    jitter = jnp.asarray(1e-9, dtype=cov.dtype)
    Wm, Wc, c = _ukf_weights(n, alpha, beta, kappa, cov.dtype)
    chol = jnp.linalg.cholesky(c * cov + jitter * jnp.eye(n, dtype=cov.dtype))
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


def _gaussian_log_likelihood(innovation: Array, covariance: Array) -> Array:
    innovation = innovation[:, None]
    solved = jnp.linalg.solve(covariance, innovation)
    mahalanobis = (innovation.T @ solved).squeeze()
    sign, logdet = jnp.linalg.slogdet(covariance)
    dim = covariance.shape[0]
    normalizer = dim * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=covariance.dtype))
    return jnp.where(
        sign > 0,
        -0.5 * (mahalanobis + logdet + normalizer),
        jnp.asarray(-jnp.inf, dtype=covariance.dtype),
    )


def ukf(
    model_or_f: DynamicsLike,
    Q_noise: Array,
    R_noise: Array,
    ys: Array,
    us: Array,
    x0: Array,
    P0: Array,
    *,
    observation: ObservationFn | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> UKFResult:
    """Run an unscented Kalman filter for nonlinear dynamics and observations.

    Uses a deterministic sigma-point transform instead of local Jacobians to
    propagate means and covariances through the nonlinear transition and
    observation models.

    This is a baseline nonlinear recursive estimation path that is often more
    robust than `ekf()` when the observation model is strongly curved. The
    current implementation assumes additive process and measurement noise and
    keeps the public API deliberately small.

    Args:
        model_or_f: Nonlinear system model or dynamics function.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        ys: Measurement sequence. Shape: `(T, p)`.
        us: Input sequence. Shape: `(T, m)`.
        x0: Initial mean. Shape: `(n,)`.
        P0: Initial covariance. Shape: `(n, n)`.
        observation: Observation function `h(x)` for plain dynamics callables.
            Omit this when passing a system model with its own output map.
        alpha: Sigma-point spread parameter. The default is intentionally wider
            than the very small textbook setting so the baseline implementation
            stays numerically well-behaved on the current small-system focus.
        beta: Prior-distribution parameter. `2.0` is standard for Gaussian
            priors.
        kappa: Secondary sigma-point scaling parameter.

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
        ...     f,
        ...     Q_noise=1e-3 * jnp.eye(2),
        ...     R_noise=1e-2 * jnp.eye(1),
        ...     ys=jnp.zeros((10, 1)),
        ...     us=jnp.zeros((10, 1)),
        ...     x0=jnp.ones(2),
        ...     P0=jnp.eye(2),
        ...     observation=h,
        ... )
    """

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
        ts = ts * dt

    # Update-then-predict: x0/P0 is the prior on x_0.
    def update_predict(carry, inputs):
        x_prior, P_prior = carry
        y, u, t = inputs

        # Update: unscented transform through h using the current prior.
        sigma_obs, Wm_obs, Wc_obs = _sigma_points(x_prior, P_prior, alpha, beta, kappa)
        y_sigma = jax.vmap(lambda s: h(t, s, u))(sigma_obs)
        y_pred = jnp.einsum("i,ij->j", Wm_obs, y_sigma)
        y_centered = y_sigma - y_pred
        S = _weighted_covariance(y_centered, Wc_obs) + R_noise
        S = _symmetrize(S)
        x_centered = sigma_obs - x_prior
        Pxy = _cross_covariance(x_centered, y_centered, Wc_obs)
        K = jnp.linalg.solve(S.T, Pxy.T).T
        innov = y - y_pred
        x_post = x_prior + K @ innov
        P_post = _symmetrize(P_prior - K @ S @ K.T)
        log_likelihood = _gaussian_log_likelihood(innov, S)

        # Predict: unscented transform through f from the posterior.
        sigma, Wm, Wc = _sigma_points(x_post, P_post, alpha, beta, kappa)
        sigma_pred = jax.vmap(lambda s: f(t, s, u))(sigma)
        x_next = jnp.einsum("i,ij->j", Wm, sigma_pred)
        sigma_pred_centered = sigma_pred - x_next
        P_next = _weighted_covariance(sigma_pred_centered, Wc) + Q_noise
        P_next = _symmetrize(P_next)
        P_cross = _cross_covariance(sigma - x_post, sigma_pred_centered, Wc)

        return (x_next, P_next), (
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
        _,
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
    ) = jax.lax.scan(update_predict, (x0, _symmetrize(P0)), (ys, us, ts))
    return UKFResult(
        x_hat=x_hats,
        P=Ps,
        innovations=innovations,
        predicted_measurements=predicted_measurements,
        innovation_covariances=innovation_covariances,
        log_likelihood_terms=log_likelihood_terms,
        predicted_state_means=predicted_state_means,
        predicted_state_covariances=predicted_state_covariances,
        transition_cross_covariances=transition_cross_covariances,
    )


def uks(
    model_or_f: DynamicsLike,
    result: UKFResult,
    Q_noise: Array,
    us: Array,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> RTSResult:
    """Run an unscented Rauch-Tung-Striebel-style smoother.

    Applies a backward smoothing pass to the output of `ukf()` using the same
    nonlinear dynamics model, process noise covariance, and input sequence.
    """
    del model_or_f, Q_noise, alpha, beta, kappa
    x_hats = result.x_hat
    Ps = result.P
    if us.shape[0] != x_hats.shape[0]:
        raise ValueError(
            "uks() requires us to have the same leading length as the filtered result."
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
    return RTSResult(x_smooth=x_smooth, P_smooth=P_smooth)
