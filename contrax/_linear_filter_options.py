"""Explicit cached-gain and whitened information-space linear filtering."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from contrax._filter_checks import input_status
from contrax.types import FilterUpdateResult, KalmanResult


def noise_status(R, diagonal):
    if diagonal:
        return jnp.where(jnp.all(jnp.isfinite(R) & (R > 0)), 0, 2)
    return input_status(R)


def information_update(sys, x, P, y, R, u, available):
    # Bayesian conditioning after x=x_prior+L z, z~N(0,I). Whiten observations
    # by diagonal R; I+H.T H is state-sized. Reference: Kailath, Sayed & Hassibi
    # (2000), Linear Estimation, information-form measurement updates.
    def update():
        L = jnp.linalg.cholesky(P)
        H = (sys.C @ L) / jnp.sqrt(R)[:, None]
        factor = jnp.linalg.cholesky(jnp.eye(x.shape[0], dtype=P.dtype) + H.T @ H)
        observed = jnp.where(available, y, jnp.zeros_like(y))
        innovation = observed - sys.C @ x - sys.D @ u
        rhs = H.T @ (innovation / jnp.sqrt(R))
        solved = solve_triangular(
            factor, jnp.concatenate([L.T, rhs[:, None]], axis=1), lower=True
        )
        W, z = solved[:, :-1], solved[:, -1]
        covariance = W.T @ W
        return x + W.T @ z, (covariance + covariance.T) / 2, innovation

    mean, covariance, innovation = jax.lax.cond(
        available, update, lambda: (x, P, jnp.zeros_like(y))
    )
    code = jnp.where(input_status(P) != 0, 2, noise_status(R, True))
    return FilterUpdateResult(mean, covariance, innovation, code)


def _close(a, b, solver):
    a, b = jax.lax.stop_gradient((a, b))
    scale = jnp.maximum(jnp.max(jnp.abs(a), initial=0), jnp.max(jnp.abs(b), initial=0))
    return (
        jnp.all(jnp.isfinite(a))
        & jnp.all(jnp.isfinite(b))
        & (jnp.max(jnp.abs(a - b), initial=0) <= solver.atol + solver.rtol * scale)
    )


def cached_update_data(sys, P, R, solver, available=True):
    design = solver.design
    n, p = sys.C.shape[1], sys.C.shape[0]
    if design.K.shape != (n, p) or design.P.shape != (n, n):
        raise ValueError("Cached gain and covariance dimensions must match the model")
    K, predicted = design.K, design.P
    S = sys.C @ predicted @ sys.C.T + R
    valid = (
        _close(P, predicted, solver)
        & _close(K @ S, predicted @ sys.C.T, solver)
        & jnp.all(jnp.isfinite(jnp.linalg.cholesky(jax.lax.stop_gradient(S))))
        & (input_status(predicted, R) == 0)
        & available
    )
    # Joseph form is computed once for a batch and retains derivatives of the
    # cached design as well as R. The cached P is the *predicted* covariance.
    I_KC = jnp.eye(n, dtype=predicted.dtype) - K @ sys.C
    posterior = I_KC @ predicted @ I_KC.T + K @ R @ K.T
    posterior = (posterior + posterior.T) / 2
    code = jnp.where(design.status != 0, design.status, jnp.where(valid, 0, 2))
    return K, predicted, posterior, code


def cached_update(sys, x, P, y, R, u, available, solver):
    K, _, posterior, code = cached_update_data(sys, P, R, solver, available)

    def update():
        observed = jnp.where(available, y, jnp.zeros_like(y))
        innovation = observed - sys.C @ x - sys.D @ u
        return x + K @ innovation, posterior, innovation

    mean, covariance, innovation = jax.lax.cond(
        available, update, lambda: (x, P, jnp.zeros_like(y))
    )
    return FilterUpdateResult(mean, covariance, innovation, code)


def cached_filter(sys, ys, Q, R, us, x0, P0, flags, solver):
    K, predicted, posterior, code = cached_update_data(
        sys, P0, R, solver, jnp.all(flags)
    )
    closed = sys.A @ (jnp.eye(P0.shape[0], dtype=P0.dtype) - K @ sys.C)
    stable = jnp.all(jnp.abs(jnp.linalg.eigvals(jax.lax.stop_gradient(closed))) < 1)
    valid = _close(sys.A @ posterior @ sys.A.T + Q, predicted, solver) & stable
    code = jnp.where(code != 0, code, jnp.where(valid & (input_status(Q) == 0), 0, 2))

    def step(prior, inputs):
        y, u, available, k = inputs

        def update():
            observed = jnp.where(available, y, jnp.zeros_like(y))
            innovation = observed - sys.C @ prior - sys.D @ u
            return prior + K @ innovation, innovation

        state, innovation = jax.lax.cond(
            available, update, lambda: (prior, jnp.zeros_like(y))
        )
        next_state = jax.lax.cond(
            k + 1 < ys.shape[0], lambda: sys.A @ state + sys.B @ u, lambda: state
        )
        return next_state, (state, innovation)

    _, (xs, innovations) = jax.lax.scan(
        step, x0, (ys, us, flags, jnp.arange(ys.shape[0]))
    )
    Ps = jnp.where(flags[:, None, None], posterior[None], predicted[None])
    return KalmanResult(xs, Ps, innovations, flags, solver_status=code)
