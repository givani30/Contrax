"""Shared Riccati algebra, validation and result construction."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from contrax.solvers import SolverStatus
from contrax.types import LQRResult


def _symmetrize(X: Array) -> Array:
    return (X + X.T) / 2


def _lqr_gain(A: Array, B: Array, S: Array, R: Array) -> Array:
    BtS = B.T @ S
    return jnp.linalg.solve(R + BtS @ B, BtS @ A)


def _dare_residual(S: Array, A: Array, B: Array, Q: Array, R: Array) -> Array:
    K = _lqr_gain(A, B, S, R)
    return S - Q - A.T @ S @ A + A.T @ S @ B @ K


def _care_residual(S: Array, A: Array, B: Array, Q: Array, R: Array) -> Array:
    R_inv_BT = jnp.linalg.solve(R, B.T)
    return _symmetrize(Q + A.T @ S + S @ A - S @ B @ R_inv_BT @ S)


def _validate_matrices(A, B, Q, R):
    A, B, Q, R = (jnp.asarray(x) for x in (A, B, Q, R))
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n or B.shape[1] == 0:
        raise ValueError("B must have shape (n, m), with m > 0")
    if Q.shape != (n, n) or R.shape != (B.shape[1], B.shape[1]):
        raise ValueError("Q and R must have shapes (n, n) and (m, m)")
    if any(jnp.iscomplexobj(x) for x in (A, B, Q, R)):
        raise ValueError("Riccati solvers require real matrices")
    dtype = jnp.result_type(A, B, Q, R, 1.0)
    return tuple(x.astype(dtype) for x in (A, B, Q, R))


def _result(A, B, Q, R, S, steps, solver, *, continuous, throw):
    K = jnp.linalg.solve(R, B.T @ S) if continuous else _lqr_gain(A, B, S, R)
    closed_loop = A - B @ K
    # Stability is a nondifferentiable validation decision. Pole objectives use
    # the result's lazy poles property; K/S losses never differentiate this eig.
    poles = jnp.linalg.eigvals(jax.lax.stop_gradient(closed_loop))
    if continuous:
        terms = (Q, A.T @ S, S @ A, -S @ B @ K)
    else:
        terms = (S, -Q, -A.T @ S @ A, A.T @ S @ B @ K)
    residual_norm = jnp.max(jnp.abs(sum(terms)))
    scale = sum(jnp.max(jnp.abs(term)) for term in terms)
    symmetric = (
        jnp.max(jnp.abs(Q - Q.T)) <= solver.atol + solver.rtol * jnp.max(jnp.abs(Q))
    ) & (jnp.max(jnp.abs(R - R.T)) <= solver.atol + solver.rtol * jnp.max(jnp.abs(R)))
    valid = (
        symmetric
        & (jnp.min(jnp.linalg.eigvalsh(_symmetrize(Q))) >= -solver.atol)
        & (jnp.min(jnp.linalg.eigvalsh(_symmetrize(R))) > 0)
    )
    stable = jnp.all(jnp.real(poles) < 0 if continuous else jnp.abs(poles) < 1)
    finite = jnp.all(jnp.isfinite(S)) & jnp.all(jnp.isfinite(K)) & jnp.isfinite(scale)
    good = finite & stable & (residual_norm <= solver.atol + solver.rtol * scale)
    exhausted = False if continuous else steps >= solver.max_steps
    status = jnp.where(
        ~valid,
        int(SolverStatus.invalid_problem),
        jnp.where(
            good,
            int(SolverStatus.successful),
            jnp.where(
                finite & exhausted,
                int(SolverStatus.max_steps_reached),
                int(SolverStatus.numerical_failure),
            ),
        ),
    )
    result = LQRResult(
        K=K,
        S=S,
        closed_loop=closed_loop,
        residual_norm=residual_norm,
        status=status,
        num_steps=steps,
    )
    return (
        eqx.error_if(
            result, status != 0, "Riccati solve failed; inspect status/residual"
        )
        if throw
        else result
    )
