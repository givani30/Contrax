"""Parameterization helpers for constrained estimation and control design."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _positive_inputs(raw, floor):
    raw, floor_array = jnp.asarray(raw), jnp.asarray(floor)
    if jnp.iscomplexobj(raw) or jnp.iscomplexobj(floor_array):
        raise ValueError("Positive parameterizations require real inputs")
    if floor_array.ndim != 0:
        raise ValueError("The lower bound must be scalar")
    dtype = jnp.result_type(raw, floor, 1.0)
    raw, floor_array = raw.astype(dtype), floor_array.astype(dtype)
    invalid = ~jnp.isfinite(floor_array) | (floor_array < 0)
    if not isinstance(invalid, jax.core.Tracer) and bool(invalid):
        raise ValueError("The lower bound must be finite and nonnegative")
    floor_array = eqx.error_if(
        floor_array, invalid, "The lower bound must be finite and nonnegative"
    )
    return raw, floor_array


def positive_exp(raw: ArrayLike, *, min_value: float = 0.0) -> Array:
    """Map unconstrained values to strictly positive values with `exp`.

    Use it for scalar or diagonal parameters that must stay positive, such as
    noise scales or design weights. Real inputs are promoted to floating point;
    explicit floating precision is preserved. The finite nonnegative scalar
    lower bound is checked in eager and compiled calls. Supports jit, vmap and
    forward/reverse differentiation. Exponentials may overflow or underflow;
    a positive lower bound prevents zero from underflow.
    """
    raw, floor = _positive_inputs(raw, min_value)
    return jnp.exp(raw) + floor


def positive_softplus(raw: ArrayLike, *, min_value: float = 0.0) -> Array:
    """Map real values with softplus and a finite nonnegative scalar floor.

    Shares positive_exp's dtype, transform and lower-bound contract. A zero
    floor permits floating-point underflow to zero for very negative inputs.
    """
    raw, floor = _positive_inputs(raw, min_value)
    return jax.nn.softplus(raw) + floor


def lower_triangular(raw: ArrayLike, *, diagonal: ArrayLike | None = None) -> Array:
    """Return a lower-triangular matrix with an optional overridden diagonal."""
    raw = jnp.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("lower_triangular() expects a square matrix.")
    L = jnp.tril(raw)
    if diagonal is None:
        return L
    diag = jnp.asarray(diagonal)
    if diag.shape != (raw.shape[0],):
        raise ValueError("diagonal must have shape (n,) matching the matrix size.")
    dtype = jnp.result_type(raw, diag)
    L, diag = L.astype(dtype), diag.astype(dtype)
    return L.at[jnp.diag_indices(raw.shape[0])].set(diag)


def spd_from_cholesky_raw(
    raw: ArrayLike,
    *,
    diagonal: str = "softplus",
    min_diagonal: float = 1e-6,
) -> Array:
    """Map an unconstrained square matrix to an SPD matrix via Cholesky factors.

    The lower-triangular part of `raw` is used as the Cholesky factor. The
    diagonal is parameterized with either `softplus` or `exp`, plus
    `min_diagonal`, giving positive definiteness in exact arithmetic. Requires
    a real square matrix and a finite nonnegative scalar floor. Floating-point
    underflow, overflow and conditioning can affect definiteness; no covariance
    repair is applied. Supports jit, vmap and forward/reverse differentiation.
    """
    raw = jnp.asarray(raw)
    if jnp.iscomplexobj(raw):
        raise ValueError("spd_from_cholesky_raw() requires a real matrix")
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("spd_from_cholesky_raw() expects a square matrix.")
    raw_diag = jnp.diag(raw)
    if diagonal == "softplus":
        diag = positive_softplus(raw_diag, min_value=min_diagonal)
    elif diagonal == "exp":
        diag = positive_exp(raw_diag, min_value=min_diagonal)
    else:
        raise ValueError("diagonal must be 'softplus' or 'exp'.")
    L = lower_triangular(raw, diagonal=diag)
    return L @ L.T


def diagonal_spd(
    raw_diagonal: ArrayLike,
    *,
    parameterization: str = "softplus",
    min_diagonal: float = 1e-6,
) -> Array:
    """Build a diagonal SPD matrix from a real vector of raw parameters.

    Uses the same floor, precision and transform contract as positive_exp or
    positive_softplus. Floating-point underflow is possible with a zero floor.
    """
    raw_diagonal = jnp.asarray(raw_diagonal)
    if raw_diagonal.ndim != 1:
        raise ValueError("raw_diagonal must have shape (n,)")
    if parameterization == "softplus":
        diag = positive_softplus(raw_diagonal, min_value=min_diagonal)
    elif parameterization == "exp":
        diag = positive_exp(raw_diagonal, min_value=min_diagonal)
    else:
        raise ValueError("parameterization must be 'softplus' or 'exp'.")
    return jnp.diag(diag)


__all__ = [
    "diagonal_spd",
    "lower_triangular",
    "positive_exp",
    "positive_softplus",
    "spd_from_cholesky_raw",
]
