"""Parameterization helpers for constrained estimation and control design."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def positive_exp(raw: ArrayLike, *, min_value: float = 0.0) -> Array:
    """Map unconstrained values to strictly positive values with `exp`.

    This is useful when optimizing scalar or diagonal quantities that must stay
    positive, such as noise scales or diagonal design weights.
    """
    raw = jnp.asarray(raw)
    return jnp.exp(raw) + jnp.asarray(min_value, dtype=raw.dtype)


def positive_softplus(raw: ArrayLike, *, min_value: float = 0.0) -> Array:
    """Map unconstrained values to positive values with `softplus`."""
    raw = jnp.asarray(raw)
    return jax.nn.softplus(raw) + jnp.asarray(min_value, dtype=raw.dtype)


def lower_triangular(raw: ArrayLike, *, diagonal: ArrayLike | None = None) -> Array:
    """Return a lower-triangular matrix with an optional overridden diagonal."""
    raw = jnp.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("lower_triangular() expects a square matrix.")
    L = jnp.tril(raw)
    if diagonal is None:
        return L
    diag = jnp.asarray(diagonal, dtype=raw.dtype)
    if diag.shape != (raw.shape[0],):
        raise ValueError("diagonal must have shape (n,) matching the matrix size.")
    return L - jnp.diag(jnp.diag(L)) + jnp.diag(diag)


def spd_from_cholesky_raw(
    raw: ArrayLike,
    *,
    diagonal: str = "softplus",
    min_diagonal: float = 1e-6,
) -> Array:
    """Map an unconstrained square matrix to an SPD matrix via Cholesky factors.

    The lower-triangular part of `raw` is used as the Cholesky factor. The
    diagonal is parameterized with either `softplus` or `exp`, plus
    `min_diagonal`, so the result stays strictly positive definite.
    """
    raw = jnp.asarray(raw)
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
    """Build a diagonal SPD matrix from unconstrained diagonal parameters."""
    raw_diagonal = jnp.asarray(raw_diagonal)
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
