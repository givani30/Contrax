"""JAX-native covariance and result checks, separate from filter arithmetic."""

import equinox as eqx
import jax
import jax.numpy as jnp


def covariance_valid(matrix):
    matrix = jax.lax.stop_gradient(matrix)
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    safe = jnp.where(jnp.isfinite(matrix), matrix, 0)
    scale = jnp.max(jnp.abs(safe), axis=(-2, -1), initial=0)
    tolerance = 32 * jnp.finfo(matrix.dtype).eps * matrix.shape[-1] * scale
    symmetric = (
        jnp.max(jnp.abs(safe - jnp.swapaxes(safe, -1, -2)), axis=(-2, -1), initial=0)
        <= tolerance
    )
    eigenvalues = jnp.linalg.eigvalsh((safe + jnp.swapaxes(safe, -1, -2)) / 2)
    return (
        finite
        & symmetric
        & (jnp.min(eigenvalues, axis=-1, initial=jnp.inf) >= -tolerance)
    )


def input_status(*covariances):
    valid = jnp.asarray(True)
    for covariance in covariances:
        valid = valid & covariance_valid(covariance)
    return jnp.where(valid, 0, 2)


def state_status(x, P, solver_status, *, trajectory=False):
    finite = jnp.all(jnp.isfinite(x), axis=-1)
    valid = covariance_valid(P)
    if trajectory:
        finite, valid = jnp.all(finite, axis=-1), jnp.all(valid, axis=-1)
    return jnp.where(solver_status != 0, solver_status, jnp.where(finite & valid, 0, 3))


def checked(result, throw, message):
    return eqx.error_if(result, ~result.successful, message) if throw else result


def prepare_state(x, P, noise, *, measurement=None, diagonal_noise=False):
    arrays = tuple(jnp.asarray(a) for a in (x, P, noise))
    if any(jnp.iscomplexobj(a) for a in arrays):
        raise ValueError("Filters require real arrays")
    dtype = jnp.result_type(*arrays, 1.0)
    x, P, noise = (a.astype(dtype) for a in arrays)
    if x.ndim != 1 or P.shape != (x.shape[0], x.shape[0]):
        raise ValueError("x and P must have shapes (n,) and (n, n)")
    if measurement is not None and len(jnp.shape(measurement)) != 1:
        raise ValueError("y must have shape (p,)")
    dim = x.shape[0] if measurement is None else jnp.shape(measurement)[0]
    expected = (dim,) if diagonal_noise else (dim, dim)
    if noise.shape != expected:
        raise ValueError(
            "Noise covariance dimensions must match the state or observation"
        )
    return x, P, noise
