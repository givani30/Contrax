"""Shared validation of sample periods and static model dimensions."""

import operator

import jax
import jax.numpy as jnp


def dimension(value, name, *, minimum=0):
    if value is None:
        return None
    try:
        if isinstance(value, (bool, jnp.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a static integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be a static integer >= {minimum}")
    return result


def sample_period(dt, dtype=None):
    raw = jnp.asarray(dt)
    if raw.ndim != 0 or jnp.iscomplexobj(raw) or jnp.issubdtype(raw.dtype, jnp.bool_):
        raise ValueError("dt must be a real scalar")
    dt = raw.astype(jnp.result_type(raw, 1.0) if dtype is None else dtype)
    invalid = ~jnp.isfinite(dt) | (dt <= 0)
    if not isinstance(invalid, jax.core.Tracer) and bool(invalid):
        raise ValueError("dt must be finite and positive")
    return dt, invalid
