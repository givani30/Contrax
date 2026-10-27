"""Shared array preparation for recursive estimators."""

import jax.numpy as jnp


def zero_input(model, dtype):
    shape = getattr(model, "_input_shape", None)
    if shape is None:
        m = (
            model.B.shape[1]
            if hasattr(model, "B")
            else getattr(model, "input_dim", None)
        )
        shape = (0 if m is None else m,)
    return jnp.zeros(shape, dtype=dtype)


def prepare_filter(
    model,
    ys,
    Q,
    R,
    us,
    x0,
    P0,
    has_measurement,
    *,
    diagonal_R=False,
    extra_arrays=(),
):
    ys, Q, R = (jnp.asarray(a) for a in (ys, Q, R))
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] == 0:
        raise ValueError("Q must have nonempty shape (n, n)")
    n = Q.shape[0]
    if ys.ndim != 2:
        raise ValueError("ys must have shape (T, p)")
    expected_R = (ys.shape[1],) if diagonal_R else (ys.shape[1], ys.shape[1])
    if R.shape != expected_R:
        raise ValueError(
            "R must be a variance vector (p,)"
            if diagonal_R
            else "R must have shape (p, p)"
        )
    arrays = [jnp.asarray(a) for a in (ys, Q, R, us, x0, P0) if a is not None]
    arrays.extend(extra_arrays)
    if hasattr(model, "A"):
        arrays.extend((model.A, model.B, model.C, model.D))
    if any(jnp.iscomplexobj(a) for a in arrays):
        raise ValueError("Filters require real arrays")
    dtype = jnp.result_type(*arrays, 1.0)
    ys, Q, R = (a.astype(dtype) for a in (ys, Q, R))
    x0 = jnp.zeros(n, dtype=dtype) if x0 is None else jnp.asarray(x0, dtype=dtype)
    P0 = jnp.eye(n, dtype=dtype) if P0 is None else jnp.asarray(P0, dtype=dtype)
    if x0.shape != (n,) or P0.shape != (n, n):
        raise ValueError("x0 and P0 must have shapes (n,) and (n, n)")
    model_n = (
        model.A.shape[0] if hasattr(model, "A") else getattr(model, "state_dim", None)
    )
    model_p = (
        model.C.shape[0] if hasattr(model, "C") else getattr(model, "output_dim", None)
    )
    if (model_n is not None and n != model_n) or (
        model_p is not None and ys.shape[1] != model_p
    ):
        raise ValueError("Noise/measurement dimensions must match the model")
    u0 = zero_input(model, dtype)
    us = (
        jnp.zeros((ys.shape[0], *u0.shape), dtype=dtype)
        if us is None
        else jnp.asarray(us, dtype=dtype)
    )
    if us.ndim < 2 or us.shape[0] != ys.shape[0]:
        raise ValueError("us must have the same leading length as ys")
    known_inputs = hasattr(model, "B") or getattr(model, "input_dim", None) is not None
    shape = getattr(model, "_input_shape", None)
    if shape is not None and (
        us.ndim != len(shape) + 1 or us.shape[1:-1] != shape[:-1]
    ):
        raise ValueError(
            "us trailing dimensions must match the sampled input structure"
        )
    if known_inputs and us.shape[1:] != u0.shape:
        raise ValueError("us trailing dimensions must match the model input shape")
    flags = (
        jnp.ones(ys.shape[0], dtype=bool)
        if has_measurement is None
        else jnp.asarray(has_measurement, dtype=bool)
    )
    if flags.shape != (ys.shape[0],):
        raise ValueError("has_measurement must have shape (T,)")
    return ys, Q, R, us, x0, P0, flags
