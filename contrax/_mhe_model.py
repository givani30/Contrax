"""Model adapters and fixed-window shape validation for MHE."""

import jax
import jax.numpy as jnp

from contrax.core import ContLTI, DiscLTI
from contrax.nonlinear import _is_system_model, _system_dt


def validate_window(xs, us, ys, x_prior, P_prior, Q_noise, R_noise):
    if xs.ndim != 2 or xs.shape[0] < 1 or xs.shape[1] < 1:
        raise ValueError("xs must have shape (N, n), with N >= 1 and n >= 1")
    N, n = xs.shape
    if us.ndim != 2 or us.shape[0] != N - 1 or ys.ndim != 2 or ys.shape[0] != N:
        raise ValueError("MHE requires N states/measurements and N-1 transition inputs")
    if x_prior.shape != (n,) or P_prior.shape != (n, n) or Q_noise.shape != (n, n):
        raise ValueError(
            "prior, arrival and process covariance dimensions must match n"
        )
    if R_noise.shape != (ys.shape[1], ys.shape[1]):
        raise ValueError("R_noise must match the measurement dimension")


def model_predictions(model, observation, xs, us, params, ts, measurement_us):
    N = xs.shape[0]
    if isinstance(model, ContLTI):
        raise TypeError(
            "MHE requires a discrete transition; discretize the model first"
        )
    if ts is None:
        dt = _system_dt(model)
        ts = jnp.arange(N, dtype=xs.dtype) * (
            1 if dt is None else jnp.asarray(dt, dtype=xs.dtype)
        )
    if ts.shape != (N,):
        raise ValueError("ts must have shape (N,)")
    if measurement_us is None:
        # N-1 controls define transitions. For observations, hold the last
        # transition input at the terminal sample; a one-state window uses zero.
        last = us[-1:] if N > 1 else jnp.zeros((1, us.shape[1]), dtype=us.dtype)
        measurement_us = jnp.concatenate([us, last], axis=0)
    if measurement_us.shape != (N, us.shape[1]):
        raise ValueError("measurement_us must have shape (N, m)")

    if isinstance(model, DiscLTI):
        if observation is not None or params is not None:
            raise ValueError(
                "LTI MHE takes model data in the system; omit observation/params"
            )

        def transition(t, x, u):
            return model.A @ x + model.B @ u

        def output(t, x, u):
            return model.C @ x + model.D @ u
    elif _is_system_model(model):
        if _system_dt(model) is None:
            raise ValueError("MHE requires a discrete model; use sample_system first")
        if observation is not None or params is not None:
            raise ValueError(
                "System MHE takes model data in the system; omit observation/params"
            )
        transition, output = model.dynamics, model.output
    else:
        if observation is None:
            raise ValueError("MHE with a callable transition requires observation=")

        def transition(t, x, u):
            return model(x, u) if params is None else model(x, u, params)

        def output(t, x, u):
            return observation(x) if params is None else observation(x, params)

    predicted_xs = jax.vmap(transition)(ts[:-1], xs[:-1], us)
    predicted_ys = jax.vmap(output)(ts, xs, measurement_us)
    return predicted_xs, predicted_ys
