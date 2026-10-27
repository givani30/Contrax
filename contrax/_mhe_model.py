"""Model adapters and fixed-window shape validation for MHE."""

import jax
import jax.numpy as jnp

from contrax.core import ContLTI, DiscLTI
from contrax.nonlinear import _is_system_model, _system_dt


def validate_window(xs, us, ys, x_prior, P_prior, Q_noise, R_noise):
    if xs.ndim != 2 or xs.shape[0] < 1 or xs.shape[1] < 1:
        raise ValueError("xs must have shape (N, n), with N >= 1 and n >= 1")
    N, n = xs.shape
    if us.ndim < 2 or us.shape[0] != N - 1 or ys.ndim != 2 or ys.shape[0] != N:
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
    input_shape = getattr(model, "_input_shape", None)
    input_dim = (
        model.B.shape[1]
        if isinstance(model, DiscLTI)
        else getattr(model, "input_dim", None)
    )
    if input_shape is not None:
        if us.ndim != len(input_shape) + 1 or us.shape[1:-1] != input_shape[:-1]:
            raise ValueError("us must match the sampled model input structure")
        if input_dim is not None and us.shape[1:] != input_shape:
            raise ValueError("us must match the model input dimensions")
    elif input_dim is not None and us.shape[1:] != (input_dim,):
        raise ValueError("us must match the model input dimensions")
    if measurement_us is None:
        # N-1 controls define transitions. For observations, hold the last
        # input endpoint at the terminal sample; a one-state window uses zero.
        last = us[-1:] if N > 1 else jnp.zeros((1, *us.shape[1:]), dtype=us.dtype)
        if input_shape is not None and input_shape[0] == 2:
            last = jnp.repeat(last[:, -1:, :], 2, axis=1)
        measurement_us = jnp.concatenate([us, last], axis=0)
    if measurement_us.shape != (N, *us.shape[1:]):
        raise ValueError(
            "measurement_us must have N inputs with the same trailing shape as us"
        )

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
