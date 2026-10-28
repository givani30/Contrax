# How To Handle Missing Measurements

This guide shows how to skip measurement updates in a compiled runtime loop
without branching in Python.

## Complete Working Code

```python
import jax
import jax.numpy as jnp
import contrax as cx

Q = jnp.array([[1e-3]])
R = jnp.array([[1e-2]])


def f(t, x, u):
    return jnp.array([0.8 * x[0] + u[0]])


def h(x):
    return jnp.array([x[0] ** 2])


@jax.jit
def one_step(x, P, y, u, has_measurement):
    return cx.ekf_step(f, x, P, y, u=u, Q=Q, R=R, observation=h, has_measurement=has_measurement)


x = jnp.array([1.0])
P = jnp.array([[0.5]])
y = jnp.array([jnp.nan])    # ignored when has_measurement is False
u = jnp.array([0.0])

x_next, P_next, innovation = one_step(x, P, y, u, jnp.array(False))
```

## What Happens

When `has_measurement=False`, the step skips the update and returns:

- the predicted state
- the predicted covariance
- a zero innovation with the same shape as `y`

The measurement flag is a JAX boolean, so the update can be skipped inside a
compiled loop without Python branching.

The same pattern works with `kalman_step()` on linear systems.

## Preconditions And Limits

- Pass a measurement array with the correct shape; missing entries may be NaN.
- Batch `kalman` and `ekf` accept a `(T,)` `has_measurement` mask. Each entry
  skips the corresponding whole observation, including all measurement channels.
- The initial mean and covariance in batch filters are priors for measurement
  zero. Call the online update first, then predict and update for later samples.
- Joseph covariance updates are the default. Select
  `solver=cx.solvers.Kalman(covariance_update="compact")` or
  `cx.solvers.EKF(covariance_update="compact")` explicitly for compact form.
- The mask skips measurements while keeping the model's sample time fixed.
  For irregular timing, provide a model for each new time interval.

## Check

Verify that on missing-measurement steps:

- the returned innovation is zero
- the output matches the pure prediction from `ekf_predict()` or
  `kalman_predict()`

## Related Pages

- [Estimation API](../api/estimation.md)
- [Estimation pipelines](../theory/estimation-pipelines.md)
