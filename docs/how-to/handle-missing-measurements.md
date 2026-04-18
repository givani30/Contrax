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


def f(x, u):
    return jnp.array([0.8 * x[0] + u[0]])


def h(x):
    return jnp.array([x[0] ** 2])


@jax.jit
def one_step(x, P, y, u, has_measurement):
    return cx.ekf_step(
        f,
        Q,
        R,
        y,
        u,
        x,
        P,
        observation=h,
        has_measurement=has_measurement,
    )


x = jnp.array([1.0])
P = jnp.array([[0.5]])
y = jnp.array([10.0])       # dummy value with the right shape
u = jnp.array([0.0])

x_next, P_next, innovation = one_step(x, P, y, u, jnp.array(False))
```

## What Happens

When `has_measurement=False`, Contrax skips the update and returns:

- the predicted state
- the predicted covariance
- a zero innovation with the same shape as `y`

This works for compiled loops because the branching happens inside JAX, not in
Python control flow.

The same pattern works with `kalman_step()` on linear systems.

## Preconditions And Limits

- Pass a dummy measurement with the correct shape even when the sample is
  missing.
- This recipe handles missing samples, not irregular redefinition of the sample
  time. Irregular timing usually needs an explicit model for the new horizon.

## Check

Verify that on missing-measurement steps:

- the returned innovation is zero
- the output matches the pure prediction from `ekf_predict()` or
  `kalman_predict()`

## Related Pages

- [Estimation API](../api/estimation.md)
- [Estimation pipelines](../theory/estimation-pipelines.md)
