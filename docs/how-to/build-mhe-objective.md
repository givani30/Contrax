# How To Build An MHE Objective

This guide shows how to construct a fixed-window moving-horizon-estimation
objective before choosing an optimizer.

## Complete Working Code

```python
import jax
import jax.numpy as jnp
import contrax as cx


def f(x, u, params):
    return jnp.array([params["a"] * x[0] + u[0] ** 2])


def h(x, params):
    return jnp.array([params["c"] * x[0]])


params = {"a": 0.5, "c": 2.0}
us = jnp.array([[0.0], [2.0], [1.0]])
xs = cx.rollout(f, jnp.array([1.0]), us, params)
ys = jax.vmap(lambda x: h(x, params))(xs)

cost = cx.mhe_objective(
    f,
    h,
    xs=xs,
    us=us,
    ys=ys,
    x_prior=xs[0],
    P_prior=jnp.eye(1),
    Q_noise=jnp.eye(1),
    R_noise=jnp.eye(1),
    params=params,
)
```

## Alignment Rules

The important shape rule is:

- `xs`: `(T + 1, n)`
- `us`: `(T, m)`
- `ys`: `(T + 1, p)`

That alignment matches:

- one arrival state at the beginning of the window
- one process residual per transition
- one measurement residual per state in the window

## Why This Is Useful

`mhe_objective()` is the pure cost-function layer of Contrax’s optimization-
based estimation story. It lets you:

- keep the horizon fixed-shape
- add custom soft costs explicitly
- choose your optimizer separately from the model definition

Two small helpers are useful in rolling-window workflows:

- `mhe_warm_start(xs, ...)` shifts a previous trajectory guess forward by one
  step, optionally propagating the final guess with a transition model
- `soft_quadratic_penalty(residuals, weight)` builds a weighted quadratic soft
  cost that you can add inside `extra_cost`

If you want the built-in thin solver wrapper, move from `mhe_objective()` to
`mhe()`.

## Check

For an exact trajectory and measurement sequence, the objective should go to
zero up to numerical tolerance.

## Related Pages

- [Estimation API](../api/estimation.md)
- [Estimation pipelines](../theory/estimation-pipelines.md)
