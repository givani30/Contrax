# Getting Started

This page is the shortest path from zero context to a working Contrax
workflow.

## Install

For local development:

```bash
uv sync
```

Contrax does not enable float64 globally on import. If you plan to use
precision-sensitive paths such as `c2d()`, `dare()`, `care()`, or continuous
Gramian helpers, opt into float64 explicitly:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## First Working Example

Start with a small discrete LTI system and design an LQR controller:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
C = jnp.eye(2)
D = jnp.zeros((2, 1))

sys = cx.dss(A, B, C, D, dt=0.05)
result = cx.lqr(sys, jnp.eye(2), jnp.array([[1.0]]))
ts, xs, ys = cx.simulate(
    sys,
    jnp.array([1.0, 0.0]),
    lambda t, x: -result.K @ x,
    num_steps=60,
)
```

At that point you already have:

- `result.K` for the state-feedback gain
- `result.S` for the Riccati solution
- `result.poles` for the closed-loop poles
- `result.residual_norm` for a solver-quality diagnostic
- `xs` and `ys` for the closed-loop trajectory

If you want the result fields defined once in a stable place, see
[Types](api/types.md).

## Read The Example As A Control Loop

The first example is small, but it already has the whole discrete closed-loop
shape:

$$
x_{k+1} = A x_k + B u_k, \qquad
u_k = -K x_k, \qquad
y_k = C x_k + D u_k
$$

In Contrax terms, that means:

- `cx.dss(...)` defines the plant
- `cx.lqr(...)` computes the feedback gain `K`
- `cx.simulate(...)` applies the policy and returns the trajectory

## What To Check

On unfamiliar systems, do not stop at “the code ran”. Check:

- whether the closed-loop poles are where you expect
- whether the Riccati residual is acceptably small for the solver path
- whether the state trajectory behaves plausibly

Discrete LQR is the clearest place to start, especially if you want gradients
through the design-and-simulate loop. Continuous `care()` and continuous
`simulate()` are also available, but the discrete path is still the most
benchmark-complete slice.

## Best Next Step

Choose the next page based on what you want to do:

- [Differentiable LQR](tutorials/differentiable-lqr.md) for optimization
  through controller design
- [Linearize, LQR, simulate](tutorials/linearize-lqr-simulate.md) for the
  main nonlinear-to-discrete control path
- [Continuous LQR](tutorials/continuous-lqr.md) for the continuous-time path
- [Kalman filtering](tutorials/kalman-filtering.md) for the estimation side of
  the library
- [Tune LQR weights with gradients](how-to/tune-lqr-with-gradients.md) for a
  task-oriented recipe
- [JAX-native workflows](examples/jax-native-workflows.md) for broader pattern
  coverage

If you prefer to browse by namespace, the API reference is organized as
[Systems](api/systems.md), [Control](api/control.md),
[Simulation](api/simulation.md), [Estimation](api/estimation.md),
[Analysis](api/analysis.md), and [Types](api/types.md).
