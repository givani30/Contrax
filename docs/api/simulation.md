# Simulation

Simulation is the public namespace for open-loop, closed-loop, and fixed-horizon
trajectory generation in Contrax.

The namespace includes:

- `lsim()` for open-loop discrete LTI simulation
- `simulate()` for closed-loop discrete and continuous simulation
- `rollout()` for fixed-horizon nonlinear transitions without a system object
- `sample_system()` for solver-backed continuous-to-discrete nonlinear model
  construction
- `foh_inputs()` for first-order-hold endpoint pairing on sampled inputs
- `step_response()`, `impulse_response()`, and `initial_response()` for
  standard response views
- `as_ode_term()` as a lower-level bridge into direct Diffrax use

## Minimal Example

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

sys = cx.dss(
    jnp.array([[1.0, 0.05], [0.0, 1.0]]),
    jnp.array([[0.0], [0.05]]),
    jnp.eye(2),
    jnp.zeros((2, 1)),
    dt=0.05,
)
K = cx.lqr(sys, jnp.eye(2), jnp.array([[1.0]])).K
ts, xs, ys = cx.simulate(
    sys,
    jnp.array([1.0, 0.0]),
    lambda t, x: -K @ x,
    num_steps=80,
)
```

## Horizon Conventions

`simulate()` is explicit about the horizon type:

- use `num_steps=...` for discrete systems
- use `duration=...` for continuous systems

For returned trajectories:

- discrete `simulate()` returns `xs` including `x0`
- continuous `simulate()` returns samples on a fixed save grid
- `lsim()` returns `(ts, xs, ys)` for an open-loop input sequence
- response helpers return `(ts, ys)`

Use `rollout()` when there is no system object and you simply want to scan a
transition function over a fixed input sequence.

Use `sample_system()` when the underlying plant model is continuous-time but
the estimation or control workflow lives on a discrete grid. The returned
object is a discrete [`NonlinearSystem`][contrax.systems.NonlinearSystem], so
it plugs directly into `ekf()`, `ukf()`, `simulate(..., num_steps=...)`, and
other discrete-time workflows.

With the default `input_interpolation="zoh"`, each step input has shape `(m,)`
and is held constant across the interval. With
`input_interpolation="foh"`, each step input has shape `(2, m)` and is treated
as the endpoint pair `(u_k, u_{k+1})`; `foh_inputs()` builds that paired
sequence from ordinary sampled inputs.

## Transform Behavior

The main transform contracts are:

- discrete `lsim()`, `simulate()`, and `rollout()` are built around
  fixed-shape JAX scans
- continuous `simulate()` uses Diffrax internally but keeps the public output
  on a fixed save grid
- `rollout()` is intended to compose with `jit`, `vmap`, and `grad`

That makes simulation suitable for both ordinary control scripts and compiled
design loops.

## Numerical Notes

Continuous `simulate()` is intentionally narrow. It exposes the most important
Diffrax escape hatches without turning every simulation call into a solver
configuration exercise.

`step_response()` and `impulse_response()` are analysis helpers. They are good
for inspection and sanity-checking, but they are not the preferred path for
optimization over nonsmooth event timing.

## Related Pages

- [Systems](systems.md) for model construction
- [Control](control.md) for controller design before simulation
- [JAX transform contract](../theory/jax-transform-contract.md) for scan and
  solver behavior under `jit`, `vmap`, and `grad`
- [Linearize, LQR, simulate](../tutorials/linearize-lqr-simulate.md) for the
  main discrete control workflow

::: contrax.simulation
