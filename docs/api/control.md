# Control

Control is the public namespace for controller design and Riccati-backed
feedback synthesis in Contrax.

The namespace includes:

- `dare()` and `care()` for algebraic Riccati solves
- `lqr()` and `lqi()` for regulator design
- `augment_integrator()` for explicit integral-state augmentation
- `place()` for design-time pole placement
- `state_feedback()` for applying a state-feedback gain to an LTI model

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
result = cx.lqr(sys, jnp.eye(2), jnp.array([[1.0]]))
closed_loop = cx.state_feedback(sys, result.K)
```

<figure class="contrax-figure">
  <img src="/assets/images/lqr-riccati-workflow.svg"
       alt="Workflow from plant model and quadratic weights through dare or care, gain recovery, and closed-loop validation" />
  <figcaption>
    <strong>The main control path:</strong> combine a linear model with
    quadratic weights, solve the appropriate Riccati equation, recover the
    stabilizing gain, then inspect the closed-loop poles and residual.
  </figcaption>
</figure>

## Conventions

- `Q`: state cost matrix with shape `(n, n)`
- `R`: input cost matrix with shape `(m, m)`
- [`LQRResult.K`](types.md#contrax.types.LQRResult.K): state-feedback gain
  with shape `(m, n)`
- [`LQRResult.S`](types.md#contrax.types.LQRResult.S): Riccati solution with
  shape `(n, n)`
- [`LQRResult.poles`](types.md#contrax.types.LQRResult.poles): closed-loop
  eigenvalues
- [`LQRResult.residual_norm`](types.md#contrax.types.LQRResult.residual_norm):
  JAX scalar Riccati residual diagnostic

For `lqi()`, the returned gain acts on the augmented state `[x; z]`, where `z`
is the integral state added by `augment_integrator()`.

## Control Equations

Contrax's control surface is centered on state-feedback design for linear
systems.

For the discrete-time model

$$
x_{k+1} = A x_k + B u_k
$$

the infinite-horizon quadratic objective used by `dare()`, `lqr()`, and
discrete `lqi()` is

$$
J_d = \sum_{k=0}^{\infty} \left(x_k^\top Q x_k + u_k^\top R u_k\right)
$$

with state-feedback law

$$
u_k = -K x_k
$$

For the continuous-time model

$$
\dot{x} = A x + B u
$$

the corresponding continuous objective behind `care()` and continuous `lqr()`
is

$$
J_c = \int_0^\infty \left(x(t)^\top Q x(t) + u(t)^\top R u(t)\right)\,dt
$$

with feedback law

$$
u(t) = -K x(t)
$$

### Riccati Equations

The stabilizing discrete Riccati equation is

$$
A^\top S A - S - A^\top S B \left(R + B^\top S B\right)^{-1} B^\top S A + Q = 0
$$

and the continuous Riccati equation is

$$
A^\top S + S A - S B R^{-1} B^\top S + Q = 0
$$

Given the stabilizing solution `S`, the corresponding gains are

$$
K_d = \left(R + B^\top S B\right)^{-1} B^\top S A
$$

$$
K_c = R^{-1} B^\top S
$$

### Closed-Loop Models

`state_feedback()` applies the gain directly to the plant matrices.

For discrete systems,

$$
x_{k+1} = (A - B K) x_k
$$

and for continuous systems,

$$
\dot{x} = (A - B K) x
$$

That is the model whose poles appear in
[`LQRResult.poles`](types.md#contrax.types.LQRResult.poles).

<figure class="contrax-figure">
  <img src="/assets/images/output-feedback-architecture.svg"
       alt="Output-feedback architecture with controller, plant, measurement, and estimator feeding a state estimate back to the controller" />
  <figcaption>
    <strong>Where state feedback fits:</strong> `state_feedback()` handles the
    plant-side algebra, while estimation supplies the state estimate when the
    full state is not measured directly.
  </figcaption>
</figure>

### Integral Augmentation

`augment_integrator()` and `lqi()` introduce an integral state `z` driven by an
output-tracking error. At the level of the augmented state

$$
\bar{x} = \begin{bmatrix} x \\ z \end{bmatrix}
$$

the returned gain acts as

$$
u = -\bar{K}\,\bar{x}
$$

with $\bar{K}$ partitioned over the physical and integral states.

## Solver Maturity

Contrax is explicit about solver maturity:

- `dare()` is the most mature Riccati path
- `care()` is supported and validated, but less benchmarked than `dare()`
- `place()` is suitable as a design-time helper, not as a hardened large-scale
  assignment solver

If the numerical context matters to your workflow, read
[Riccati solvers](../theory/riccati-solvers.md) before treating all paths as
equally mature.

## Transform Behavior

`dare()`, `care()`, and `lqr()` are designed to participate in compiled JAX
workflows.

The key contracts are:

- `lqr()` dispatches on [`DiscLTI`][contrax.systems.DiscLTI] vs
  [`ContLTI`][contrax.systems.ContLTI] outside traced runtime values
- the Riccati paths use custom or implicit VJPs rather than blindly unrolling
  the forward solve in reverse mode
- `state_feedback()` is just state-space algebra on the model matrices

That makes controller design inside a JIT-compiled objective a first-class use
case rather than an accidental one.

## Numerical Notes

Contrax does not enable float64 globally on import. Riccati solves require
`jax_enable_x64=True`.

`state_feedback(sys, K)` is specifically about applying a state-feedback gain
to an LTI model. It is not the general interconnection operator for arbitrary
controller/plant block diagrams.

For integral-action designs, `augment_integrator()` exposes the augmented model
directly and `lqi()` is the thin convenience wrapper on top.

## Related Pages

- [Systems](systems.md) for model construction before design
- [Simulation](simulation.md) for validating the designed controller
- [Types](types.md) for [`LQRResult`][contrax.types.LQRResult]
- [Riccati solvers](../theory/riccati-solvers.md) for solver details
- [Tune LQR weights with gradients](../how-to/tune-lqr-with-gradients.md) for
  a task-oriented recipe

::: contrax.control
