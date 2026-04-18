# Contrax

Contrax is a JAX-native systems, control, simulation, and estimation library
with MATLAB-familiar names at the API surface and JAX-first behavior
underneath.

The core idea is simple: controller design should live in the same JAX program
as the rest of the model. Linearization, simulation, batching, and
differentiation should compose instead of being split across separate tools.

<div class="contrax-status-strip" markdown="1">

- `dare` is the most mature Riccati solver path in the library
- `care` is supported and validated, but less benchmarked than `dare`
- continuous `simulate()` uses Diffrax behind a narrow Contrax-shaped surface

</div>

## One Shared Loop

<figure class="contrax-figure">
  <img src="/assets/images/control-loop-overview.svg"
       alt="Closed-loop overview showing controller, plant, estimator, and the Contrax workflow around them" />
  <figcaption>
    <strong>The core Contrax idea:</strong> keep the model, controller design,
    simulation, and estimator on one shared state-space story instead of
    splitting them across separate tools.
  </figcaption>
</figure>

## Why Contrax

Contrax is built for workflows such as:

- differentiating through `lqr()` and closed-loop simulation
- linearizing nonlinear dynamics directly into state-space form
- vmapping controller design across operating points
- keeping design and validation inside one compiled objective

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="/assets/images/pendulum_lqr.gif"
       alt="Gradient descent through a differentiable Contrax LQR loop, showing a pendulum trajectory improving over 50 gradient steps"
       style="max-width: 720px; width: 100%; border-radius: 8px;" />
  <p style="color: var(--md-default-fg-color--light); font-size: 0.8rem; margin-top: 0.5rem;">
    50 gradient steps through <code>ss → c2d → lqr → simulate</code>.
    The pendulum settles faster as the cost weights are tuned automatically.
  </p>
</div>

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
SYS = cx.dss(A, B, jnp.eye(2), jnp.zeros((2, 1)), dt=0.05)
X0 = jnp.array([1.0, 0.0])


def closed_loop_cost(log_q_diag, log_r):
    Q = jnp.diag(jnp.exp(log_q_diag))
    R = jnp.exp(log_r)[None, None]
    K = cx.lqr(SYS, Q, R).K
    _, xs, _ = cx.simulate(SYS, X0, lambda t, x: -K @ x, num_steps=80)
    return jnp.sum(xs**2)


objective_and_grad = jax.jit(jax.value_and_grad(closed_loop_cost, argnums=(0, 1)))
cost, grads = objective_and_grad(jnp.zeros(2), jnp.array(0.0))
```

That is the central Contrax story: control primitives that behave like ordinary
JAX building blocks instead of sealed-off toolbox calls.

Contrax is deliberately focused. It is not trying to be a plotting-heavy
controls environment or a full MATLAB clone. It is trying to make the most
useful systems-and-control workflows feel native inside JAX.

## Core Equations

<div class="contrax-equation-grid" markdown="1">

<div class="contrax-equation-card" markdown="1">

### Model Families

Contrax keeps the familiar control-theory objects visible in the docs and API.

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

$$
\dot{x} = f(t, x, u), \qquad y = h(t, x, u)
$$

$$
\dot{x} = \bigl(J - R(x)\bigr)\nabla H(x) + G(x)u
$$

</div>

<div class="contrax-equation-card" markdown="1">

### Design And Estimation Loops

The design and estimation side stays just as explicit.

$$
x_{k+1} = A x_k + B u_k, \qquad
u_k = -K x_k
$$

$$
\hat{x}_k^+ = \hat{x}_k^- + K_k\bigl(y_k - h(\hat{x}_k^-)\bigr), \qquad
\hat{x}_{k+1}^- = f(\hat{x}_k^+, u_k)
$$

</div>

</div>

## Capabilities

<div class="grid cards" markdown="1">

- __Systems__

  `contrax.systems` for state-space models, nonlinear models, linearization,
  and interconnection.

- __Control__

  `contrax.control` for LQR/LQI, Riccati solvers, pole placement, and
  state-feedback helpers.

- __Simulation__

  `contrax.simulation` for `lsim()`, `simulate()`, response helpers, and fixed
  horizon `rollout()`.

- __Estimation__

  `contrax.estimation` for Kalman-family filters, smoothers, and MHE helpers.

- __Analysis__

  `contrax.analysis` for structural checks, transfer evaluation, poles, and
  finite-horizon Gramians.

- __Types__

  `contrax.types` for the public result bundles such as `LQRResult` and
  `KalmanResult`.

- __Interoperability__

  `contrax.compat.python_control` for optional bidirectional conversion
  between Contrax LTI models and `python-control` `StateSpace` objects.

</div>

## Where To Start

<div class="grid cards contrax-link-grid" markdown="1">

- __[Getting started](getting-started.md)__

  The shortest path from zero context to a working controller.

- __[Tutorials](tutorials/differentiable-lqr.md)__

  Learning-oriented end-to-end workflows for the main control and estimation
  paths.

- __[How-to guides](how-to/tune-lqr-with-gradients.md)__

  Task-oriented recipes such as tuning LQR weights or handling missing
  measurements.

- __[API reference](api/systems.md)__

  Namespace-oriented reference pages for systems, control, simulation,
  estimation, analysis, and types.

- __[Theory](theory/riccati-solvers.md)__

  Short explanation pages on solver choices, discretization, estimation
  pipelines, and JAX transform behavior.

- __[Examples](examples/jax-native-workflows.md)__

  Runnable examples that show how batching, transforms, and multi-step
  workflows fit together.

</div>

## Scope And Honesty

Contrax is explicit about what it is and is not.

It is a systems-and-control library for JAX-first workflows. It is not a
plotting package, not a Simulink-style environment, and not a library that
hides solver maturity behind a larger API surface.

Important solver caveats, transform contracts, and validation expectations are
part of the docs because they are part of the product.
