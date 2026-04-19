# JAX-Native Workflows

This page collects the main workflow patterns Contrax supports well.

If you are new to the project, read [Differentiable LQR](../tutorials/differentiable-lqr.md)
first. That page is the clearest end-to-end example. This page is the broader
pattern map.

Runnable scripts:

- `examples/differentiable_lqr.py`
- `examples/linearize_lqr_simulate.py`
- `examples/continuous_lqr.py`
- `examples/lqr_optimal_execution.py`
- `examples/continuous_nonlinear_estimation.py`
- `examples/structured_nonlinear_estimation.py`

Use this page as a bridge between the high-level tutorials and the namespace
reference pages:

- [Systems](../api/systems.md)
- [Control](../api/control.md)
- [Simulation](../api/simulation.md)
- [Estimation](../api/estimation.md)

## 1. Tune LQR Weights With Gradients

`dare` and `lqr` can sit inside an objective and be differentiated with respect
to controller weights.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
sys = cx.dss(A, B, jnp.eye(2), jnp.zeros((2, 1)), dt=0.05)
x0 = jnp.array([1.0, 0.0])


def objective(q_diag, log_r):
    Q = jnp.diag(q_diag)
    R = jnp.exp(log_r)[None, None]
    K = cx.lqr(sys, Q, R).K
    _, xs, _ = cx.simulate(sys, x0, lambda t, x: -K @ x, num_steps=80)
    return jnp.sum(xs**2)


objective_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
cost, (dq, dlog_r) = objective_and_grad(jnp.ones(2), jnp.array(0.0))
```

That makes weight tuning feel like normal JAX optimization rather than a
separate controller-design step.

Related pages:
- [Tune LQR with gradients](../how-to/tune-lqr-with-gradients.md)
- [Differentiable LQR](../tutorials/differentiable-lqr.md)

## 2. Turn Nonlinear Dynamics Into State-Space Models

`linearize` (also available as `linearize_ss`) lets you move from nonlinear
plant code to local state-space models with one JAX-compatible call.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx


def pendulum(t, x, u):
    theta, theta_dot = x
    torque = u[0]
    return jnp.array([theta_dot, -jnp.sin(theta) + torque])


def sensor(x, u):
    return x


x_eq = jnp.array([0.1, 0.0])
u_eq = jnp.array([jnp.sin(0.1)])

sys_c = cx.linearize(pendulum, x_eq, u_eq, output=sensor)
sys_d = cx.c2d(sys_c, dt=0.05)
```

That makes it easy to keep your plant model nonlinear and only linearize where
you actually need a controller or estimator.

Related pages:
- [Systems API](../api/systems.md)
- [Discretization and linearization](../theory/discretization-and-linearization.md)

## 3. Design Controllers Over Batches Of Operating Points

Because the workflow is JAX-native, you can `vmap` over operating points or
design conditions.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx


def pendulum(t, x, u):
    return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])


def sensor(x, u):
    return x


def design(x_eq, u_eq):
    sys_c = cx.linearize(pendulum, x_eq, u_eq, output=sensor)
    sys_d = cx.c2d(sys_c, dt=0.05)
    return cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1))).K


x_eqs = jnp.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]])
u_eqs = jnp.zeros((3, 1))
batched_design = jax.jit(jax.vmap(design))
Ks = batched_design(x_eqs, u_eqs)
```

This is a natural starting point for gain scheduling, operating-point sweeps,
or controller redesign over a grid.

Related pages:
- [Batch controller design](../how-to/batch-controller-design.md)
- [Linearize, LQR, simulate](../tutorials/linearize-lqr-simulate.md)

## 4. JIT The Full Closed-Loop Path

Controller design and simulation can stay in one compiled pipeline.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
sys = cx.dss(A, B, jnp.eye(2), jnp.zeros((2, 1)), dt=0.05)


@jax.jit
def run(q_scale):
    Q = jnp.diag(jnp.array([10.0, 1.0]) * q_scale)
    R = jnp.array([[1.0]])
    K = cx.lqr(sys, Q, R).K
    return cx.simulate(sys, jnp.array([1.0, 0.0]), lambda t, x: -K @ x, num_steps=60)


ts, xs, ys = run(jnp.array(1.0))
```

That keeps redesign and simulation in the same JAX program instead of pushing
part of the workflow into a separate control toolbox.

## 5. Estimate Continuous Nonlinear Systems On A Discrete Grid

The newer estimation surface is meant to keep continuous plant models inside the
same JAX program as recursive estimation. `sample_system()` builds a discrete
transition map by integrating the continuous dynamics over one sample interval,
and `foh_inputs()` lets that bridge see first-order-hold inputs instead of a
constant value inside each step.

Runnable scripts:
- `examples/continuous_nonlinear_estimation.py`
- `examples/structured_nonlinear_estimation.py`

Related pages:
- [Continuous nonlinear estimation](continuous-nonlinear-estimation.md)
- [Structured nonlinear estimation](structured-nonlinear-estimation.md)
- [Estimation API](../api/estimation.md)
- [Simulation API](../api/simulation.md)

## 6. Estimation And Optimization Live In The Same JAX World

Contrax's estimation surface is also meant to compose with the same fixed-shape
JAX workflows. Batch filters use scans, one-step helpers fit runtime loops, and
MHE objective construction is written as an ordinary differentiable cost.

That means the library is not only about classical LTI design; it is also
aiming toward a broader systems-estimation-control workflow where filtering,
smoothing, and trajectory fitting feel like neighboring tools.

Related pages:
- [Kalman filtering](../tutorials/kalman-filtering.md)
- [Handle missing measurements](../how-to/handle-missing-measurements.md)
- [Build an MHE objective](../how-to/build-mhe-objective.md)
- [Estimation pipelines](../theory/estimation-pipelines.md)

## 7. Cast Quadratic Optimal Execution As LQR

Contrax is not a finance library, but some finance problems map cleanly onto
standard control objects. A simple execution model is one of them.

Treat remaining inventory as the state and signed inventory change as the
control:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0]])
B = jnp.array([[1.0]])
sys = cx.dss(A, B, jnp.array([[1.0]]), jnp.zeros((1, 1)), dt=1.0)

Q = jnp.array([[2.5]])   # inventory risk
R = jnp.array([[0.4]])   # temporary impact / trading cost
K = cx.lqr(sys, Q, R).K
```

Then the usual LQR feedback law gives a liquidation schedule. Because the
whole path stays inside JAX, you can also differentiate through the Riccati
solve to tune the execution urgency or `vmap` the design across many assets.

Runnable script:
- `examples/lqr_optimal_execution.py`

Related page:
- [LQR optimal execution](lqr-optimal-execution.md)
