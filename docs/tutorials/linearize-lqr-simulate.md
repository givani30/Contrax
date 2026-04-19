# Linearize, LQR, Simulate

This tutorial shows a standard control workflow on a pendulum. The model starts
as ordinary nonlinear JAX code, then Contrax linearizes it, discretizes the
local state-space realization, designs an LQR controller, and simulates the
closed loop.

<figure class="contrax-figure">
  <img src="/assets/images/linearize-lqr-pipeline.svg"
       alt="Pipeline from nonlinear dynamics through linearization, discretization, LQR design, and closed-loop simulation" />
  <figcaption>
    <strong>The tutorial path:</strong> start from ordinary nonlinear JAX code,
    build a local linear model, discretize it, design `K`, and validate the
    resulting closed-loop rollout.
  </figcaption>
</figure>

Runnable script: `examples/linearize_lqr_simulate.py`

This tutorial is the clearest first tour through the new docs structure:
[Systems](../api/systems.md) for model construction and linearization,
[Control](../api/control.md) for LQR design, and
[Simulation](../api/simulation.md) for the closed-loop rollout.

The pendulum model in the example is

$$
\dot{\theta} = \omega, \qquad
\dot{\omega} = -\sin(\theta) + \tau
$$

and the local LQR design solves the usual discrete finite-energy objective
after `linearize() -> c2d()`:

$$
J = \sum_{k=0}^{\infty} \left(x_k^\top Q x_k + u_k^\top R u_k\right)
$$

## Define Dynamics

```python
--8<-- "examples/linearize_lqr_simulate.py:setup"

--8<-- "examples/linearize_lqr_simulate.py:dynamics"
```

## Linearize

`linearize` computes the local state and input Jacobians with JAX automatic
differentiation and wraps them directly as a continuous state-space system.
You can pass raw functions or a `nonlinear_system(...)` model object; the
resulting local `ContLTI` is the same.

Around the operating point $(x_{\mathrm{eq}}, u_{\mathrm{eq}})$, the local
model is

$$
\delta \dot{x} \approx A\,\delta x + B\,\delta u, \qquad
\delta y \approx C\,\delta x + D\,\delta u
$$

```python
x_eq = jnp.array([0.1, 0.0])
u_eq = jnp.array([jnp.sin(0.1)])

sys_c = cx.linearize(pendulum, x_eq, u_eq, output=sensor)
```

## Build a System

The linearized model is continuous-time, so the next step is discretization.
For this local-control workflow the usual path is zero-order hold with `c2d`.

```python
sys_d = cx.c2d(sys_c, dt=0.05)
```

## Design an LQR Controller

`lqr()` dispatches to the discrete Riccati solver for `DiscLTI` systems. The
result bundle exposes the feedback gain, Riccati solution, and closed-loop
poles.

```python
result = cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1)))
closed_loop = cx.state_feedback(sys_d, result.K)
```

In a JAX-first workflow, put the fixed-shape design and simulation path behind a
compiled boundary:

```python
--8<-- "examples/linearize_lqr_simulate.py:design-and-simulate"
```

That compiled function keeps the full path in one JAX program: choose an
operating point, linearize, design the controller, and simulate the closed
loop.

That is one of Contrax's main ergonomic goals: the code still reads like a
standard controls workflow, but the full pipeline also stays inside a
fixed-shape JAX program.

## What The Script Prints

Running `examples/linearize_lqr_simulate.py` prints the designed poles and the
closed-loop decay summary:

```text
Linearize -> c2d -> lqr -> simulate
closed-loop poles = [0.96558903+0.04716561j 0.96558903-0.04716561j]
initial state norm = 0.250000
final state norm   = 0.026099
final state        = [-0.02007372  0.01667922]
time horizon       = 3.950 s
```

The exact pole locations depend on the linearization point and weights. The
useful check is simpler: poles inside the unit circle and a final state norm
much smaller than the initial one.

## Validate The Result

On this discrete closed-loop example, the first checks are simple: poles inside
the unit circle, the expected fixed shapes, and a final state smaller than the
initial one.

```python
--8<-- "examples/linearize_lqr_simulate.py:validate"
```

For unfamiliar systems, inspect the closed-loop poles and the response
trajectory before trusting a controller. The runnable script mirrors these
checks so this tutorial stays tied to executable code.

## How To Read This Workflow In Contrax

- `linearize` belongs to the Systems surface because it turns a nonlinear
  model into a local state-space system.
- `c2d` is also part of Systems because discretization is still part of model
  preparation.
- `lqr` and `state_feedback` belong to Control because they design and apply
  the controller.
- `simulate` belongs to Simulation because it executes the resulting closed
  loop.

## Where To Go Next

- [Getting started](../getting-started.md) for the fastest route into the library
- [Systems API](../api/systems.md) for `nonlinear_system`, `linearize`, and `c2d`
- [Control API](../api/control.md) for `lqr` and `state_feedback`
- [Simulation API](../api/simulation.md) for closed-loop rollout conventions
- [Batch controller design](../how-to/batch-controller-design.md) for the vmapped version of this pattern
- [Discretization and linearization](../theory/discretization-and-linearization.md) for the explanation layer
