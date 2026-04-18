# Continuous LQR

This tutorial shows the continuous-time control path on the double integrator.
We will build a continuous state-space model, design an LQR controller through
`care()`, simulate the closed loop with the Diffrax-backed continuous
`simulate()` path, and finish with a small gradient check.

```text
continuous state-space model -> care-backed lqr -> continuous simulate -> gradient smoke test
```

Runnable script: `examples/continuous_lqr.py`

This tutorial spans [Systems](../api/systems.md),
[Control](../api/control.md), and [Simulation](../api/simulation.md), but it
stays on the continuous-time branch all the way through.

The control problem is the continuous LQR setup

$$
\dot{x} = A x + B u, \qquad
u(t) = -K x(t)
$$

with infinite-horizon objective

$$
J_c = \int_0^\infty \left(x(t)^\top Q x(t) + u(t)^\top R u(t)\right)\,dt
$$

## Define The Continuous System

The double integrator is a standard first continuous-time example: position and
velocity are the state, and acceleration is the control input.

```python
--8<-- "examples/continuous_lqr.py:setup"

--8<-- "examples/continuous_lqr.py:system"
```

## Design The Controller And Simulate

For `ContLTI` systems, `lqr()` dispatches through `care()`. The resulting gain
is then used in a continuous closed-loop simulation with explicit `duration`
and sample spacing `dt`.

The resulting closed-loop dynamics are

$$
\dot{x} = (A - B K) x
$$

```python
--8<-- "examples/continuous_lqr.py:design-and-simulate"
```

This workflow is part of the public API and is backed by explicit residual and
stability checks in eager mode. The main caveat is comparative maturity:
`care()` is less benchmarked than `dare()`, but it is a supported solver path.

That caveat matters for the docs structure too: continuous LQR is documented as
a real supported workflow, but it is not presented as the most battle-tested
slice of the library.

## Check That Gradients Stay Finite

The continuous path is also differentiable in representative cases. The example
keeps that claim modest and concrete by checking only that a scalar gradient
through `care()` and continuous `simulate()` stays finite.

```python
--8<-- "examples/continuous_lqr.py:gradient-check"
```

## What The Script Prints

Running `examples/continuous_lqr.py` prints the gain, poles, residual summary,
state decay, and gradient smoke-test output:

```text
Continuous LQR — double integrator
  K = [[1.         1.73205081]]
  closed-loop poles = [-0.8660254+0.5j -0.8660254-0.5j]
  all poles stable  = True
  residual norm     = 1.776357e-15
  x[0]  = [1. 0.]
  x[-1] = [-0.00023873  0.00033244]  (should be near zero)
  time horizon      = 10.000 s

Gradient smoke test (d/d(log q) of settling cost):
  grad = -3.848696  (finite: True)

All assertions passed.
```

The exact gradient value is not the main takeaway. The useful checks are stable
continuous-time poles, a small Riccati residual, state convergence toward zero,
and a finite gradient through the end-to-end workflow.

## Validate The Result

For this tutorial, the first checks are:

- the closed-loop poles should have negative real part
- the Riccati residual should be small
- the final state should be much closer to zero than the initial state
- the gradient smoke test should stay finite

Those checks are mirrored by assertions in the runnable example so the tutorial
stays tied to executable behavior.

## Where To Go Next

- [Getting started](../getting-started.md) for the fastest route into the library
- [Control API](../api/control.md) for `care` and continuous `lqr`
- [Simulation API](../api/simulation.md) for continuous `simulate` semantics
- [Linearize, LQR, simulate](linearize-lqr-simulate.md) for the discrete workflow from nonlinear dynamics
- [JAX transform contract](../theory/jax-transform-contract.md) for compiled and differentiable workflow expectations
- [Riccati solvers](../theory/riccati-solvers.md) for continuous and discrete solver details
