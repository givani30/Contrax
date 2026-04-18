# FOH Estimation: EKF with First-Order-Hold Inputs

This example filters a nonlinear oscillator whose input varies continuously
between sample steps. Rather than assuming the input is constant over each
sample interval (zero-order hold), it models the piecewise-linear profile
exactly using first-order-hold (FOH) interpolation.

Runnable script: `examples/foh_estimation.py`

## Problem Setup

The plant is a Van der Pol oscillator driven by a scalar input:

$$
\dot{x}_1 = x_2, \qquad
\dot{x}_2 = \mu (1 - x_1^2) x_2 - x_1 + u(t)
$$

The sensor observes the first state only:

$$
y_k = x_1(t_k) + v_k.
$$

The input $u(t)$ is sinusoidal and sampled on the same grid as the
measurements. Within each sample interval, $u(t)$ is piecewise-linear rather
than constant, so FOH gives a tighter discrete model than ZOH.

## Setup and Model

```python
--8<-- "examples/foh_estimation.py:setup"
```

```python
--8<-- "examples/foh_estimation.py:model"
```

`sample_system(..., input_interpolation="foh")` builds a discrete transition
model that integrates the continuous dynamics under a linearly interpolated
input $u(t) = (1 - \tau) u_k + \tau u_{k+1}$ over $\tau \in [0, 1]$.

Each step of this discrete model takes a stacked input pair
`u_pair = [[u_k], [u_{k+1}]]` with shape `(2, m)`. `foh_inputs(us)` converts
a raw `(T, m)` sequence into the `(T, 2, m)` pairs the model expects.

## Simulating and Filtering

```python
--8<-- "examples/foh_estimation.py:generate-data"
```

```python
--8<-- "examples/foh_estimation.py:filter"
```

The EKF loop uses the one-step helpers `ekf_predict` and `ekf_update`. The
FOH-discrete model is passed as the dynamics callable, so the filter linearizes
it automatically via JAX Jacobians.

## What the Script Prints

```text
Continuous-model EKF with FOH input interpolation
  steps              = 50
  position RMSE      = 0.1204
  velocity RMSE      = 0.1999
  true  final state  = [0.77012901 2.5961284 ]
  estim final state  = [0.75520658 2.62896228]

All assertions passed.
```

## Related Pages

- [Continuous nonlinear estimation](continuous-nonlinear-estimation.md) for a
  similar workflow using UKF/UKS instead of EKF
- [Estimation API](../api/estimation.md)
- [Simulation API](../api/simulation.md) for `sample_system` and `foh_inputs`
