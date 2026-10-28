# FOH Estimation: EKF with First-Order-Hold Inputs

This example filters a nonlinear oscillator with an input that varies between
samples. First-order-hold (FOH) interpolation represents the input as a linear
profile over each sample interval.

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
measurements. FOH follows this piecewise-linear profile within each interval;
zero-order hold models the input as constant there.

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

The EKF uses `ekf_predict` and `ekf_update` inside one `jax.lax.scan`, compiling
the time loop once. The sampled system object supplies dynamics, observation
and integration diagnostics; JAX computes the local Jacobians. Prediction uses
the interval's FOH pair, and the following measurement uses its right endpoint.

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
