# Nonlinear MHE with Constrained Parameter Estimation

This example estimates both state and a physical parameter from noisy
measurements using Moving Horizon Estimation. The parameter (damping
coefficient) is kept positive throughout optimization by representing it in an
unconstrained raw form and applying `positive_softplus()` inside the dynamics.

Runnable script: `examples/nonlinear_mhe.py`

## Problem Setup

A damped pendulum is driven by a small random torque:

$$
\ddot{\theta} = -\frac{g}{l} \sin\theta - b\,\dot{\theta} + \tau, \qquad
y_k = \theta(t_k) + v_k.
$$

The damping coefficient $b > 0$ is unknown. The estimator must recover both the
state $(\theta, \dot\theta)$ and $b$ jointly from angle measurements alone.

## Augmented State and Constrained Parameterization

```python
--8<-- "examples/nonlinear_mhe.py:setup"
```

```python
--8<-- "examples/nonlinear_mhe.py:model"
```

The model augments the state with $b_\text{raw}$ using identity dynamics
(random-walk model), then recovers the physical parameter with

$$
b = \text{softplus}(b_\text{raw}) > 0.
$$

The softplus map keeps $b > 0$ for every raw parameter value, so the optimizer
works over unconstrained $b_\text{raw}$ values.

## Solving MHE

```python
--8<-- "examples/nonlinear_mhe.py:mhe-solve"
```

`mhe()` minimizes `mhe_objective()` over the full window trajectory using
LBFGS and accepts the nonlinear system model directly. The window has N states
and measurements, with N−1 input transitions. The Kalman-style prior term (`x_prior`, `P_prior`) regularizes the
start of the window, and `Q_noise` controls how much the parameter is allowed
to drift across the horizon.

## What the Script Prints

The script reports convergence, status, gradient norm, final cost, estimated
damping and terminal angle alongside the ground truth. It also checks positive
damping and terminal-angle accuracy.

The angle estimate is close to ground truth. The damping estimate is biased:
a 12-step window with a slowly varying random-walk prior gives a short
identification horizon for one scalar parameter. Longer windows or a tighter
prior on $b_\text{raw}$ dynamics may improve the estimate. Here
`solver_converged=False`; inspect status, gradient norm, and cost reduction
before using the estimate or interpreting implicit gradients.

## Related Pages

- [How-to: Build an MHE objective](../how-to/build-mhe-objective.md)
- [FOH estimation](foh-estimation.md) for a complementary EKF workflow on a
  continuous nonlinear model
- [Estimation API](../api/estimation.md)
