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

The key design pattern is **parameter augmentation**: $b$ is lifted into the
state vector as $b_\text{raw}$ with identity dynamics (random-walk model), and
the physical value is recovered inside the model via

$$
b = \text{softplus}(b_\text{raw}) > 0.
$$

This keeps the MHE objective unconstrained everywhere. The optimizer explores
all of $\mathbb{R}$ in $b_\text{raw}$ space and the constraint $b > 0$ is
enforced structurally, not by an inequality constraint or a projection step.

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

The angle estimate is close to ground truth. The damping estimate is biased
because a 12-step window with a slowly-varying random-walk prior is a
short identification horizon for a single scalar parameter — longer windows
or a tighter prior on $b_\text{raw}$ dynamics improve convergence. The
`solver_converged=False` flag means convergence was not established. The finite
accepted estimate may still be useful, but inspect status, gradient norm and
cost reduction before using it or interpreting implicit gradients.

## Related Pages

- [How-to: Build an MHE objective](../how-to/build-mhe-objective.md)
- [FOH estimation](foh-estimation.md) for a complementary EKF workflow on a
  continuous nonlinear model
- [Estimation API](../api/estimation.md)
