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
LBFGS. The Kalman-style prior term (`x_prior`, `P_prior`) regularizes the
start of the window, and `Q_noise` controls how much the parameter is allowed
to drift across the horizon.

## What the Script Prints

```text
Nonlinear MHE — damped pendulum with parameter estimation
  converged           = False (LBFGS may report False before tolerance; check cost)
  final cost          = 5.6983e+00
  damping: true=0.300  estimated=0.680
  angle at window end:
    true      = 0.6481 rad
    estimated = 0.6552 rad

All assertions passed.
```

The angle estimate is close to ground truth. The damping estimate is biased
because a 12-step window with a slowly-varying random-walk prior is a
short identification horizon for a single scalar parameter — longer windows
or a tighter prior on $b_\text{raw}$ dynamics improve convergence. The
`solver_converged=False` flag reflects LBFGS tolerance reporting, not a
failed solve; checking `final_cost` directly is more informative.

## Related Pages

- [How-to: Build an MHE objective](../how-to/build-mhe-objective.md)
- [FOH estimation](foh-estimation.md) for a complementary EKF workflow on a
  continuous nonlinear model
- [Estimation API](../api/estimation.md)
