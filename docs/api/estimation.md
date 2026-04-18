# Estimation

Estimation is the public namespace for recursive filtering, smoothing, and the
first optimization-based estimation primitives in Contrax.

The namespace includes:

- `kalman()` and `kalman_gain()` for linear Gaussian filtering and steady-state
  estimator design
- `kalman_predict()`, `kalman_update()`, `kalman_step()` for one-step runtime
  loops
- `ekf()` and `ukf()` for nonlinear filtering
- `ekf_predict()`, `ekf_update()`, `ekf_step()` for one-step nonlinear loops
- `rts()` and `uks()` for offline smoothing
- `innovation_diagnostics()`, `likelihood_diagnostics()`,
  `ukf_diagnostics()`, and `smoother_diagnostics()` for routine estimator-health summaries
- `mhe_objective()`, `mhe()`, `mhe_warm_start()`, and
  `soft_quadratic_penalty()` for fixed-window optimization-based estimation
- `positive_exp()`, `positive_softplus()`, `spd_from_cholesky_raw()`, and
  `diagonal_spd()` for lightweight constrained parameterization

## Minimal Example

```python
import jax.numpy as jnp
import contrax as cx

sys = cx.dss(
    jnp.array([[1.0, 0.1], [0.0, 1.0]]),
    jnp.array([[0.0], [0.1]]),
    jnp.eye(2),
    jnp.zeros((2, 1)),
    dt=0.1,
)
result = cx.kalman(
    sys,
    Q_noise=1e-3 * jnp.eye(2),
    R_noise=1e-2 * jnp.eye(2),
    ys=jnp.zeros((20, 2)),
)
```

<figure class="contrax-figure">
  <img src="/assets/images/estimation-pipeline.svg"
       alt="Estimation pipeline from model and measurements through filtering, smoothing, and moving-horizon estimation" />
  <figcaption>
    <strong>The estimation surface is layered:</strong> batch filters produce
    filtered trajectories, smoothers revisit them offline, and MHE uses the
    same model and noise assumptions in an explicit fixed-horizon objective.
  </figcaption>
</figure>

## Conventions

- `ys`: measurement sequence with shape `(T, p)`
- `x0`: prior mean on `x_0`
- `P0`: prior covariance on `x_0`
- [`KalmanResult.x_hat`](types.md#contrax.types.KalmanResult.x_hat): filtered
  state sequence with shape `(T, n)`
- [`KalmanResult.P`](types.md#contrax.types.KalmanResult.P): filtered
  covariance sequence with shape `(T, n, n)`
- [`KalmanResult.innovations`](types.md#contrax.types.KalmanResult.innovations):
  measurement residual sequence with shape `(T, p)` for `kalman()` and `ekf()`
- [`UKFResult.predicted_measurements`](types.md#contrax.types.UKFResult.predicted_measurements):
  pre-update measurement means with shape `(T, p)` for `ukf()`
- [`UKFResult.innovation_covariances`](types.md#contrax.types.UKFResult.innovation_covariances):
  innovation covariance sequence with shape `(T, p, p)` for `ukf()`

Contrax uses the update-first batch convention for `kalman()`, `ekf()`, and
`ukf()`: `(x0, P0)` is the prior on `x_0`, and the first scan step updates
with `y_0`.

## Estimator Equations

Contrax uses an update-first batch convention: the pair `(x0, P0)` is the
prior on `x_0`, the first step updates with `y_0`, and only then predicts the
prior on `x_1`.

For linear discrete models, the state and observation maps are

$$
x_{k+1} = A x_k + B u_k + w_k, \qquad
y_k = C x_k + D u_k + v_k
$$

with process noise covariance `Q` and measurement noise covariance `R`.

### Linear Kalman Filter

`kalman()` implements the standard discrete Gaussian filter:

$$
\hat{y}_k^- = C \hat{x}_k^- + D u_k, \qquad
S_k = C P_k^- C^\top + R
$$

$$
K_k = P_k^- C^\top S_k^{-1}
$$

$$
\hat{x}_k^+ = \hat{x}_k^- + K_k \bigl(y_k - \hat{y}_k^-\bigr), \qquad
P_k^+ = (I - K_k C) P_k^-
$$

$$
\hat{x}_{k+1}^- = A \hat{x}_k^+ + B u_k, \qquad
P_{k+1}^- = A P_k^+ A^\top + Q
$$

### Extended Kalman Filter

For nonlinear models

$$
x_{k+1} = f(t_k, x_k, u_k) + w_k, \qquad
y_k = h(t_k, x_k, u_k) + v_k
$$

`ekf()` linearizes the transition and observation maps with JAX Jacobians:

$$
H_k =
\left.\frac{\partial h}{\partial x}\right|_{(t_k, \hat{x}_k^-, u_k)},
\qquad
F_k =
\left.\frac{\partial f}{\partial x}\right|_{(t_k, \hat{x}_k^+, u_k)}
$$

$$
\hat{y}_k^- = h(t_k, \hat{x}_k^-, u_k), \qquad
S_k = H_k P_k^- H_k^\top + R
$$

$$
K_k = P_k^- H_k^\top S_k^{-1}
$$

$$
\hat{x}_k^+ = \hat{x}_k^- + K_k \bigl(y_k - \hat{y}_k^-\bigr), \qquad
P_k^+ = (I - K_k H_k) P_k^-
$$

$$
\hat{x}_{k+1}^- = f(t_k, \hat{x}_k^+, u_k), \qquad
P_{k+1}^- = F_k P_k^+ F_k^\top + Q
$$

If you use the one-step `ekf_update(..., num_iter>1)` helper, the measurement
map is relinearized around the current update iterate while keeping the
prediction covariance fixed.

### Unscented Kalman Filter

`ukf()` avoids local Jacobians and instead propagates sigma points. Given
dimension $n$ and scaling parameter $\lambda$,

$$
\chi^{(0)} = \hat{x}, \qquad
\chi^{(i)} = \hat{x} \pm \left[\sqrt{(n + \lambda) P}\right]_i
$$

At the update stage, sigma points from the current prior
$(\hat{x}_k^-, P_k^-)$ are pushed through the observation map:

$$
\gamma_k^{(i)} = h(t_k, \chi_k^{(i)}, u_k), \qquad
\hat{y}_k^- = \sum_i W_i^{(m)} \gamma_k^{(i)}
$$

$$
S_k =
\sum_i W_i^{(c)}
\bigl(\gamma_k^{(i)} - \hat{y}_k^-\bigr)
\bigl(\gamma_k^{(i)} - \hat{y}_k^-\bigr)^\top + R
$$

$$
P_{xy,k} =
\sum_i W_i^{(c)}
\bigl(\chi_k^{(i)} - \hat{x}_k^-\bigr)
\bigl(\gamma_k^{(i)} - \hat{y}_k^-\bigr)^\top,
\qquad
K_k = P_{xy,k} S_k^{-1}
$$

$$
\hat{x}_k^+ = \hat{x}_k^- + K_k \bigl(y_k - \hat{y}_k^-\bigr), \qquad
P_k^+ = P_k^- - K_k S_k K_k^\top
$$

For prediction, sigma points from $(\hat{x}_k^+, P_k^+)$ are pushed through
the transition map:

$$
\xi_k^{(i)} = f(t_k, \chi_k^{(i)}, u_k), \qquad
\hat{x}_{k+1}^- = \sum_i W_i^{(m)} \xi_k^{(i)}
$$

$$
P_{k+1}^- =
\sum_i W_i^{(c)}
\bigl(\xi_k^{(i)} - \hat{x}_{k+1}^-\bigr)
\bigl(\xi_k^{(i)} - \hat{x}_{k+1}^-\bigr)^\top + Q
$$

### Smoothers

`rts()` is the linear Rauch-Tung-Striebel smoother on top of
[`KalmanResult`][contrax.types.KalmanResult]:

$$
G_k = P_k^+ A^\top (P_{k+1}^-)^{-1}
$$

$$
\hat{x}_k^s = \hat{x}_k^+ + G_k \bigl(\hat{x}_{k+1}^s - \hat{x}_{k+1}^-\bigr)
$$

$$
P_k^s = P_k^+ + G_k \bigl(P_{k+1}^s - P_{k+1}^-\bigr) G_k^\top
$$

`uks()` applies the same backward shape to the unscented filter, but replaces
$A P_k^+$ with an unscented cross-covariance:

$$
P_{x_k x_{k+1}} =
\sum_i W_i^{(c)}
\bigl(\chi_k^{(i)} - \hat{x}_k^+\bigr)
\bigl(\xi_k^{(i)} - \hat{x}_{k+1}^-\bigr)^\top
$$

$$
G_k = P_{x_k x_{k+1}} (P_{k+1}^-)^{-1}
$$

$$
\hat{x}_k^s = \hat{x}_k^+ + G_k \bigl(\hat{x}_{k+1}^s - \hat{x}_{k+1}^-\bigr),
\qquad
P_k^s = P_k^+ + G_k \bigl(P_{k+1}^s - P_{k+1}^-\bigr) G_k^\top
$$

### Moving-Horizon Estimation

`mhe_objective()` and `mhe()` expose the fixed-window optimization form:

$$
J_{\mathrm{MHE}} =
\|x_0 - x_{\mathrm{prior}}\|_{P^{-1}}^2 +
\sum_{k=0}^{T-1}\|x_{k+1} - f(t_k, x_k, u_k)\|_{Q^{-1}}^2 +
\sum_{k=0}^{T}\|y_k - h(t_k, x_k, u_k)\|_{R^{-1}}^2
$$

This is the optimization sibling of the recursive filters: the same model and
noise assumptions appear, but they are expressed as an explicit horizon cost.

For rolling-window use:

- `mhe_warm_start(xs, ...)` shifts a previous solution forward by one step to
  build the next initial guess
- `soft_quadratic_penalty(residuals, weight)` is a small helper for soft state,
  terminal, or envelope penalties inside `extra_cost`

## One-Step Helper Signatures

The one-step helpers follow the order: **model, state `(x, P)`, runtime data
`(u/y)`, noise matrices `(Q/R)`**.

```python
# predict: advances state by one step
x_pred, P_pred = cx.ekf_predict(model, x, P, u, Q_noise)
x_pred, P_pred = cx.kalman_predict(sys, x, P, Q_noise, u=None)  # u optional

# update: fuses a new measurement
x, P, innov = cx.ekf_update(model, x_pred, P_pred, y, R_noise)
x, P, innov = cx.kalman_update(sys, x_pred, P_pred, y, R_noise, u=None)

# combined predict-update step
x, P, innov = cx.ekf_step(model, x, P, u, y, Q_noise, R_noise, observation=h)
x, P, innov = cx.kalman_step(sys, x, P, y, Q_noise, R_noise, u=None)
```

`has_measurement=False` can be passed to all step helpers to skip the update
(useful for streams with missing samples inside `jax.lax.scan`).

## Model Inputs

For nonlinear estimation, the main docs path is reusable system models such as
[`NonlinearSystem`][contrax.systems.NonlinearSystem] or
[`PHSSystem`][contrax.systems.PHSSystem].

That keeps simulation, linearization, and estimation on one shared model
object instead of redefining the same dynamics in several places.

The intended public workflow is:

```python
sys_c = cx.nonlinear_system(f_continuous, output=h, dt=None)
sys_d = cx.sample_system(sys_c, dt=0.1)
result = cx.ukf(sys_d, Q_noise, R_noise, ys, us, x0, P0)
smoothed = cx.uks(sys_d, result, Q_noise, us)
```

`ukf()` returns
[`UKFResult`][contrax.types.UKFResult] rather than
[`KalmanResult`][contrax.types.KalmanResult] because sigma-point filtering
exposes additional public intermediates: predicted measurements, innovation
covariances, per-step likelihood terms, and the one-step prediction quantities
used by `uks()`.

For continuous dynamics with discrete observations, `sample_system()` is the
main library bridge. It integrates the continuous vector field over one sample
interval and returns a discrete-time `NonlinearSystem` that can be passed
directly to `ekf()` or `ukf()`.

## Transform Behavior

Use the one-step helpers when measurements arrive in a runtime loop or when the
filter needs to live inside a larger `jax.lax.scan`.

The important transform contracts are:

- `kalman()`, `ekf()`, and `ukf()` are batch scans over fixed-shape sequences
- the one-step helpers expose missing-measurement handling without Python
  branching on traced values
- `mhe_objective()` is a pure cost function over an explicit candidate
  trajectory, which makes it suitable for JAX-native optimization loops

## Numerical Notes

`kalman_gain()` is the estimator dual of discrete LQR and returns the
measurement-update gain used by `kalman_update()`.

`rts()` requires the same `Q_noise` used by the forward `kalman()` pass so the
smoother uses the same process model. Pass zeros for deterministic dynamics.

`mhe_objective()` is the lowest-level optimization-based estimator surface in
the library. `mhe()` is a thin solver wrapper, not a full nonlinear
programming framework.

## Diagnostics

Contrax now includes a lightweight diagnostics layer for routine estimation
health checks:

- `innovation_diagnostics(innovations, innovation_covariances)` computes
  per-step normalized innovation squared (NIS), innovation norms, and
  covariance conditioning summaries
- `likelihood_diagnostics(log_likelihood_terms)` summarizes innovation-form
  Gaussian log-likelihood terms
- `ukf_diagnostics(result)` builds the standard innovation and likelihood
  summaries directly from a [`UKFResult`][contrax.types.UKFResult]
- `innovation_rms(result)` gives a quick residual-scale summary for either
  [`KalmanResult`][contrax.types.KalmanResult] or
  [`UKFResult`][contrax.types.UKFResult]
- `smoother_diagnostics(smoothed, filtered)` checks whether the smoother
  satisfied its covariance-reduction guarantee
  (`P_smooth ≤ P_filtered`) and reports the worst-case violation, state
  correction magnitudes, and PSD health of the smoothed covariances.
  Accepts `RTSResult` paired with either `KalmanResult` or `UKFResult`.

These helpers are intentionally honest and lightweight. They are good for
filter tuning, regression checks, and notebook-free workflow validation, but
they are not a substitute for a full statistical consistency study.

## Parameterization Helpers

Contrax also exposes a small constrained-parameterization layer for estimation
and tuning workflows:

- `positive_exp(raw)` and `positive_softplus(raw)` map unconstrained values to
  positive ones
- `spd_from_cholesky_raw(raw)` maps an unconstrained square matrix to an SPD
  matrix through a lower-triangular Cholesky factor
- `diagonal_spd(raw_diagonal)` builds a diagonal SPD matrix from unconstrained
  diagonal parameters

These are intentionally low-level building blocks rather than a parameter
framework. They exist so downstream repos do not need to reinvent the same
positivity and covariance-parameterization transforms.

## Related Pages

- [Types](types.md) for [`KalmanResult`][contrax.types.KalmanResult],
  [`UKFResult`][contrax.types.UKFResult],
  [`KalmanGainResult`][contrax.types.KalmanGainResult],
  [`RTSResult`][contrax.types.RTSResult],
  [`InnovationDiagnostics`][contrax.types.InnovationDiagnostics],
  [`SmootherDiagnostics`][contrax.types.SmootherDiagnostics], and
  [`MHEResult`][contrax.types.MHEResult]
- [Kalman filtering](../tutorials/kalman-filtering.md) for an end-to-end
  workflow
- [Handle missing measurements](../how-to/handle-missing-measurements.md) for
  a task-oriented runtime recipe
- [Estimation pipelines](../theory/estimation-pipelines.md) for how the batch,
  one-step, smoothing, and MHE pieces fit together

::: contrax.estimation
