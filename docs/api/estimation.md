# Estimation

Use the estimation namespace for recursive filters, smoothers, and
fixed-window optimization.

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
result = cx.kalman(sys, jnp.zeros((20, 2)), Q=0.001 * jnp.eye(2), R=0.01 * jnp.eye(2))
```

<figure class="contrax-figure">
  <img src="/assets/images/estimation-pipeline.svg"
       alt="Estimation pipeline from model and measurements through filtering, smoothing, and moving-horizon estimation" />
  <figcaption>
    <strong>Estimation workflow:</strong> batch filters produce filtered
    trajectories, smoothers use future observations, and MHE optimizes an
    explicit fixed-horizon objective.
  </figcaption>
</figure>

## Conventions

- `ys`: measurement sequence with shape `(T, p)`
- `x0`: prior mean on `x_0`
- `P0`: prior covariance on `x_0`
- [`KalmanResult.xs`](types.md#contrax.types.KalmanResult.xs): filtered
  state sequence with shape `(T, n)`
- [`KalmanResult.Ps`](types.md#contrax.types.KalmanResult.Ps): filtered
  covariance sequence with shape `(T, n, n)`
- [`KalmanResult.innovations`](types.md#contrax.types.KalmanResult.innovations):
  measurement residual sequence with shape `(T, p)` for `kalman()` and `ekf()`
- [`UKFResult.predicted_measurements`](types.md#contrax.types.UKFResult.predicted_measurements):
  pre-update measurement means with shape `(T, p)` for `ukf()`
- [`UKFResult.innovation_covariances`](types.md#contrax.types.UKFResult.innovation_covariances):
  innovation covariance sequence with shape `(T, p, p)` for `ukf()`

Batch `kalman()`, `ekf()`, and `ukf()` use an update-first convention: `(x0,
P0)` is the prior on `x_0`, and the first step updates with `y_0`.

Linear filtering accepts `us=(T, m)` for both feedthrough at k and the transition
from k to k+1. EKF uses the same convention. `kalman` and `ekf` accept a `(T,)`
`has_measurement` mask; missing observations may contain NaNs. Skipped updates
leave the prior unchanged and return zero innovations. The final predicted
transition is not included in `KalmanResult`.

Joseph covariance updates are the default for linear filtering and EKF:

$$
P^+ = (I-KH)P^-(I-KH)^\top + K R K^\top.
$$

Here `H` is `C` for the linear filter. Select compact form with
`cx.solvers.Kalman(covariance_update="compact")` or
`cx.solvers.EKF(covariance_update="compact")`. Pass consistent dtypes for the
model, noise, initial conditions, and observations; float32 workflows retain
float32.
For a controlled linear smoother, pass the forward controls to `rts(..., us=us)`.

## Estimator Equations

The batch pass starts from prior `(x0, P0)` on `x_0`, updates with `y_0`, and
then predicts the prior on `x_1`.

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

Pass `solver=cx.solvers.EKF(num_iter=3)` to `ekf`, `ekf_update` or `ekf_step`
to relinearize around each update iterate while keeping the prediction
covariance fixed. All three use the same configuration.

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
\sum_{k=0}^{N-2}\|x_{k+1} - f(t_k, x_k, u_k)\|_{Q^{-1}}^2 +
\sum_{k=0}^{N-1}\|y_k - h(t_k, x_k, u_k)\|_{R^{-1}}^2
$$

The objective uses the same model and noise assumptions as the recursive
filters, expressed as a cost over the full horizon.

Pass a discrete LTI or nonlinear system directly: `mhe(model, xs_init, us, ys,
x_prior, P_prior, Q_noise, R_noise)`. The window contains N states and
measurements, with N−1 transition inputs; a one-state window is valid. For
input-dependent measurements, `measurement_us` supplies N controls; its default
holds the last transition input at the final observation (zero for N=1).
For sampled FOH models, transition inputs have shape `(N-1, 2, m)` and
`measurement_us` has shape `(N, 2, m)`. Each observation uses its pair's left
endpoint. The default terminal pair holds the last right endpoint at both ends.
`ts` supplies explicit sample times for time-dependent models. Callable
transitions accept `f(x, u)` and require `observation=h` separately.

The arrival, process, and observation covariances must be finite, symmetric,
and positive definite. Invalid covariances report `invalid_problem`, even if
the optimizer converges.

Native Optimistix minimizers are accepted through `solver=`. The wrapper retains
the best finite accepted iterate when the iteration limit is reached and
returns `.status`, `.successful`, `.num_steps`, `.gradient_norm`, `.initial_cost`,
and `.final_cost`. Success requires solver convergence and valid result checks;
a finite cost reduction alone does not establish convergence. `throw=True`
enables checked convergence failures.

Implicit JVP/VJP derivatives include model, noise and explicit parameter data
passed through optimizer arguments. They assume a converged stationary point
with a nonsingular Hessian. Check convergence before interpreting derivatives
of a limited or ill-conditioned solve.

Some Optimistix LBFGS versions that allocate history in JAX's default dtype
cannot initialize float32 trajectories while x64 mode is enabled. The wrapper
preserves the arrays and reports this configuration error. A validated option for small
float32 windows in that context is native `optx.BFGS(rtol=1e-6, atol=1e-6)`;
its dense Hessian storage grows quadratically in the window's variable count.
Alternatively, select a float32 JAX context before solving. JAX global
precision settings are not changed by the estimator.

For rolling-window use:

- `mhe_warm_start(xs, ...)` shifts a previous solution forward by one step to
  build the next initial guess
- `soft_quadratic_penalty(residuals, weight)` is a small helper for soft state,
  terminal, or envelope penalties inside `extra_cost`

Scalar and matrix penalty weights represent cost or precision weighting:
`w * sum(r**2)` or `sum(r.T @ W @ r)`. For covariance-weighted residuals,
apply an explicit factor solve before the penalty.

## Explicit Linear Filter Algorithms

Ordinary `kalman` uses innovation Cholesky solves and Joseph covariance updates.
The innovation covariance must be positive definite. Choose another algorithm
explicitly when its assumptions fit the workflow:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import contrax as cx

sys = cx.dss([[0.9]], [[1.0]], [[1.0]], [[0.0]], dt=0.1)
Q, R = 0.01 * jnp.eye(1), 0.1 * jnp.eye(1)
ys = jnp.zeros((20, 1))
design = cx.kalman_gain(sys, Q, R)
steady = cx.kalman(sys, ys, Q=Q, R=R,
                   solver=cx.solvers.SteadyStateKalman(design))
information = cx.kalman(sys, ys, Q=Q, R=jnp.diag(R),
                       solver=cx.solvers.InformationKalman())
assert steady.successful & information.successful
```

`design.P` is the predicted covariance **before** the first measurement update.
Steady-state mode uses it as the default `P0` and computes the Joseph posterior
covariance once. It validates the gain, Riccati fixed point, initialization and
estimator stability. A mismatched design or missing observation reports
`invalid_problem`, with optional checked failure through `throw=True`.
`SteadyStateKalman` uses the supplied design on each call. To differentiate a
loss with respect to model or noise parameters, construct `kalman_gain` inside
that differentiable workflow and pass its result to the solver. A design held
fixed while its model or noise data changes is inconsistent.

Information mode takes a positive variance vector `(p,)` and factors matrices
in state dimension after whitening the prior. It forms no prior inverse and
no `p`-by-`p` innovation matrix. This can reduce work and storage for wide
observation maps and avoid loss of small diagonal noise in a dense innovation
matrix. It requires positive-definite priors at observed steps; covariance-form
filtering also supports semidefinite priors with nonsingular innovations.
Information mode retains the usual controls, whole-observation masks,
zero/identity initialization defaults, and float32/float64 precision. Supply
positive-definite covariances; numerical failures are returned in the result.
Both options are accepted by the
linear online update and combined-step helpers as well as batch filtering.

## One-Step Helper Signatures

All batch filters take `(model, ys, *, Q, R, us=None, x0=None, P0=None)`.
Initial means default to zero and covariances to identity; both describe the
prior at the first observation. Explicit steady-state mode instead defaults
the covariance to its design's predicted covariance. Omitted controls use the
model's input dimension. Set `input_dim` on controlled nonlinear models or
provide `us`. Sampled FOH models accept `(T, 2, m)` endpoint pairs.

The one-step helpers take `(model, x, P)` for predictions and
`(model, x, P, y)` for updates and combined steps. Noise and controls are
keyword-only, with the same `Q`, `R` and `u` names throughout.

```python
# predict: advances state by one step
prediction = cx.ekf_predict(model, x, P, u=u, Q=Q_noise)
x_pred, P_pred = prediction.x, prediction.P
prediction = cx.kalman_predict(sys, x, P, Q=Q_noise, u=None)  # u optional
x_pred, P_pred = prediction.x, prediction.P

# update: fuses a new measurement
update = cx.ekf_update(model, x_pred, P_pred, y, R=R_noise)
x, P, innov = update.x, update.P, update.innovation
update = cx.kalman_update(sys, x_pred, P_pred, y, R=R_noise, u=None)
x, P, innov = update.x, update.P, update.innovation

# combined predict-update step
update = cx.ekf_step(model, x, P, y, u=u, Q=Q_noise, R=R_noise)
x, P, innov = update.x, update.P, update.innovation
update = cx.kalman_step(sys, x, P, y, Q=Q_noise, R=R_noise, u=None)
x, P, innov = update.x, update.P, update.innovation
```

`has_measurement=False` can be passed to all step helpers to skip the update
(useful for streams with missing samples inside `jax.lax.scan`).

For an online predict-update step, `u` belongs to the transition from the
previous posterior. Pass `u_measurement=` separately when the input used by
the observation has changed. For time-dependent EKF models, `t` is the
transition start time and `measurement_time` is the observation time; the
default observation time adds the model's sample period when available.

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
result = cx.ukf(sys_d, ys, Q=Q_noise, R=R_noise, us=us, x0=x0, P0=P0)
smoothed = cx.uks(result)
```

`ukf()` returns [`UKFResult`][contrax.types.UKFResult], which includes predicted
measurements, innovation covariances, per-step likelihood terms, and the
prediction quantities used by `uks()`.

The filter result retains `.has_measurement`. Missing observations have zero
innovation, predicted measurement, innovation covariance and likelihood
placeholders. `ukf_diagnostics` excludes them from solves and averages. The
individual diagnostics functions also accept `has_measurement=`. With no
observations, aggregate means and extrema are NaN, likelihood totals are zero,
and `nonfinite` remains false when all observations are missing.

For continuous dynamics with discrete observations, `sample_system()` integrates
the continuous vector field over one sample
interval and returns a discrete-time `NonlinearSystem` that can be passed
directly to `ekf()` or `ukf()`.

## Transform Behavior

Batch filters, online helpers, and smoothers expose JAX scalar `.status` and
`.successful` diagnostics. Online prediction results have `.x` and `.P`;
updates also have `.innovation`. Invalid noise or initial covariances return
`invalid_problem`; nonfinite outputs or
loss of positive semidefiniteness return `numerical_failure`. Sampled-model
integration failures retain their backend status, including iteration limits.
Shape/configuration errors raise immediately.

`throw=False` returns these diagnostics. With `throw=True`, failure raises
during execution, including under `jit` when the caller selects only a state
field. Covariance validation uses dtype-scaled roundoff tolerances. Status
decisions are nondifferentiable; objectives on states,
covariances and likelihoods retain their transform contracts.

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

`rts()` requires the same `Q` used by the forward `kalman()` pass so the
smoother uses the same process model. Pass zeros for deterministic dynamics.

`mhe_objective()` exposes the fixed-window cost directly. `mhe()` minimizes it
with a configured optimizer and returns convergence diagnostics.

## Diagnostics

Estimation diagnostics summarize routine filter and smoother health:

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

Use these summaries for filter tuning and regression checks. Statistical
consistency still depends on the model, noise assumptions, and data.

## Parameterization Helpers

Constrained-parameterization helpers support estimation and tuning workflows:

- `positive_exp(raw)` and `positive_softplus(raw)` map unconstrained values to
  positive ones
- `spd_from_cholesky_raw(raw)` maps an unconstrained square matrix to an SPD
  matrix through a lower-triangular Cholesky factor
- `diagonal_spd(raw_diagonal)` builds a diagonal SPD matrix from unconstrained
  diagonal parameters

These helpers accept real inputs and support `jit`, `vmap` and forward/reverse
differentiation. Integer raw parameters are promoted to floating point; floating
inputs retain their precision unless an explicitly higher-precision floor is
provided. Floors must be finite, nonnegative scalars. Cholesky maps require a
square matrix and `diagonal_spd` requires a vector. An overridden triangular
diagonal participates in dtype promotion, so fractional entries are retained.

Positive definiteness holds in exact arithmetic. With a zero floor, very negative
parameters can underflow to zero; exponential overflow and ill-conditioned
Cholesky factors can also defeat floating-point definiteness. The helpers apply
the requested transform directly.

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
