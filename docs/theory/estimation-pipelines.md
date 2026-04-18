# Estimation Pipelines

Contrax supports both recursive and optimization-based estimation workflows.

That matters because the right question is usually not “which filter do I
know?”, but “what estimation pipeline fits this system, data stream, and JAX
workflow?”

<figure class="contrax-figure">
  <img src="/assets/images/estimation-pipeline.svg"
       alt="Estimation pipeline showing model and data flowing into filters, smoothers, and MHE" />
  <figcaption>
    <strong>Think in layers:</strong> recursive filters are the online core,
    smoothers revisit the same trajectory offline, and MHE turns the same model
    into an explicit optimization objective over a fixed window.
  </figcaption>
</figure>

## Recursive Filters

The recursive side of the library includes:

- `kalman()` for linear Gaussian systems
- `ekf()` for nonlinear models with local Jacobian linearization
- `ukf()` for nonlinear models with sigma-point propagation

Each of those also has one-step helpers for runtime loops.

Use the one-step helpers when measurements arrive online or when the estimator
must live inside a service loop. Use the batch filter when you want an entire
offline pass over a fixed sequence.

Contrax uses an update-first batch convention. The forward pass starts from a
prior on `x_0`, updates with `y_0`, then predicts the prior on `x_1`.

### Linear Kalman Filter

For the linear discrete model

$$
x_{k+1} = A x_k + B u_k + w_k, \qquad
y_k = C x_k + D u_k + v_k
$$

the filter recursion is

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

This is the right starting point when the model is already linear and the
Gaussian assumptions are a reasonable first approximation.

### Extended Kalman Filter

For nonlinear transition and observation maps,

$$
x_{k+1} = f(t_k, x_k, u_k) + w_k, \qquad
y_k = h(t_k, x_k, u_k) + v_k
$$

the EKF replaces the linear matrices with local Jacobians:

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
\hat{x}_k^+ = \hat{x}_k^- +
P_k^- H_k^\top S_k^{-1}\bigl(y_k - \hat{y}_k^-\bigr)
$$

$$
P_{k+1}^- = F_k P_k^+ F_k^\top + Q
$$

This is the lightest nonlinear recursive path in the library. It works best
when the local linearization is a good approximation over one update step.

### Unscented Kalman Filter

The UKF keeps the same state-estimation goal but avoids local Jacobians. It
starts from sigma points

$$
\chi^{(0)} = \hat{x}, \qquad
\chi^{(i)} = \hat{x} \pm \left[\sqrt{(n + \lambda) P}\right]_i
$$

and pushes them through the observation and transition maps:

$$
\gamma_k^{(i)} = h(t_k, \chi_k^{(i)}, u_k), \qquad
\xi_k^{(i)} = f(t_k, \chi_k^{(i)}, u_k)
$$

The predicted moments are reconstructed with sigma-point weights:

$$
\hat{y}_k^- = \sum_i W_i^{(m)} \gamma_k^{(i)}, \qquad
\hat{x}_{k+1}^- = \sum_i W_i^{(m)} \xi_k^{(i)}
$$

$$
S_k =
\sum_i W_i^{(c)}
\bigl(\gamma_k^{(i)} - \hat{y}_k^-\bigr)
\bigl(\gamma_k^{(i)} - \hat{y}_k^-\bigr)^\top + R
$$

$$
P_{k+1}^- =
\sum_i W_i^{(c)}
\bigl(\xi_k^{(i)} - \hat{x}_{k+1}^-\bigr)
\bigl(\xi_k^{(i)} - \hat{x}_{k+1}^-\bigr)^\top + Q
$$

and the update uses the cross-covariance

$$
P_{xy,k} =
\sum_i W_i^{(c)}
\bigl(\chi_k^{(i)} - \hat{x}_k^-\bigr)
\bigl(\gamma_k^{(i)} - \hat{y}_k^-\bigr)^\top
$$

to form

$$
K_k = P_{xy,k} S_k^{-1}, \qquad
\hat{x}_k^+ = \hat{x}_k^- + K_k \bigl(y_k - \hat{y}_k^-\bigr)
$$

This is the better fit when the local Jacobian story is too crude but you
still want a recursive estimator rather than a full horizon solve.

## Smoothers

Recursive filters consume information causally. Smoothers revisit the sequence
with future information available.

Contrax provides:

- `rts()` for filtered linear Kalman results
- `uks()` for filtered unscented Kalman results

These are offline tools. They are not replacements for runtime loops, but they
are valuable when you want a retrospective state estimate or a comparison
target for an optimization-based method.

### RTS Smoother

For the linear filter, the Rauch-Tung-Striebel backward pass is

$$
G_k = P_k^+ A^\top (P_{k+1}^-)^{-1}
$$

$$
\hat{x}_k^s = \hat{x}_k^+ + G_k \bigl(\hat{x}_{k+1}^s - \hat{x}_{k+1}^-\bigr)
$$

$$
P_k^s = P_k^+ + G_k \bigl(P_{k+1}^s - P_{k+1}^-\bigr) G_k^\top
$$

### Unscented RTS-Style Smoother

For `uks()`, the same backward structure is used, but the smoother gain is
built from the unscented cross-covariance:

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

## MHE As A Sibling Workflow

Moving-horizon estimation is not “just another Kalman filter.” In Contrax it is
treated as a sibling fixed-horizon optimization workflow:

- an arrival cost anchors the start of the window
- process costs enforce model consistency across transitions
- measurement costs enforce agreement with the observed outputs
- optional extra costs let the user express soft constraints or domain terms

That is why `mhe_objective()` exists as a first-class pure cost function. It
lets you keep the model, horizon, and objective explicit rather than burying
them inside a solver wrapper.

The fixed-horizon objective is the optimization sibling of the recursive
filters:

$$
J_{\mathrm{MHE}} =
\|x_0 - x_{\mathrm{prior}}\|_{P^{-1}}^2 +
\sum_{k=0}^{T-1}\|x_{k+1} - f(t_k, x_k, u_k)\|_{Q^{-1}}^2 +
\sum_{k=0}^{T}\|y_k - h(t_k, x_k, u_k)\|_{R^{-1}}^2
$$

## Choosing Between The Current Paths

Prefer:

- `kalman()` when the model is linear and Gaussian assumptions are a good first
  approximation
- `ekf()` when a local linearization is a reasonable approximation and you want
  the lightest nonlinear recursive path
- `ukf()` when sigma-point propagation is a better fit than local Jacobians
- `rts()` or `uks()` when you need an offline smoothed trajectory
- `mhe_objective()` or `mhe()` when you want a fixed-window optimization-based
  estimate with explicit costs

## Transform Contracts

The estimation surface is designed around fixed-shape JAX workflows:

- batch filters are scans
- one-step helpers can live inside larger scans
- missing-measurement handling avoids Python branching on traced values
- `mhe_objective()` is a pure JAX cost over explicit arrays

That makes estimation pipelines compatible with the broader Contrax story:
compiled, differentiable, and composable.

## Related Pages

- [Estimation API](../api/estimation.md)
- [Kalman filtering](../tutorials/kalman-filtering.md)
- [Handle missing measurements](../how-to/handle-missing-measurements.md)
- [Build an MHE objective](../how-to/build-mhe-objective.md)
