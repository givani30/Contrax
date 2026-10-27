# Types

Contrax uses small `eqx.Module` result bundles for solver, simulation-adjacent,
and estimator outputs. These types are part of the public API.

Use them as named return values rather than unpacking by position:

- [`SimResult`][contrax.types.SimResult] for `simulate()`, `lsim()`,
  `step_response()`, `impulse_response()`, and `initial_response()`
- [`SampleResult`][contrax.types.SampleResult] for a discrete nonlinear model's
  `step()` method, including sampled integration diagnostics
- [`LyapunovResult`][contrax.types.LyapunovResult] for `lyap()`, `dlyap()` and
  controllability/observability Gramians
- [`PlaceResult`][contrax.types.PlaceResult] for `place()`
- [`LQRResult`][contrax.types.LQRResult] for `dare()`, `care()`, `lqr()`, and
  `lqi()`
- [`KalmanGainResult`][contrax.types.KalmanGainResult] for `kalman_gain()`
- [`KalmanResult`][contrax.types.KalmanResult] for `kalman()` and `ekf()`
- [`UKFResult`][contrax.types.UKFResult] for `ukf()`
- [`RTSResult`][contrax.types.RTSResult] for `rts()` and `uks()`
- [`MHEResult`][contrax.types.MHEResult] for `mhe()`
- [`PHSStructureDiagnostics`][contrax.types.PHSStructureDiagnostics] for
  `phs_diagnostics()`
- [`InnovationDiagnostics`][contrax.types.InnovationDiagnostics] and
  [`LikelihoodDiagnostics`][contrax.types.LikelihoodDiagnostics] for
  filter-health helpers
- [`SmootherDiagnostics`][contrax.types.SmootherDiagnostics] for
  RTS/UKS covariance-reduction health checks

## Why Result Bundles Exist

The result types are not decorative wrappers. They are part of the numerical
workflow design:

- they make multi-output solver contracts explicit
- they keep field names stable across tutorials, examples, and compiled code
- they remain pytrees, so downstream JAX code can return or differentiate
  through fields such as [`LQRResult.K`](#contrax.types.LQRResult.K) without
  conversion to ad hoc dicts

## Important Fields

Some fields are especially worth knowing about:

Simulation, filtering, smoothing and MHE all expose the state trajectory as
`.xs` and its last state as `.final_state`. Filtering and smoothing covariances
are `.Ps`. A single covariance, such as `kalman_gain(...).P`, remains `.P`.
An empty trajectory has no final state: accessing `.final_state` raises
`ValueError`. Apply `vmap` to individual solves when working with batches.

Simulation exposes `.state_times` aligned with `.xs` and `.output_times`
aligned with `.ys`. `.ts` is the output-time vector used by tuple unpacking.

`LQRResult.closed_loop` and `KalmanGainResult.closed_loop` retain the designed
error/feedback transition matrices. Their `.poles` properties evaluate
eigenvalues on access, so unused pole diagnostics do not restrict higher-order
derivatives of gain/covariance objectives. Pole objectives follow JAX's
nonsymmetric eigenvalue differentiation limits; higher derivatives of those
eigenvalues are not provided by JAX.

- [`LQRResult.residual_norm`](#contrax.types.LQRResult.residual_norm): a JAX
  scalar Riccati residual diagnostic
- [`KalmanResult.innovations`](#contrax.types.KalmanResult.innovations): the
  measurement residual sequence
- [`UKFResult.log_likelihood_terms`](#contrax.types.UKFResult.log_likelihood_terms):
  per-step innovation-form Gaussian log-likelihood terms
- [`KalmanGainResult.K`](#contrax.types.KalmanGainResult.K): the steady-state
  measurement-update gain
- [`MHEResult.final_state`](#contrax.types.MHEResult.final_state): the terminal state
  estimate from the optimized window
- [`PHSStructureDiagnostics.power_balance_residual`](#contrax.types.PHSStructureDiagnostics.power_balance_residual):
  the local port-Hamiltonian power-balance residual
- [`InnovationDiagnostics.mean_nis`](#contrax.types.InnovationDiagnostics.mean_nis):
  the mean normalized innovation squared summary
- [`SmootherDiagnostics.min_covariance_reduction`](#contrax.types.SmootherDiagnostics.min_covariance_reduction):
  the worst-case smoother covariance-reduction health flag; negative means the
  smoother increased variance in some direction

## Related Pages

- [Control](control.md) for [`LQRResult`][contrax.types.LQRResult]
- [Estimation](estimation.md) for the Kalman-family and MHE result bundles
- [Systems](systems.md) for the structured-system diagnostics surface

::: contrax.types
