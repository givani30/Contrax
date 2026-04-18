# Types

Contrax uses small `eqx.Module` result bundles for solver, simulation-adjacent,
and estimator outputs. These types are part of the public API.

Use them as named return values rather than unpacking by position:

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

- [`LQRResult.residual_norm`](#contrax.types.LQRResult.residual_norm): a JAX
  scalar Riccati residual diagnostic
- [`KalmanResult.innovations`](#contrax.types.KalmanResult.innovations): the
  measurement residual sequence
- [`UKFResult.log_likelihood_terms`](#contrax.types.UKFResult.log_likelihood_terms):
  per-step innovation-form Gaussian log-likelihood terms
- [`KalmanGainResult.K`](#contrax.types.KalmanGainResult.K): the steady-state
  measurement-update gain
- [`MHEResult.x_hat`](#contrax.types.MHEResult.x_hat): the terminal state
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
