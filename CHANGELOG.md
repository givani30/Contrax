# Changelog

All notable changes to Contrax are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-10-28

### Breaking changes and migration from 0.1.0

This release changes call signatures, return types and several numerical
defaults. Update callers before upgrading.

- **Batch filters:** observations are the second positional argument; noise,
  controls and initialization are keyword-only. `Q_noise`/`R_noise` become
  `Q`/`R` in the recursive filter family, including `kalman_gain` and `rts`.

  ```python
  # 0.1.0
  cx.kalman(sys, Q, R, ys, x0, P0)
  cx.ekf(model, Q, R, ys, us, x0, P0)

  # 0.2.0 (the same convention applies to ukf)
  cx.kalman(sys, ys, Q=Q, R=R, x0=x0, P0=P0)
  cx.ekf(model, ys, Q=Q, R=R, us=us, x0=x0, P0=P0)
  ```

- **Online filters:** noise and controls are keyword-only. For example,
  `kalman_predict(sys, x, P, Q, u)` becomes
  `kalman_predict(sys, x, P, Q=Q, u=u)`, and update helpers use `R=R`.
  Prediction/update helpers return named bundles with `.x` and `.P`
  (and `.innovation` for updates); tuple unpacking remains supported.
- **Trajectory results:** Kalman/EKF/UKF `.x_hat` and `.P` become `.xs` and
  `.Ps`. RTS `.x_smooth` and `.P_smooth` likewise become `.xs` and `.Ps`.
  Use `.final_state` for terminal estimates, including MHE's former terminal
  `.x_hat` accessor. These names distinguish a trajectory from its last state.
- **Solver configuration:** pass Contrax configuration objects through
  keyword-only `solver=` instead of loose tolerances/iteration/method keywords.
  For example, `place(sys, poles, method="YT", maxiter=50)` becomes
  `place(sys, poles, solver=cx.solvers.PolePlacement(method="YT", max_steps=50))`.
  CARE/DARE/LQR/LQI, Kalman gain design and EKF settings follow their respective
  solver objects. Ordinary `cx.lqr(sys, Q, R)` calls remain unchanged.
- **Analysis return values:** `lyap`, `dlyap`, `ctrb_gramian` and `obsv_gramian`
  return `LyapunovResult`; access the matrix through `.X`. `place` returns
  `PlaceResult`; access the feedback gain through `.K` instead of treating
  the result as an array.
- **Gramian horizons:** omitting the horizon now requests an infinite-horizon
  solve for a strictly stable system, rather than a ten-second integral.
  To retain the finite continuous horizon, use `ctrb_gramian(sys, t=10.0).X`
  or `obsv_gramian(sys, t=10.0).X`. Discrete horizons use `num_steps=`.
- **Linearization:** `linearize` returns an LTI model instead of `(A, B)`;
  use `lin = cx.linearize(model, x0, u0)` followed by `lin.A` and `lin.B`.
  `linearize_ss` is an alias. Discrete models now produce `DiscLTI` with the
  model's sample period; bare vector fields produce `ContLTI`.
- **Bare dynamics callbacks:** `linearize` and recursive estimators use
  `f(t, x, u)` instead of `f(x, u)`. Wrap an autonomous function as
  `lambda t, x, u: old_f(x, u)`. Model callbacks already use `(t, x, u)`;
  bare observation conveniences such as `h(x)` remain supported. Generic
  rollout and MHE callable interfaces retain their documented conventions.
- **MHE calls and indexing:** replace `mhe(f, h, ...)` and
  `mhe_objective(f, h, ...)` with `mhe(f, ..., observation=h)` and
  `mhe_objective(f, ..., observation=h)`, or pass a system model. A window
  contains N candidate states and measurements with N−1 transition inputs,
  rather than N+1 states. MHE noise arguments remain `Q_noise`/`R_noise`.
- **Quadratic penalties:** a matrix `soft_quadratic_penalty` weight is now a
  cost/precision matrix, consistent with scalar weights: `r.T @ W @ r`.
  Previously matrix weights acted as covariances through a solve. Callers
  passing a covariance must convert it to the intended precision/cost weight.
- **Simulation returns:** `lsim` and `simulate` return `SimResult`; existing
  `(ts, xs, ys)` unpacking remains supported. For discrete simulation, `.ts`
  aligns with T outputs, while `.state_times` aligns with T+1 states. Use
  `result.state_times` when plotting the full state trajectory.
- **Numerical defaults:** linear Kalman and EKF use Joseph covariance updates;
  choose `Kalman(covariance_update="compact")` or the corresponding `EKF`
  configuration explicitly for compact updates. Numerical solver failure
  defaults to `throw=False`: inspect `.status`/`.successful`, or request
  `throw=True` for checked errors. Invalid shapes/configuration still raise
  immediately. Precision-sensitive design calls require x64 mode; filters
  preserve their input precision.

### Added

- Public `contrax.solvers` configuration objects and consistent numerical status
  and success diagnostics, with checked failures available through `throw=True`.
- Continuous LTI `lsim` with ZOH/FOH inputs on uniform and arbitrary time grids.
- General finite invariant zeros, discrete and infinite-horizon Gramians, and
  binary-composition `quadratic_rollout_cost`.
- Explicit steady-state and information-space Kalman filtering options.

### Changed

- Standardized Kalman/EKF/UKF calls around observations and keyword-only noise,
  controls and initialization; estimator trajectories use `xs` and `Ps` with
  explicit terminal-state accessors.
- Moved numerical configuration into keyword-only `solver=` arguments.
- Defaulted linear/EKF covariance updates to Joseph form and numerical failures
  to returned diagnostics rather than exceptions.
- Aligned MHE to N states/measurements and N−1 transitions; simulation results
  expose state times separately from output times.
- Added implicit forward/reverse Riccati and Lyapunov differentiation, including
  forward-mode ZOH discretization support. Precision-sensitive design routines
  require float64; filters preserve their input precision.
- Expanded reference documentation, runnable examples and API contracts.

### Fixed

- Refined stagnated nonnormal Lyapunov solves within their iteration budget,
  resolving the benchmark's 32-state DARE sensitivity failure without relaxing
  residual tolerances or introducing large automatic dense fallbacks.
- Reduced RTS validation overhead with Cholesky checks and selective spectral
  fallback, retaining singular PSD acceptance and failure propagation.
- Reused uniform-grid hold transitions while preserving independent interval
  derivatives and the arbitrary-grid simulation path.
- Hardened missing-data branches, controls/feedthrough, covariance validation,
  UKF likelihood factor reuse, MHE gradients and convergence diagnostics.
- Corrected state-feedback output maps, model metadata, parameterization
  precision and optional python-control conversion contracts.

### Numerical limitations

- CARE retains its Hamiltonian stable-subspace conditioning limitation;
  refinement and residual/stability checks detect failures but cannot guarantee
  accurate solutions for every ill-conditioned problem.
- Rank-dependent invariant-zero and pole-placement analysis remains design-time
  functionality. Fully singular smoother trajectories can incur additional
  validation overhead before spectral fallback.

## [0.1.0] — 2026

Initial public release.

### Added

**Modeling**
- `ss`, `dss` — continuous and discrete LTI state-space constructors
- `c2d` — ZOH and Tustin discretization with custom VJPs through `_safe_expm`
- `NonlinearSystem`, `nonlinear_system` — generic nonlinear state-space model
- `PHSSystem`, `phs_system` — port-Hamiltonian system with user-supplied H, J, R, G maps
- `linearize`, `linearize_ss` — JAX-native Jacobian linearization at operating points
- `series`, `parallel` — LTI interconnect helpers with operator overloads

**Simulation**
- `lsim` — open-loop discrete simulation via `lax.scan`
- `simulate` — closed-loop simulation for discrete, continuous, nonlinear, and PHS systems
- `step_response`, `impulse_response`, `initial_response` — standard response helpers
- `rollout` — fixed-shape nonlinear rollout compatible with `jit`, `vmap`, and `grad`
- `foh_inputs`, `sample_system`, `as_ode_term` — continuous-to-discrete estimation bridge

**Control design**
- `lqr`, `lqi` — LQR and integral-state augmented LQR via CARE/DARE
- `dare`, `care` — algebraic Riccati solvers with custom VJPs
- `place` — JAX-native KNV0/YT pole placement with Ackermann SISO fallback
- `feedback`, `state_feedback`, `augment_integrator` — closed-loop and augmentation helpers

**Estimation**
- `kalman`, `kalman_gain` — batch Kalman filter and steady-state design
- `kalman_predict`, `kalman_update`, `kalman_step` — online one-step helpers
- `ekf`, `ekf_predict`, `ekf_update`, `ekf_step` — Extended Kalman Filter
- `ukf`, `uks` — Unscented Kalman Filter and Smoother
- `rts` — Rauch–Tung–Striebel smoother
- `mhe_objective`, `mhe`, `mhe_warm_start`, `soft_quadratic_penalty` — fixed-window MHE

**Analysis**
- `poles`, `ctrb`, `obsv` — eigenvalue and structural analysis
- `evalfr`, `freqresp`, `dcgain` — frequency-domain evaluation
- `ctrb_gramian`, `obsv_gramian` — finite-horizon Gramians via Van Loan construction
- `lyap`, `dlyap` — continuous and discrete Lyapunov equation solvers
- `zeros` — transmission zeros via Rosenbrock pencil

**Parameterization**
- `positive_exp`, `positive_softplus`, `lower_triangular` — constrained parameter maps
- `spd_from_cholesky_raw`, `diagonal_spd` — symmetric positive-definite constructions

**Diagnostics**
- `innovation_diagnostics`, `likelihood_diagnostics`, `innovation_rms` — filter health
- `ukf_diagnostics` — UKF sigma-point and covariance checks
- `phs_diagnostics` — PHS structural invariant checks

**PHS helpers**
- `canonical_J`, `schedule_phs`, `partition_state`, `block_observation`, `block_matrix`
- `symmetrize_matrix`, `project_psd`, `phs_to_ss`

**Compatibility**
- `contrax.compat.python_control` — optional bidirectional LTI conversion with `python-control`
