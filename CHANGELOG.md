# Changelog

All notable changes to Contrax are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
