# Solvers

Solver objects configure algorithm choices, tolerances, and iteration limits.
Calls use validated defaults unless a `solver=` is supplied. Diffrax and
Optimistix objects retain their native interfaces in simulation and
optimization.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import contrax as cx

solver = cx.solvers.Lyapunov(method="doubling", rtol=1e-12)
result = cx.dlyap(0.5 * jnp.eye(2), jnp.eye(2), solver=solver)
assert result.successful
```

Close over solver objects in `jax.jit`, or pass them through Equinox transforms.
Model matrices remain differentiable dynamic data. Numerical result statuses
are integer JAX scalars, using `SolverStatus` codes. `throw=False` returns
failure diagnostics; `throw=True` checks the result during execution. Invalid
matrix shapes or algorithm configuration raise immediately.

`DARE` configures structured doubling; `CARE` configures Hamiltonian-subspace
design with accepted Newton corrections. `lqr` and `lqi` accept the solver for
their system's time domain. `kalman_gain` accepts a `DARE` object for the dual
design equation. All these return `.status` and `.successful` in eager and
compiled execution. They require float64 mode and use implicit JVP/VJP rules.

`Kalman` and `EKF` configure covariance updates. Both default to Joseph form;
compact form requires `covariance_update="compact"`. `EKF(num_iter=3)` enables
iterated measurement linearization in online and batch estimators. Filtering
supports float32 and float64 inputs and preserves their precision.

`SteadyStateKalman(design)` explicitly reuses a `kalman_gain` result. Its gain
and covariance are differentiable data. The initial predicted covariance
defaults to `design.P`; a supplied `P0` must match it. The batch checks the
fixed-point equations for the supplied `A`, `C`, `Q`, `R` and the estimator's
stability. An inconsistent design or a missing observation reports
`invalid_problem`. Use covariance-form filtering for varying covariances or
missing observations. Design creation still requires float64 mode.

`InformationKalman()` accepts diagonal observation variances as `R.shape=(p,)`.
It whitens the prior and observations, then factors an `n`-by-`n` matrix. It
avoids forming the prior inverse or a dense innovation covariance, and is useful
when observations outnumber states. Predicted covariances at observed steps
must be positive definite and all variances positive. Controls, masks,
initialization defaults and differentiation follow ordinary Kalman filtering.

`UKF(alpha=1, beta=2, kappa=0)` configures the scaled unscented transform.
Sigma-point and innovation covariances must be positive definite. Missing
observations skip the measurement transform and contribute zero log likelihood.
The innovation Cholesky factor is reused for the gain and likelihood.

`Lyapunov` supports stable doubling, general dense solves, and bounded automatic
fallback. See [Analysis](analysis.md) and [Control](control.md) for equation
assumptions, residual criteria and transform behavior.

`PolePlacement` configures the design-time KNV0/YT method, update limit,
determinant-improvement tolerance, achieved-pole tolerances and conditioning
limit. `fallback_limit=8` bounds SISO Ackermann fallback; zero disables it.
Pole matching preserves multiplicities and conjugate pairs. Configuration
errors raise immediately; convergence and numerical failures return status
diagnostics unless `throw=True`. Read the [control reference](control.md) for
its float64 and transform contract.

::: contrax.solvers
