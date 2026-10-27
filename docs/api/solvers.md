# Solvers

Contrax solver objects configure algorithm choices, tolerances and iteration
limits. Ordinary calls use validated defaults; pass a keyword-only `solver=`
to choose another algorithm. Diffrax and Optimistix objects retain their native
interfaces in simulation and optimization.

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
preserves input precision and does not require float64 mode.

`UKF(alpha=1, beta=2, kappa=0)` configures the scaled unscented transform.
Sigma-point covariances and innovation covariances must be positive definite;
no diagonal jitter is added. Missing observations skip the entire measurement
transform and contribute zero log likelihood. The innovation Cholesky factor
is reused for the gain and likelihood.

```python
design = cx.lqr(sys, Q, R, solver=cx.solvers.DARE(max_steps=64))
```

`Lyapunov` supports stable doubling, general dense solves, and bounded automatic
fallback. See [Analysis](analysis.md) and [Control](control.md) for equation
assumptions, residual criteria and transform behavior.

::: contrax.solvers
