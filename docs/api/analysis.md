# Analysis

Use the analysis namespace to inspect controllability, observability, poles,
transfer response, Gramians, Lyapunov solutions, and finite invariant zeros.

The namespace includes:

- `ctrb()` and `obsv()` for controllability and observability matrices
- `poles()` for eigenvalue inspection
- `evalfr()`, `freqresp()`, and `dcgain()` for transfer evaluation
- `ctrb_gramian()` and `obsv_gramian()` for continuous/discrete, finite/infinite Gramians
- `lyap()` and `dlyap()` for continuous and discrete Lyapunov equation solvers
- `zeros()` for finite invariant zeros of square or rectangular LTI systems
- `quadratic_rollout_cost()` for exact finite-horizon discrete feedback costs

## Minimal Example

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

sys = cx.ss(
    jnp.array([[0.0, 1.0], [-2.0, -0.4]]),
    jnp.array([[0.0], [1.0]]),
    jnp.array([[1.0, 0.0]]),
    jnp.zeros((1, 1)),
)

poles = cx.poles(sys)
dc = cx.dcgain(sys)
Wc = cx.ctrb_gramian(sys, t=2.0).X
```

## Conventions

- `poles(sys)` returns the system eigenvalues
- `evalfr(sys, point)` evaluates the transfer map at one complex point
- `freqresp(sys, omega)` evaluates over a frequency grid
- `dcgain(sys)` evaluates the zero-frequency or steady-state gain

The transfer helpers evaluate state-space realizations directly.

## Transform Behavior

Array-based analysis helpers compose with `jit` and batching when the
underlying linear algebra operation supports those transformations.

Gramian helpers support `jit`, `vmap`, JVP and VJP. Discrete finite horizon
counts are static Python integers. Continuous finite horizons use a matrix
exponential; infinite horizons differentiate the Lyapunov equation implicitly.

## Gramians and Horizons

With no horizon argument, `ctrb_gramian(sys)` and `obsv_gramian(sys)` solve the
stable infinite-horizon equation. The default uses Lyapunov doubling. Both
continuous and discrete systems must be strictly stable for an infinite-horizon
Gramian; a marginal or unstable model returns `invalid_problem` even if a
general dense Lyapunov equation could have a finite solution.

Specify `t=...` for a continuous integration horizon, or `num_steps=...` for a
discrete sum over k=0,...,N−1. The arguments are distinct to avoid confusing
seconds and step counts. Finite horizons support marginal and unstable models;
zero returns the zero Gramian. Discrete sums have per-step weighting and do not
include a sample-time factor. Controllability sums use A^k B B.T (A^k).T;
observability sums use (A^k).T C.T C A^k.

Both functions return `LyapunovResult`: the Gramian is `.X`, with `.status`,
`.successful` and `.residual_norm` diagnostics. `throw=True` checks failure in
eager or compiled execution. The finite-horizon residual checks the corresponding
Gramian equation with its terminal forcing term. For finite discrete horizons,
`.num_steps` is the requested term count; infinite solves report doubling steps.

Select an infinite-horizon equation algorithm using
`solver=cx.solvers.Lyapunov(...)`. Finite discrete horizons use binary
composition; finite continuous horizons use Van Loan's block exponential.
Very long continuous horizons can overflow the block exponential even for a
stable model; use the infinite-horizon solve when that is the intended quantity.

## Lyapunov Equations

`lyap(A, Q)` solves the continuous Lyapunov equation $A X + X A^\top + Q = 0$.
`dlyap(A, Q)` solves the discrete form $A X A^\top - X + Q = 0$.

Both return `LyapunovResult`; use `.X` for the matrix and `.successful` or
`.status` to check the solve. The default uses squared Smith iteration for
stable systems and validates the equation residual. Automatic dense fallback
is restricted to at most eight states. It supports small nonstable equations;
for larger nonstable equations select the dense solver explicitly:

```python
result = cx.dlyap(A, Q, solver=cx.solvers.Lyapunov(method="dense"))
X = result.X
```

Dense solves need O(n^4) memory and O(n^6) work. The stable doubling route uses
O(n^2) storage and O(n^3) work per step. For continuous equations, a Cayley
transform supplies the discrete equation. `method="doubling"` requires a
stable state matrix and reports `invalid_problem` otherwise.

Both equations require float64 mode and real square matrices. The solution is
unique when no eigenvalue sum is zero (continuous) or product is one
(discrete). This condition also permits nonstable systems. Forward and reverse
derivatives solve the implicit matrix equation; they do not differentiate
through the stopping loop. Use `vmap` for independent equations.

`throw=False` returns failure diagnostics; `throw=True` raises through Equinox
checks in eager or compiled execution. Singular equations and nonconvergent
iterations must be handled before using their solutions. The returned residual
is a Frobenius norm; success requires it to be at most
`atol + rtol * scale`, where `scale` sums norms of the individual equation
terms. This bounds backward error; it does not bound forward error for an
ill-conditioned equation.

## Finite Invariant Zeros

`zeros(sys)` computes finite rank drops of the Rosenbrock system pencil
relative to its normal rank. It accepts square and rectangular realizations,
including zero, singular and full-rank feedthrough matrices. For a minimal
realization these are transmission zeros. Nonminimal realizations can include
decoupling zeros absent from the minimal transfer realization.

For square invertible $D$, the finite pencil reduces to
$A - B D^{-1} C - \lambda I$.

The general algorithm uses SVD-based orthogonal staircase reductions of the
algebraic output constraints, followed by removal of right singular structure.
The remaining finite pencil has identity coefficient for the spectral
parameter and uses `jnp.linalg.eigvals`. SciPy and SLICOT are offline references.
The formulation follows the finite-pencil reduction described by
[SLICOT AB08ND](https://www.slicot.org/objects/software/shared/doc/AB08ND.html).
Validation includes its rectangular nonminimal example and the transposed
system, as well as independent generalized eigenvalues for regular square
pencils.

Zero computation uses host-side rank decisions and requires float64 precision;
it cannot be jitted, vmapped, or differentiated. Tolerances determine numerical
rank, so check sensitivity to `rtol`, `atol`, and realization scaling near rank
transitions. The result contains finite zeros; it omits infinite zeros and
Kronecker indices.

## Quadratic Feedback Costs

`quadratic_rollout_cost(sys, K, Q, R, x0, num_steps=N)` computes

$$
\sum_{k=0}^{N-1} (x_k^\top Q x_k + u_k^\top R u_k)
+ x_N^\top Q_{\mathrm{terminal}} x_N,\qquad u_k=-Kx_k.
$$

Omit `Q_terminal` for zero terminal cost. `Q` and `R` weight each step
independently of the discrete system's sample time. Binary Gramian
composition uses O(log N) matrix products and O(n²) storage, without retaining
a trajectory. Stability is unnecessary for a finite horizon, but long unstable
horizons may overflow. It returns a scalar for one `(n,)` initial state;
use `vmap` for batches:

```python
cost = cx.quadratic_rollout_cost(
    sys_d, K, Q, R, x0, num_steps=100, Q_terminal=Q,
)
costs = jax.vmap(lambda x: cx.quadratic_rollout_cost(
    sys_d, K, Q, R, x, num_steps=100,
))(initial_states)
```

The horizon is a static nonnegative Python integer. Zero steps return only
the optional initial-state terminal cost. Float32 and float64 inputs are
supported. The cost composes with `jit`, `vmap`, JVP, and VJP. Validation
compares values and gain derivatives with an explicit state/input trajectory
sum.

## Numerical Notes

Finite-horizon Gramians use a Van Loan block-matrix exponential construction
and require `jax_enable_x64=True`.

When comparing realizations, prefer invariant checks such as rank,
conditioning, pole location, and transfer behavior. Exact matrix equality is
only meaningful when the realization basis is intentionally identical.

## Related Pages

- [Systems](systems.md) for the models these helpers inspect
- [Simulation](simulation.md) for time-domain validation alongside frequency-
  domain or structural checks
- [JAX-native workflows](../examples/jax-native-workflows.md) for broader
  end-to-end usage patterns

::: contrax.analysis
