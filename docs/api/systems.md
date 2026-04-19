# Systems

Systems is the public modeling namespace in Contrax. It brings together the
objects and helpers you use to represent dynamics before you design a
controller, run an estimator, or simulate a trajectory.

The namespace includes:

- continuous and discrete LTI models:
  [`ContLTI`][contrax.systems.ContLTI],
  [`DiscLTI`][contrax.systems.DiscLTI], `ss()`, `dss()`
- nonlinear and port-Hamiltonian model objects:
  [`NonlinearSystem`][contrax.systems.NonlinearSystem],
  [`PHSSystem`][contrax.systems.PHSSystem], `nonlinear_system()`,
  `phs_system()`, `schedule_phs()`
- PHS structure helpers: `canonical_J()` to define the symplectic structure
  map, `phs_to_ss()` to extract the local state-space linearization around an
  operating point
- structured-system helpers: `partition_state()`, `block_observation()`,
  `block_matrix()`, `symmetrize_matrix()`, `project_psd()`,
  `phs_diagnostics()`
- discretization and local model construction: `c2d()`, `linearize()`,
  `linearize_ss()`
- simple state-space composition: `series()`, `parallel()`, plus operator
  overloads on [`ContLTI`][contrax.systems.ContLTI] and
  [`DiscLTI`][contrax.systems.DiscLTI]

## Minimal Example

Use the systems namespace when you want the model-building story directly:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax.systems as cxs

sys_c = cxs.ss(
    jnp.array([[0.0, 1.0], [0.0, 0.0]]),
    jnp.array([[0.0], [1.0]]),
    jnp.eye(2),
    jnp.zeros((2, 1)),
)
sys_d = cxs.c2d(sys_c, dt=0.05)
```

## Canonical Model Forms

<div class="contrax-equation-grid" markdown="1">

<div class="contrax-equation-card" markdown="1">

### LTI And Nonlinear Systems

The LTI and generic nonlinear contracts are the broad modeling surface.

$$
\dot{x} = A x + B u, \qquad y = C x + D u
$$

$$
x_{k+1} = A x_k + B u_k, \qquad y_k = C x_k + D u_k
$$

$$
\dot{x} = f(t, x, u), \qquad y = h(t, x, u)
$$

</div>

<div class="contrax-equation-card" markdown="1">

### Port-Hamiltonian Systems

[`PHSSystem`][contrax.systems.PHSSystem] is the structured nonlinear family in
Contrax.

$$
\dot{x} = \bigl(J - R(x)\bigr)\nabla H(x) + G(x)u
$$

For scheduled coefficients, `schedule_phs()` binds an exogenous context into
that canonical model:

$$
\dot{x} = \bigl(J - R(t, x, \theta(t))\bigr)\nabla H(x) + G(t, x, \theta(t))u
$$

</div>

</div>

## Conventions

- `A`: state matrix with shape `(n, n)`
- `B`: input matrix with shape `(n, m)`
- `C`: output matrix with shape `(p, n)`
- `D`: feedthrough matrix with shape `(p, m)`
- [`DiscLTI.dt`](#contrax.systems.DiscLTI.dt): scalar sample time stored as an
  array leaf

For nonlinear models, the reusable model-object contract is:

- `dynamics(t, x, u)` for the transition or vector field
- `output(t, x, u)` for the output or measurement map

Contrax keeps continuous and discrete LTI systems on the same field names so
code can move between them with minimal ceremony.

For [`PHSSystem`][contrax.systems.PHSSystem], the base library contract is
intentionally narrower:

- `H(x)` is the storage function
- `R(x)` is the dissipation map when present
- `G(x)` is the input map when present
- `J(x)` defaults to the canonical symplectic structure unless you override it
- `schedule_phs()` is the extension path when dissipation or input maps depend
  on observed scheduling context
- `schedule_phs()` can also bind scheduled structure maps `J(t, x, context)`
  when the structure itself varies with observed context

## Structured Helpers

Contrax now includes a small generic helper layer around structured nonlinear
models so downstream repos do not need their own mini utility module just to
express common PHS patterns.

Use:

- `partition_state(x, block_sizes)` to split a structured state into
  consecutive blocks
- `block_observation(block_sizes, block_indices)` to build partial-state output
  maps from those blocks
- `block_matrix(row_block_sizes, col_block_sizes, blocks)` to assemble dense
  structured input, process, or observation maps from block entries
- `symmetrize_matrix(M)` and `project_psd(M)` when dissipation or covariance
  matrices pick up small numerical asymmetry or PSD drift
- `phs_diagnostics(sys, x, u)` for local skew-symmetry, dissipation, and
  instantaneous power-balance diagnostics

These are intentionally lightweight helpers, not a large structured-model
framework. They are there to make the public PHS layer ergonomic and reusable.

## Port-Hamiltonian Structure Helpers

`canonical_J(n)` returns the $n \times n$ canonical symplectic structure
matrix for even $n$:

$$
J_c = \begin{bmatrix} 0 & I_{n/2} \\ -I_{n/2} & 0 \end{bmatrix}
$$

Use it when building a [`PHSSystem`][contrax.systems.PHSSystem] whose
structure map is standard symplectic rather than a custom dissipative one.

`phs_to_ss(sys, x_eq, u_eq)` linearizes a PHS at an operating point and
returns a [`ContLTI`][contrax.systems.ContLTI]. This is the standard bridge
from a structured nonlinear PHS model into the linear control design stack
(`lqr`, `place`, `kalman_gain`, etc.):

```python
lti = cx.phs_to_ss(phs_sys, x_eq=jnp.zeros(4), u_eq=jnp.zeros(1))
design = cx.lqr(cx.c2d(lti, dt=0.01), Q, R)
```

## Linearization And Discretization

`linearize()` is the main JAX-native bridge from nonlinear plant code into the
linear control stack (`linearize_ss` is an alias).

Use:

- `linearize(f, x0, u0)` when you have a plain dynamics callable with
  signature `(t, x, u) → x_dot`; returns a
  [`ContLTI`][contrax.systems.ContLTI] with full-state output by default
- `linearize(f, x0, u0, output=h)` when you also want a specific output map
  `h(x, u) → y`; `A`, `B`, `C`, `D` are all computed and packed into the
  returned [`ContLTI`][contrax.systems.ContLTI]
- `linearize(sys, x0, u0)` when you have a reusable
  [`NonlinearSystem`][contrax.systems.NonlinearSystem] or
  [`PHSSystem`][contrax.systems.PHSSystem]; the system's own `dynamics` and
  `output` callables are used directly


The local linear model is the usual first-order expansion around an operating
point:

$$
\delta \dot{x} \approx A\,\delta x + B\,\delta u, \qquad
\delta y \approx C\,\delta x + D\,\delta u
$$

with

$$
A = \left.\frac{\partial f}{\partial x}\right|_{(x_0, u_0)}, \quad
B = \left.\frac{\partial f}{\partial u}\right|_{(x_0, u_0)}, \quad
C = \left.\frac{\partial h}{\partial x}\right|_{(x_0, u_0)}, \quad
D = \left.\frac{\partial h}{\partial u}\right|_{(x_0, u_0)}
$$

Then use `c2d()` when the downstream workflow is discrete-time design or
simulation.

The `c2d()` methods are:

- `zoh`: the stronger path, with a custom VJP around the matrix
  exponential
- `tustin`: a convenience bilinear transform

<figure class="contrax-figure">
  <img src="/assets/images/linearize-lqr-pipeline.svg"
       alt="Pipeline from nonlinear dynamics through linearization, discretization, LQR design, and closed-loop simulation" />
  <figcaption>
    <strong>The main model-preparation path:</strong> start with a nonlinear or
    structured model, build a local <a href="#contrax.systems.ContLTI"><code>ContLTI</code></a>
    with <code>linearize()</code>, discretize it with <code>c2d()</code>,
    then move into controller design and validation.
  </figcaption>
</figure>

## Interconnection

Simple state-space composition lives in the same public namespace because
it is part of the model-building story.

Use:

- `series(sys2, sys1)` or `sys2 @ sys1` for feedforward composition
- `parallel(sys1, sys2)` or `sys1 + sys2` for summed-output composition
- `sys1 - sys2` for parallel subtraction

Mixed continuous/discrete interconnections are intentionally unsupported, and
discrete interconnections require matching `dt`.

State ordering is explicit:

- `series(sys2, sys1)` puts `sys1` states first, then `sys2`
- `parallel(sys1, sys2)` puts `sys1` states first, then `sys2`

## Transform Behavior

The system containers themselves are pytrees, so they work naturally with
`jit` and `vmap` when shapes line up. That matters because the intended
Contrax workflows include vmapped linearization, batched controller design, and
compiled design-and-simulate objectives.

The important user-facing transform contracts here are:

- `linearize()` and `linearize_ss()` are designed to compose with `jit` and
  `vmap`
- [`DiscLTI.dt`](#contrax.systems.DiscLTI.dt) stays as an array leaf rather
  than static metadata, which makes batching over systems more natural
- composition helpers are ordinary state-space algebra once family and shape
  checks pass

## Numerical Notes

Contrax does not enable float64 globally on import. Precision-sensitive system
helpers such as `c2d()` require `jax_enable_x64=True`.

For nonlinear local models, remember that `linearize()` and `linearize_ss()`
produce local behavior around one operating point; they are not a guarantee of
global model fidelity.

The structured-system diagnostics are local algebraic checks. They are useful
for sanity checking model definitions and recursive-estimation plumbing, but
they are not a formal certificate of passivity or robustness.

## Related Pages

- [Simulation](simulation.md) for `lsim()`, `simulate()`, and response helpers
- [Control](control.md) for LQR, Riccati solves, and state feedback
- [Discretization and linearization](../theory/discretization-and-linearization.md)
  for the explanation page behind `c2d()` and `linearize_ss()`
- [Linearize, LQR, simulate](../tutorials/linearize-lqr-simulate.md) for an
  end-to-end workflow

::: contrax.systems
