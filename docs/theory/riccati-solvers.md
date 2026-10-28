# Riccati Solvers

Riccati solvers turn infinite-horizon LQR problems into algebraic equations.
Their implementation must also support the required JAX compilation and
differentiation paths.

## LQR Setup

Riccati solvers are the numerical core behind linear-quadratic regulator
design. They provide both the forward solution and the differentiable map used
inside controller-design objectives.

For the discrete-time model

$$
x_{k+1} = A x_k + B u_k
$$

the infinite-horizon cost is

$$
J_d = \sum_{k=0}^{\infty} \left(x_k^\top Q x_k + u_k^\top R u_k\right)
$$

and the optimal state-feedback law has the form

$$
u_k = -K_d x_k
$$

For the continuous-time model

$$
\dot{x} = A x + B u
$$

the cost is

$$
J_c = \int_0^\infty \left(x(t)^\top Q x(t) + u(t)^\top R u(t)\right)\,dt
$$

with optimal feedback

$$
u(t) = -K_c x(t)
$$

The Riccati equations are what turn those optimization problems into algebraic
solver calls.

<figure class="contrax-figure">
  <img src="/assets/images/lqr-riccati-workflow.svg"
       alt="Riccati-backed LQR design workflow from plant and weights to dare or care, gain recovery, and closed-loop checks" />
  <figcaption>
    <strong>Riccati design:</strong> map the plant and cost weights to a gain,
    then inspect the closed-loop diagnostics.
  </figcaption>
</figure>

## Riccati Equations

The stabilizing algebraic Riccati equations used by LQR are:

$$
A^\top S A - S - A^\top S B \left(R + B^\top S B\right)^{-1} B^\top S A + Q = 0
$$
for the discrete algebraic Riccati equation, and

$$
A^\top S + S A - S B R^{-1} B^\top S + Q = 0
$$
for the continuous algebraic Riccati equation.

Given the stabilizing solution `S`, the gains are

$$
K_d = \left(R + B^\top S B\right)^{-1} B^\top S A
$$

$$
K_c = R^{-1} B^\top S
$$

and the closed-loop matrices are

$$
A_{\mathrm{cl},d} = A - B K_d, \qquad
A_{\mathrm{cl},c} = A - B K_c
$$

## Solver Paths

The public paths are:

- `dare()` for discrete systems, used by `lqr()` on `DiscLTI`
- `care()` for continuous systems, used by `lqr()` on `ContLTI`

Their maturity differs:

- `dare()` is the most mature solver path
- `care()` is a validated continuous-time solver, but still less benchmarked
  than `dare()`

See the [Control API](../api/control.md) for call signatures and the tutorials
for complete design workflows.

## Discrete Riccati: `dare()`

`dare()` solves the discrete algebraic Riccati equation with structured
doubling.

At the control level, that means `dare()` is the numerical heart of discrete
`lqr()`: solve for `S`, recover `K_d`, then inspect the poles of
$A_{\mathrm{cl},d}$.

The solver provides:

- robust forward solves on the existing benchmark slice
- residual and closed-loop pole validation
- JAX-native execution without CPU-only Schur or QZ routines in the hot path
- implicit JVP and VJP rules that avoid unrolling solver iterations

The backward pass solves the adjoint discrete Lyapunov equation for the
converged Riccati solution and then lets gain computation differentiate from
that solution. Both Riccati adjoints share the public Lyapunov equation kernel:
residual-validated squared Smith iteration, with a Cayley transform for the
continuous equation. Automatic dense fallback is limited to eight states. For
stable nonnormal systems, residual corrections refine a stagnated doubling
solution within the same iteration budget and without relaxing its tolerance.
The reported Lyapunov iteration count includes these correction solves. For
larger systems, failed iterations return nonfinite gradients; they do not
switch to a dense Kronecker solve. The public Lyapunov routines also expose
implicit forward-mode differentiation and an explicit general dense solver.

The practical implication is that gradients with respect to `A`, `B`, `Q`, and
`R` are attached to the converged Riccati solution instead of the raw iteration
history.

Validation for `dare()` includes residual checks, closed-loop pole checks,
Octave-backed reference tests, JIT agreement, and finite-difference gradient
checks.

## Continuous Riccati: `care()`

`care()` uses a Hamiltonian stable-subspace solve with residual-decreasing
Newton corrections and implicit forward/reverse differentiation.

At the control level, `care()` plays the same role for continuous `lqr()`:
solve for `S`, recover `K_c`, then inspect the spectrum of
$A_{\mathrm{cl},c}$.

Reference: Laub (1979), "A Schur Method for Solving Algebraic Riccati
Equations". The implementation uses a JAX eigendecomposition to extract the
stable subspace.

Newton corrections solve a continuous Lyapunov equation for the CARE residual.
A correction is accepted only if its residual decreases and the resulting
closed loop remains stable. The result validates cost definiteness, finite
values, residual and stability in JAX in eager and compiled execution.
`num_steps` counts accepted corrections. Poor separation of stable and unstable
Hamiltonian eigenvalues can still prevent an accurate candidate; inspect status
instead of assuming refinement guarantees success.

## Solver Selection And Schur-Based Methods

Schur- and QZ-based methods are standard in classical control software. The
JAX implementation uses methods that fit its execution model:

- CPU-only decomposition paths do not run within compiled JAX and GPU workflows
- a solver path that is numerically fine in forward mode may be a poor fit for
  differentiation
- unrolling an iterative solver in the backward pass can require excessive
  memory

Both the forward algorithm and the derivative rule determine whether a solver
fits a compiled design workflow.

## Failure Modes And Diagnostics

The first signs of trouble on unfamiliar systems are usually:

- large Riccati residuals
- non-stabilizing closed-loop poles
- failure to isolate the required stable Hamiltonian subspace
- poor agreement with reference solvers on representative systems
- unstable or non-finite gradients through small design objectives

## How to Validate a Riccati Solve

On unfamiliar systems, the minimum useful checks are:

- Riccati residual size
- closed-loop stability
- agreement with Octave or SciPy on representative reference systems
- finite gradients through small objectives involving `Q`, `R`, `A`, or `B`

Benchmark coverage is part of assessing solver maturity.

## JAX Behavior

Both Riccati solvers are written to stay inside the JAX execution model:

- `dare()` uses a JAX-native structured-doubling forward solve
- `care()` uses a JAX-native Hamiltonian eigendecomposition and Newton refinement
- both use a custom JVP for the implicit Riccati equation; JAX transposes its
  Lyapunov solve for reverse differentiation

Gain and Riccati-solution Hessians have analytic scalar reference coverage,
including a composed `c2d -> lqr` objective. Pole diagnostics are evaluated on
access, so they do not enter those objectives. Differentiating poles themselves
has the separate restrictions of JAX's nonsymmetric eigensolver.

Both paths can appear in compiled controller-design objectives; conditioning
and benchmark coverage determine where they are suitable.

## Related Pages

- [Systems API](../api/systems.md) for `ContLTI` and `DiscLTI`
- [Differentiable LQR](../tutorials/differentiable-lqr.md)
- [Continuous LQR](../tutorials/continuous-lqr.md)
- [JAX transform contract](jax-transform-contract.md)
- [Control API](../api/control.md)
