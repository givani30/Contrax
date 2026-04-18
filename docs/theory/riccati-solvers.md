# Riccati Solvers

Riccati solvers sit at the center of Contrax's control story. They are also one
of the places where "JAX-native" really matters, because a solver can look fine
in forward mode but behave badly under `grad`, `jit`, or GPU execution.

## LQR Setup

Riccati solvers are the numerical core behind linear-quadratic regulator
design. In Contrax they matter as both forward solvers and differentiable JAX
primitives.

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
    <strong>The solver story in one picture:</strong> the Riccati solve is not
    an isolated matrix trick; it is the map from plant plus weights to the
    gain and closed-loop diagnostics used by the rest of the control surface.
  </figcaption>
</figure>

## Riccati Equations

Contrax implements the stabilizing algebraic Riccati solves behind LQR:

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

In the docs structure, that means:

- the [Control API](../api/control.md) records the callable contract
- tutorials show the solver inside full workflows
- this page explains why the solver choices look the way they do

## Discrete Riccati: `dare()`

Contrax uses a structured-doubling forward solve for the discrete
algebraic Riccati equation.

At the control level, that means `dare()` is the numerical heart of discrete
`lqr()`: solve for `S`, recover `K_d`, then inspect the poles of
$A_{\mathrm{cl},d}$.

The main goals of that implementation are:

- robust forward solves on the existing benchmark slice
- residual and closed-loop pole validation
- JAX-native execution without CPU-only Schur or QZ routines in the hot path
- a custom VJP that avoids unrolling solver iterations in the backward pass

The backward pass solves the adjoint discrete Lyapunov equation for the
converged Riccati solution and then lets gain computation differentiate from
that solution.

The practical implication is that gradients with respect to `A`, `B`, `Q`, and
`R` are attached to the converged Riccati solution instead of the raw iteration
history.

Validation for `dare()` includes residual checks, closed-loop pole checks,
Octave-backed reference tests, JIT agreement, and finite-difference gradient
checks.

## Continuous Riccati: `care()`

`care()` uses a Hamiltonian stable-subspace solve with an
implicit-differentiation backward pass.

At the control level, `care()` plays the same role for continuous `lqr()`:
solve for `S`, recover `K_c`, then inspect the spectrum of
$A_{\mathrm{cl},c}$.

Reference: Laub (1979), "A Schur Method for Solving Algebraic Riccati
Equations". Contrax follows the same stable-subspace idea but uses a JAX-native
eigendecomposition path rather than a Schur-based LAPACK routine in the hot
path.

That makes continuous `lqr()` a supported design path, but
the solver maturity is still lower than `dare()`:

- the forward solve validates the Hamiltonian stable-subspace split and checks
  the CARE residual before returning
- the implementation has Octave-reference tests, JIT agreement tests, and
  finite-difference gradient checks
- the benchmark slice is smaller
- broader conditioning diagnostics are still needed
- Newton-Kleinman polishing may still prove useful on harder systems

## Solver Selection And Schur-Based Methods

In classical control software, Schur- or QZ-based methods are standard. In
Contrax, the issue is not mathematical legitimacy. The issue is the execution
story:

- CPU-only decomposition paths are a bad fit for the intended JAX/GPU story
- a solver path that is numerically fine in forward mode may be a poor fit for
  differentiation
- unrolling an iterative solver in the backward pass is the wrong memory story

That is why Contrax cares about both the forward algorithm and the backward
contract.

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

For the public discrete path, also treat benchmark coverage as part of solver
maturity rather than as a separate research exercise.

## JAX Behavior

Both Riccati solvers are written to stay inside the JAX execution model:

- `dare()` uses a JAX-native structured-doubling forward solve plus a custom VJP
- `care()` uses a JAX-native Hamiltonian eigendecomposition plus an implicit
  custom VJP

That makes both paths suitable for compiled controller-design objectives, with
the usual caveat that conditioning and benchmark coverage still matter.

## Related Pages

- [Systems API](../api/systems.md) for `ContLTI` and `DiscLTI`
- [Differentiable LQR](../tutorials/differentiable-lqr.md)
- [Continuous LQR](../tutorials/continuous-lqr.md)
- [JAX transform contract](jax-transform-contract.md)
- [Control API](../api/control.md)
