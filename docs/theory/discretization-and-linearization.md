# Discretization And Linearization

Discretization and linearization are the bridge between nonlinear plant models
and the most mature control-design slice in Contrax.

They are also where “JAX-native” starts to matter for model construction, not
just for solvers.

## Where They Fit

<figure class="contrax-figure">
  <img src="/assets/images/linearize-lqr-pipeline.svg"
       alt="Pipeline from nonlinear model through linearize_ss, ContLTI, c2d, DiscLTI, and LQR design" />
  <figcaption>
    <strong>The key bridge:</strong> `linearize_ss()` turns a nonlinear model
    into a local `ContLTI`, and `c2d()` carries that local model into the most
    mature discrete controller-design path.
  </figcaption>
</figure>

That flow matters because Contrax’s strongest controller-design path is
still the discrete Riccati slice around `dare()`.

## Linearization

`linearize()` computes local Jacobians `A` and `B` around an operating point.
`linearize_ss()` computes `A`, `B`, `C`, and `D` and returns them as a
`ContLTI`.

The key design choice is that these are direct JAX differentiation helpers, not
symbolic or offline preprocessing steps. That makes vmapped operating-point
sweeps and compiled local-model construction part of the intended workflow.

Use:

- `linearize(f, x0, u0)` when you only need the dynamics Jacobians
- `linearize_ss(f, x0, u0, output=h)` when you have plain callables
- `linearize_ss(sys, x0, u0)` when you want a state-space model from a
  reusable `NonlinearSystem` or `PHSSystem`

The local approximation is the usual first-order model around an operating
point `(x_0, u_0)`:

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

Contrax computes those Jacobians directly with JAX automatic differentiation,
so operating-point sweeps and vmapped local-model construction are part of the
intended workflow rather than separate preprocessing steps.

## Discretization

`c2d()` converts a `ContLTI` to a `DiscLTI`.

The public methods are:

- `zoh`: zero-order hold, the stronger path
- `tustin`: bilinear transform, mainly a convenience discretization

The `zoh` path matters more because it is the one intended for differentiable
and reference-checked design workflows.

## Zero-Order Hold Path

The zero-order-hold discretization goes through the matrix exponential. In
plain forward mode that is straightforward, but gradients through the matrix
exponential can become numerically fragile on stiff systems.

That is why Contrax uses a custom VJP around the matrix exponential in this
path: the forward formula alone is not the whole contract. The gradient story
is part of the feature.

For zero-order hold, the continuous-to-discrete map is

$$
\begin{bmatrix}
\Phi & \Gamma \\
0 & I
\end{bmatrix}

=
\exp\left(
\begin{bmatrix}
A & B \\
0 & 0
\end{bmatrix}
\Delta t
\right)
$$

so that the discrete model becomes

$$
x_{k+1} = \Phi x_k + \Gamma u_k
$$

## Caveats

- `linearize()` and `linearize_ss()` are local approximations, not global model
  guarantees
- `c2d()` is precision-sensitive and requires `jax_enable_x64=True`
- `zoh` is the path to prefer for serious workflows today
- `tustin` is useful, but not the headline differentiable discretization path

## Practical Checks

On unfamiliar workflows, the useful checks are:

- linearized shapes match what the downstream control design expects
- vmapped operating-point pipelines stay finite
- discrete closed-loop poles look reasonable after `c2d() -> lqr()`
- gradients through representative `c2d()` objectives stay finite

## Related Pages

- [Systems API](../api/systems.md)
- [Linearize, LQR, simulate](../tutorials/linearize-lqr-simulate.md)
- [JAX transform contract](jax-transform-contract.md)
