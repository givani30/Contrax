# Discretization And Linearization

Discretization and linearization are the bridge between nonlinear plant models
and the most mature control-design slice in Contrax.

They are also where “JAX-native” starts to matter for model construction, not
just for solvers.

## Where They Fit

<figure class="contrax-figure">
  <img src="/assets/images/linearize-lqr-pipeline.svg"
       alt="Pipeline from nonlinear model through linearize, ContLTI, c2d, DiscLTI, and LQR design" />
  <figcaption>
    <strong>The key bridge:</strong> `linearize()` turns a nonlinear model
    into a local `ContLTI`, and `c2d()` carries that local model into the most
    mature discrete controller-design path.
  </figcaption>
</figure>

That flow matters because Contrax’s strongest controller-design path is
still the discrete Riccati slice around `dare()`.

## Linearization

`linearize()` computes `A`, `B`, `C`, and `D` Jacobians around an operating
point and returns them as a `ContLTI`. `linearize_ss` is an alias for
`linearize`.

The key design choice is that these are direct JAX differentiation helpers, not
symbolic or offline preprocessing steps. That makes vmapped operating-point
sweeps and compiled local-model construction part of the intended workflow.

Use:

- `linearize(f, x0, u0)` for plain dynamics `(t, x, u) → x_dot`; defaults to
  full-state output
- `linearize(f, x0, u0, output=h)` when you also have an output map
  `h(x, u) → y`
- `linearize(sys, x0, u0)` when you have a reusable `NonlinearSystem` or
  `PHSSystem`

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

- `linearize()` (and its alias `linearize_ss()`) produces a local approximation,
  not a global model guarantee
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
