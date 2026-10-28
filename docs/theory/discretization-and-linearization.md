# Discretization And Linearization

Linearize nonlinear dynamics at an operating point, then discretize the local
model for discrete-time controller design. JAX derivatives support model
construction and batching over operating points.

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

The discrete Riccati path through `dare()` has the broadest benchmark coverage.

## Linearization

`linearize()` computes `A`, `B`, `C`, and `D` Jacobians around an operating
point and returns them as a `ContLTI`. `linearize_ss` is an alias for
`linearize`.

JAX differentiation computes the model Jacobians, so compiled local-model
construction and vmapped operating-point sweeps use the same code path.

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

## Discretization

`c2d()` converts a `ContLTI` to a `DiscLTI`.

The public methods are:

- `zoh`: zero-order hold, recommended for differentiable and reference-checked
  workflows
- `tustin`: bilinear transform for applications that call for that mapping

The `zoh` path has broader derivative and reference coverage.

## Zero-Order Hold Path

Zero-order-hold discretization uses JAX's native matrix exponential and its
forward/reverse derivatives. It supports `jacfwd`, `jacrev` and higher-order
workflows without a reverse-only custom rule. Stiff nonnormal regression cases
compare both derivative directions with independent SciPy Frechet derivatives;
scalar ZOH Jacobians and Hessians have closed-form reference checks.

Like any scaling-and-squaring computation, numerical range and conditioning
limit usable inputs. Large matrix norms can exceed the exponential's squaring
budget; a badly conditioned exponential or derivative remains sensitive to
roundoff. Enable float64 before constructing precision-sensitive design data.

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

- `linearize()` (and its alias `linearize_ss()`) approximates behavior near the
  chosen operating point
- `c2d()` is precision-sensitive and requires `jax_enable_x64=True`
- `zoh` is the path to prefer for serious workflows today
- `tustin` provides the bilinear frequency mapping

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
