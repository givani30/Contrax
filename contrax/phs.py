"""Port-Hamiltonian systems and structure-preserving helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from contrax._model_validation import dimension, sample_period
from contrax.core import ContLTI, DiscLTI
from contrax.nonlinear import NonlinearSystem
from contrax.parameterization import _positive_inputs
from contrax.types import PHSStructureDiagnostics


def canonical_J(n_dofs: int, *, dtype: jnp.dtype | None = None) -> Array:
    """Construct the (2n, 2n) canonical structure for a positive static integer n."""
    n_dofs = dimension(n_dofs, "n_dofs", minimum=1)
    if n_dofs is None:
        raise ValueError("n_dofs must be positive.")
    dtype = jnp.float32 if dtype is None else dtype
    eye = jnp.eye(n_dofs, dtype=dtype)
    zeros = jnp.zeros((n_dofs, n_dofs), dtype=dtype)
    return jnp.block([[zeros, eye], [-eye, zeros]])


def _zero_input_matrix(x: Array, u: Array) -> Array:
    return jnp.zeros((x.shape[0], u.shape[0]), dtype=jnp.result_type(x.dtype, u.dtype))


class PHSSystem(eqx.Module):
    """Port-Hamiltonian system with user-supplied storage and structure maps."""

    H: Callable[[Array], Array]
    R: Callable[[Array], Array] | None = None
    G: Callable[[Array], Array] | None = None
    J: Callable[[Array], Array] | None = None
    observation: Callable | None = None
    dt: Array | None = None
    # Dimension metadata. static=True is appropriate here because these are
    # structural integers, not array leaves. Consequence: vmapping over a batch
    # of PHSSystem instances requires all systems in the batch to share the same
    # (state_dim, input_dim, output_dim) values or JAX will retrace per shape.
    state_dim: int | None = eqx.field(static=True, default=None)
    input_dim: int | None = eqx.field(static=True, default=None)
    output_dim: int | None = eqx.field(static=True, default=None)

    def dynamics(self, t: Array | float, x: Array, u: Array) -> Array:
        """Evaluate the port-Hamiltonian vector field."""
        del t  # State-only structure maps; use schedule_phs for time dependence.
        dim = x.shape[0]
        grad_H = jax.grad(self.H)(x)
        if self.J is None:
            if dim == 0 or dim % 2:
                raise ValueError(
                    "Canonical PHS states must have positive even dimension"
                )
            # Canonical J @ grad(H) is a block swap/sign change; no dense J or
            # zero dissipation/input matrices are needed for this structure.
            half = dim // 2
            value = jnp.concatenate([grad_H[half:], -grad_H[:half]])
        else:
            value = self.J(x) @ grad_H
        if self.R is not None:
            value = value - self.R(x) @ grad_H
        if self.G is not None:
            value = value + self.G(x) @ u
        return value

    def output(self, t: Array | float, x: Array, u: Array) -> Array:
        """Evaluate the observation map, defaulting to the full state."""
        if self.observation is None:
            return x
        return self.observation(t, x, u)

    def as_nonlinear_system(self) -> NonlinearSystem:
        """Return the model as a generic nonlinear system."""
        return NonlinearSystem(
            dynamics=self.dynamics,
            observation=self.output,
            dt=self.dt,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )


class SeparablePHSSystem(PHSSystem):
    """PHSSystem subtype for separable Hamiltonians H(q,p) = V(q) + T(p).

    Stores `V` and `T` as separate callables so structure-preserving integrators
    such as Störmer-Verlet can use their gradients directly.

    The stored `H` is always the derived `V(q) + T(p)`.  All PHS behaviour
    (dynamics, diagnostics, simulation) falls through to `PHSSystem` unchanged.

    Create instances with `separable_phs_system()`, which validates the fields
    and derives the Hamiltonian from `V` and `T`.
    """

    V: Callable[[Array], Array] | None = None
    T: Callable[[Array], Array] | None = None
    n_dofs: int = eqx.field(static=True, default=0)

    def __check_init__(self) -> None:
        if self.V is None or self.T is None:
            raise ValueError(
                "SeparablePHSSystem requires V and T. "
                "Use separable_phs_system() to construct."
            )
        if self.n_dofs < 1:
            raise ValueError(
                f"n_dofs must be >= 1, got {self.n_dofs}. "
                "Use separable_phs_system() to construct."
            )


def separable_phs_system(
    V: Callable[[Array], Array],
    T: Callable[[Array], Array],
    *,
    n_dofs: int,
    R: Callable[[Array], Array] | None = None,
    G: Callable[[Array], Array] | None = None,
    J: Callable[[Array], Array] | None = None,
    output: Callable | None = None,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> SeparablePHSSystem:
    """Construct a separable port-Hamiltonian system.

    Builds a [SeparablePHSSystem][contrax.systems.SeparablePHSSystem] whose
    Hamiltonian is `H(q, p) = V(q) + T(p)`. The potential and kinetic energy
    remain separate model fields. `sample_system(method="sv")` requires this
    type with the default canonical `J`; other methods use the ordinary
    `PHSSystem` interface.

    Args:
        V: Potential energy `V(q)`. Maps `(n_dofs,) -> scalar`.
        T: Kinetic energy `T(p)`. Maps `(n_dofs,) -> scalar`.
        n_dofs: Number of generalized coordinates. `state_dim = 2 * n_dofs`.
        R: Optional dissipation map `R(x)` returning a `(2*n_dofs, 2*n_dofs)`
            positive-semidefinite matrix.
        G: Optional input map `G(x)` returning a `(2*n_dofs, input_dim)` matrix.
        J: Optional structure map. Defaults to `canonical_J(n_dofs)`.
        output: Optional measurement map with signature `(t, x, u) -> y`.
        input_dim: Optional static input dimension metadata.
        output_dim: Optional static output dimension metadata.

    Returns:
        [SeparablePHSSystem][contrax.systems.SeparablePHSSystem] with `H`
        derived from `V` and `T`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def V(q): return 0.5 * jnp.dot(q, q)
        >>> def T(p): return 0.5 * jnp.dot(p, p)
        >>> sys = cx.separable_phs_system(V, T, n_dofs=1, input_dim=1)
        >>> isinstance(sys, cx.PHSSystem)
        True
    """
    n_dofs = dimension(n_dofs, "n_dofs", minimum=1)
    if n_dofs is None:
        raise ValueError(f"n_dofs must be >= 1, got {n_dofs}.")

    def H(x: Array) -> Array:
        q, p = x[:n_dofs], x[n_dofs:]
        return V(q) + T(p)

    return SeparablePHSSystem(
        H=H,
        V=V,
        T=T,
        n_dofs=n_dofs,
        R=R,
        G=G,
        J=J,
        observation=output,
        dt=None,
        state_dim=2 * n_dofs,
        input_dim=dimension(input_dim, "input_dim"),
        output_dim=dimension(output_dim, "output_dim"),
    )


def phs_system(
    H: Callable[[Array], Array],
    *,
    R: Callable[[Array], Array] | None = None,
    G: Callable[[Array], Array] | None = None,
    J: Callable[[Array], Array] | None = None,
    output: Callable | None = None,
    observation: Callable | None = None,
    dt: ArrayLike | None = None,
    state_dim: int | None = None,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> PHSSystem:
    """Construct a port-Hamiltonian system object.

    Args:
        H: Storage/Hamiltonian function.
        R: Optional dissipation map.
        G: Optional input map.
        J: Optional structure map.
        output: Optional output/measurement map with signature `(t, x, u)`.
        observation: Deprecated synonym for `output`. Pass only one of
            `output` or `observation`.
        dt: Optional finite positive real scalar sample time.
        state_dim: Optional nonnegative static integer; positive even for default J.
        input_dim: Optional nonnegative static integer input dimension.
        output_dim: Optional nonnegative static integer output dimension.

    Returns:
        [PHSSystem][contrax.systems.PHSSystem]: A reusable port-Hamiltonian
            model.
    """
    if output is not None and observation is not None:
        raise ValueError("Pass only one of output= or observation=.")
    observation = output if output is not None else observation
    dt_arr, invalid = (None, False) if dt is None else sample_period(dt)
    state_dim = dimension(state_dim, "state_dim")
    if J is None and state_dim is not None and (state_dim == 0 or state_dim % 2):
        raise ValueError("Default canonical J requires a positive even state_dim")
    result = PHSSystem(
        H=H,
        R=R,
        G=G,
        J=J,
        observation=observation,
        dt=dt_arr,
        state_dim=state_dim,
        input_dim=dimension(input_dim, "input_dim"),
        output_dim=dimension(output_dim, "output_dim"),
    )
    return (
        result
        if dt is None
        else eqx.error_if(result, invalid, "dt must be finite and positive")
    )


def partition_state(x: ArrayLike, block_sizes: Sequence[int]) -> tuple[Array, ...]:
    """Split a vector into blocks with positive static integer sizes summing to n."""
    x = jnp.asarray(x)
    sizes = _block_sizes(block_sizes)
    if x.ndim != 1:
        raise ValueError("partition_state() expects a one-dimensional state vector.")
    total = sum(sizes)
    if x.shape[0] != total:
        raise ValueError(
            f"State length {x.shape[0]} does not match block_sizes sum {total}."
        )
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return tuple(x[offsets[i] : offsets[i + 1]] for i in range(len(sizes)))


def block_observation(
    block_sizes: Sequence[int],
    block_indices: Sequence[int],
) -> Callable[[Array | float, Array, Array], Array]:
    """Concatenate selected blocks, using positive static sizes and valid indices."""
    sizes = _block_sizes(block_sizes)
    indices = tuple(dimension(index, "block index") for index in block_indices)
    if not indices:
        raise ValueError("block_indices must not be empty.")
    if any(index is None or index >= len(sizes) for index in indices):
        raise ValueError("block_indices must refer to valid state blocks.")

    def output(t: Array | float, x: Array, u: Array) -> Array:
        del t, u
        parts = partition_state(x, sizes)
        return jnp.concatenate([parts[index] for index in indices], axis=0)

    return output


def _block_sizes(values):
    sizes = tuple(dimension(size, "block size", minimum=1) for size in values)
    if not sizes or any(size is None for size in sizes):
        raise ValueError("Block sizes must contain positive static integers")
    return sizes


def block_matrix(
    row_block_sizes: Sequence[int],
    col_block_sizes: Sequence[int],
    blocks: dict[tuple[int, int], ArrayLike] | None = None,
    *,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Assemble blocks with positive static sizes and (row, column) index keys.

    Entries outside the declared grid or with incorrect shapes raise ValueError.
    Omitted blocks are zero; provided blocks determine the promoted dtype.
    """
    row_sizes = _block_sizes(row_block_sizes)
    col_sizes = _block_sizes(col_block_sizes)

    block_entries = {} if blocks is None else dict(blocks)
    for key in block_entries:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Block keys must be (row, column) index pairs")
        i, j = (dimension(index, "block index") for index in key)
        if i is None or j is None or i >= len(row_sizes) or j >= len(col_sizes):
            raise ValueError("Block index lies outside the declared block grid")

    # First pass: determine the fully promoted dtype across all user-supplied blocks.
    result_dtype = jnp.float32 if dtype is None else dtype
    for block_val in block_entries.values():
        result_dtype = jnp.result_type(result_dtype, jnp.asarray(block_val).dtype)

    # Second pass: build rows with a consistent dtype so zero-fill blocks match.
    rows = []
    for i, row_size in enumerate(row_sizes):
        row = []
        for j, col_size in enumerate(col_sizes):
            if (i, j) in block_entries:
                block = jnp.asarray(block_entries[(i, j)], dtype=result_dtype)
                if block.shape != (row_size, col_size):
                    raise ValueError(
                        f"Block {(i, j)} has shape {block.shape}, expected "
                        f"{(row_size, col_size)}."
                    )
            else:
                block = jnp.zeros((row_size, col_size), dtype=result_dtype)
            row.append(block)
        rows.append(row)

    return jnp.block(rows)


def symmetrize_matrix(M: ArrayLike) -> Array:
    """Return the symmetric part `0.5 * (M + M.T)` of a square matrix."""
    M = jnp.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("symmetrize_matrix() expects a square matrix.")
    return 0.5 * (M + M.T)


def project_psd(
    M: ArrayLike,
    *,
    min_eigenvalue: float = 0.0,
) -> Array:
    """Project a real square matrix onto the symmetric PSD cone.

    min_eigenvalue must be a finite nonnegative scalar. Supports JAX transforms
    where the eigendecomposition is differentiable; repeated eigenvalues and
    the clipping boundary can make derivatives undefined.
    """
    M, floor = _positive_inputs(M, min_eigenvalue)
    M = symmetrize_matrix(M)
    eigvals, eigvecs = jnp.linalg.eigh(M)
    clipped = jnp.maximum(eigvals, floor)
    return symmetrize_matrix((eigvecs * clipped) @ eigvecs.T)


def phs_diagnostics(
    sys: PHSSystem,
    x: ArrayLike,
    u: ArrayLike | None = None,
) -> PHSStructureDiagnostics:
    """Evaluate local structure and power-balance diagnostics for a PHS state."""
    x = jnp.asarray(x)
    u_arr = jnp.zeros((0,), dtype=x.dtype) if u is None else jnp.asarray(u)
    dim = x.shape[0]
    J = sys.J(x) if sys.J is not None else canonical_J(dim // 2, dtype=x.dtype)
    R = sys.R(x) if sys.R is not None else jnp.zeros((dim, dim), dtype=x.dtype)
    G = sys.G(x) if sys.G is not None else _zero_input_matrix(x, u_arr)
    grad_H = jax.grad(sys.H)(x)
    R_sym = symmetrize_matrix(R)
    # Reuse already-computed J, R, G rather than calling sys.dynamics() again.
    Gu = G @ u_arr
    xdot = (J - R) @ grad_H + Gu
    storage_rate = jnp.vdot(grad_H, xdot)
    dissipation_power = jnp.vdot(grad_H, R_sym @ grad_H)
    supplied_power = jnp.vdot(grad_H, Gu)
    # Standard PHS energy balance: dH/dt = supply - dissipation
    # => supplied_power - dissipation_power - storage_rate = 0
    return PHSStructureDiagnostics(
        skew_symmetry_error=jnp.max(jnp.abs(J + J.T)),
        dissipation_symmetry_error=jnp.max(jnp.abs(R - R.T)),
        min_dissipation_eigenvalue=jnp.min(jnp.linalg.eigvalsh(R_sym)),
        storage_rate=storage_rate,
        dissipation_power=dissipation_power,
        supplied_power=supplied_power,
        power_balance_residual=supplied_power - dissipation_power - storage_rate,
    )


def schedule_phs(
    sys: PHSSystem,
    context_fn: Callable[[Array | float], Any],
    *,
    J: Callable[[Array | float, Array, Any], Array] | None = None,
    R: Callable[[Array | float, Array, Any], Array] | None = None,
    G: Callable[[Array | float, Array, Any], Array] | None = None,
    observation: Callable[[Array | float, Array, Array, Any], Array] | None = None,
) -> NonlinearSystem:
    """Bind an exogenous schedule/context to a canonical PHS model.

    The returned object is a [NonlinearSystem][contrax.systems.NonlinearSystem]
    accepted by simulation, estimation, and linearization functions.

    `context_fn(t)` should be a pure JAX-friendly callable returning any pytree
    of observed scheduling/context variables, such as a scale parameter
    `theta(t)`.

    Args:
        sys: Canonical PHS model with state-only `H(x)` and optional state-only
            `J(x)` / `R(x)` / `G(x)` fallbacks.
        context_fn: Callable mapping time to observed scheduling/context data.
        J: Optional scheduled structure map `J(t, x, context)`.
        R: Optional scheduled dissipation map `R(t, x, context)`.
        G: Optional scheduled input map `G(t, x, context)`.
        observation: Optional scheduled observation map
            `observation(t, x, u, context)`.

    Returns:
        [NonlinearSystem][contrax.systems.NonlinearSystem]: A scheduled
            nonlinear system with the context bound.
    """

    def dynamics(t: Array | float, x: Array, u: Array) -> Array:
        context = context_fn(t)
        dim = x.shape[0]
        J_mat = (
            J(t, x, context)
            if J is not None
            else (
                sys.J(x) if sys.J is not None else canonical_J(dim // 2, dtype=x.dtype)
            )
        )
        R_mat = (
            R(t, x, context)
            if R is not None
            else (
                sys.R(x) if sys.R is not None else jnp.zeros((dim, dim), dtype=x.dtype)
            )
        )
        G_mat = (
            G(t, x, context)
            if G is not None
            else (sys.G(x) if sys.G is not None else _zero_input_matrix(x, u))
        )
        grad_H = jax.grad(sys.H)(x)
        return (J_mat - R_mat) @ grad_H + G_mat @ u

    def scheduled_observation(t: Array | float, x: Array, u: Array) -> Array:
        if observation is not None:
            context = context_fn(t)
            return observation(t, x, u, context)
        if sys.observation is None:
            return x
        return sys.observation(t, x, u)

    return NonlinearSystem(
        dynamics=dynamics,
        observation=scheduled_observation,
        dt=sys.dt,
        state_dim=sys.state_dim,
        input_dim=sys.input_dim,
        output_dim=sys.output_dim,
    )


def phs_to_ss(
    sys: PHSSystem,
    x_eq: Array,
    u_eq: Array,
) -> ContLTI | DiscLTI:
    """Linearize a port-Hamiltonian model, preserving its time domain.

    Computes the Jacobian linearization of the PHS dynamics and observation map
    at `(x_eq, u_eq)`. The result models state/input/output deviations; constant
    operating-point offsets are omitted. Continuous models use
    delta_x_dot = A delta_x + B delta_u; discrete models use delta_x_next.

    For a linear PHS with quadratic Hamiltonian `H(x) = ½ x^T M x`, the
    linearization is exact and gives `A = (J - R) M`, `B = G`.

    Calls `linearize_ss` on the system's nonlinear representation at time zero
    and returns the local LTI model.

    Args:
        sys: Port-Hamiltonian system.
        x_eq: Equilibrium state. Shape: `(n,)`.
        u_eq: Equilibrium input. Shape: `(m,)`.

    Returns:
        ContLTI | DiscLTI: Local LTI approximation at `(x_eq, u_eq)`, retaining
            the model's discrete sample period when supplied.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def H(x): return 0.5 * jnp.dot(x, x)
        >>> phs = cx.phs_system(H, state_dim=2, input_dim=1)
        >>> lti = cx.phs_to_ss(phs, jnp.zeros(2), jnp.zeros(1))
        >>> isinstance(lti, cx.ContLTI)
        True
    """
    from contrax.core import linearize_ss

    nl = sys.as_nonlinear_system()
    return linearize_ss(nl, x_eq, u_eq, t=0.0)


__all__ = [
    "PHSSystem",
    "SeparablePHSSystem",
    "block_matrix",
    "block_observation",
    "canonical_J",
    "partition_state",
    "phs_diagnostics",
    "phs_system",
    "phs_to_ss",
    "project_psd",
    "schedule_phs",
    "separable_phs_system",
    "symmetrize_matrix",
]
