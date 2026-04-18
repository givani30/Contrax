"""contrax.phs — structure-preserving Hamiltonian/PHS helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from contrax.nonlinear import NonlinearSystem
from contrax.types import PHSStructureDiagnostics


def canonical_J(n_dofs: int, *, dtype: jnp.dtype | None = None) -> Array:
    """Construct the canonical skew-symmetric PHS structure matrix."""
    if n_dofs < 1:
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
        del t  # canonical first pass is time-invariant
        dim = x.shape[0]
        J = self.J(x) if self.J is not None else canonical_J(dim // 2, dtype=x.dtype)
        R = self.R(x) if self.R is not None else jnp.zeros((dim, dim), dtype=x.dtype)
        G = self.G(x) if self.G is not None else _zero_input_matrix(x, u)
        grad_H = jax.grad(self.H)(x)
        return (J - R) @ grad_H + G @ u

    def output(self, t: Array | float, x: Array, u: Array) -> Array:
        """Evaluate the observation map, defaulting to the full state."""
        if self.observation is None:
            return x
        return self.observation(t, x, u)

    def as_nonlinear_system(self) -> NonlinearSystem:
        """Expose the PHS model under the generic nonlinear-system contract."""
        return NonlinearSystem(
            dynamics=self.dynamics,
            observation=self.output,
            dt=self.dt,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
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
        dt: Optional discrete sample time.
        state_dim: Optional static state dimension metadata.
        input_dim: Optional static input dimension metadata.
        output_dim: Optional static output dimension metadata.

    Returns:
        [PHSSystem][contrax.systems.PHSSystem]: A reusable port-Hamiltonian
            model.
    """
    if output is not None and observation is not None:
        raise ValueError("Pass only one of output= or observation=.")
    observation = output if output is not None else observation
    dt_arr = None if dt is None else jnp.asarray(dt, dtype=float)
    return PHSSystem(
        H=H,
        R=R,
        G=G,
        J=J,
        observation=observation,
        dt=dt_arr,
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def partition_state(x: ArrayLike, block_sizes: Sequence[int]) -> tuple[Array, ...]:
    """Split a one-dimensional state vector into consecutive logical blocks."""
    x = jnp.asarray(x)
    sizes = tuple(int(size) for size in block_sizes)
    if x.ndim != 1:
        raise ValueError("partition_state() expects a one-dimensional state vector.")
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("block_sizes must contain only positive integers.")
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
    """Build an output map that concatenates selected state blocks."""
    sizes = tuple(int(size) for size in block_sizes)
    indices = tuple(int(index) for index in block_indices)
    if not sizes:
        raise ValueError("block_sizes must not be empty.")
    if not indices:
        raise ValueError("block_indices must not be empty.")
    if any(index < 0 or index >= len(sizes) for index in indices):
        raise ValueError("block_indices must refer to valid state blocks.")

    def output(t: Array | float, x: Array, u: Array) -> Array:
        del t, u
        parts = partition_state(x, sizes)
        return jnp.concatenate([parts[index] for index in indices], axis=0)

    return output


def block_matrix(
    row_block_sizes: Sequence[int],
    col_block_sizes: Sequence[int],
    blocks: dict[tuple[int, int], ArrayLike] | None = None,
    *,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Assemble a dense matrix from block entries."""
    row_sizes = tuple(int(size) for size in row_block_sizes)
    col_sizes = tuple(int(size) for size in col_block_sizes)
    if not row_sizes or not col_sizes:
        raise ValueError("row_block_sizes and col_block_sizes must not be empty.")
    if any(size <= 0 for size in row_sizes + col_sizes):
        raise ValueError("Block sizes must contain only positive integers.")

    block_entries = {} if blocks is None else dict(blocks)

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
    """Project a square matrix onto the symmetric PSD cone."""
    M = symmetrize_matrix(M)
    eigvals, eigvecs = jnp.linalg.eigh(M)
    clipped = jnp.maximum(eigvals, jnp.asarray(min_eigenvalue, dtype=eigvals.dtype))
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

    The returned object is an ordinary
    [NonlinearSystem][contrax.systems.NonlinearSystem], so it composes with the
    same simulation, estimation, and linearization surface as any other
    nonlinear system in Contrax.

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


__all__ = [
    "PHSSystem",
    "block_matrix",
    "block_observation",
    "canonical_J",
    "partition_state",
    "phs_diagnostics",
    "phs_system",
    "project_psd",
    "schedule_phs",
    "symmetrize_matrix",
]
