"""Exact finite-horizon quadratic costs for discrete LTI feedback."""

import jax.numpy as jnp
from jax import Array

from contrax.core import DiscLTI


def _compose_gramian(A, W, num_steps):
    """Return A**N and sum (A**k).T W A**k by binary composition."""
    phi, gramian = jnp.eye(A.shape[0], dtype=A.dtype), jnp.zeros_like(W)
    # Reference: finite Gramian semigroup identity
    # H_(a+b) = H_a + (A**a).T H_b A**a.
    # The loop is over static horizon bits, never over array runtime values.
    for bit in bin(num_steps)[2:] if num_steps else ():
        gramian = gramian + phi.T @ gramian @ phi
        phi = phi @ phi
        if bit == "1":
            gramian = gramian + phi.T @ W @ phi
            phi = A @ phi
    return phi, (gramian + gramian.T) / 2


def quadratic_rollout_cost(
    sys: DiscLTI,
    K: Array,
    Q: Array,
    R: Array,
    x0: Array,
    *,
    num_steps: int,
    Q_terminal: Array | None = None,
) -> Array:
    """Evaluate a finite-horizon LTI feedback cost without storing a trajectory.

    Returns sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) plus the optional
    terminal cost x_N^T Q_terminal x_N, with u_k=-K x_k. Uses binary Gramian
    composition in O(log N) matrix products and O(n^2) storage. No stability
    assumption is needed for finite N; long unstable horizons may overflow.

    Supports `jit`, `vmap`, JVP and VJP and preserves input precision. The
    horizon is a static nonnegative Python integer. Use `vmap` for batches of
    initial states or gains; x0 itself describes one initial state.

    Args:
        sys: Discrete LTI system with n states and m inputs.
        K: Feedback gain. Shape: `(m, n)`.
        Q: State cost. Shape: `(n, n)`.
        R: Input cost. Shape: `(m, m)`.
        x0: Initial state. Shape: `(n,)`.
        num_steps: Number of cost terms/transitions. Zero is valid.
        Q_terminal: Optional terminal-state cost. Shape: `(n, n)`.

    Returns:
        Array: Scalar quadratic cost for the single initial state.

    Raises:
        TypeError: For a continuous model or noninteger horizon.
        ValueError: For a negative horizon or inconsistent shapes.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> sys = cx.dss(jnp.eye(1), jnp.eye(1), jnp.eye(1), jnp.zeros((1, 1)), 1)
        >>> cx.quadratic_rollout_cost(sys, jnp.zeros((1, 1)), jnp.eye(1),
        ...                           jnp.eye(1), jnp.ones(1), num_steps=3)
        Array(3., dtype=float64)
    """
    if not isinstance(sys, DiscLTI):
        raise TypeError("quadratic_rollout_cost requires a discrete LTI system")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool):
        raise TypeError("num_steps must be a static Python integer")
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    K, Q, R, x0 = (jnp.asarray(x) for x in (K, Q, R, x0))
    n, m = sys.B.shape
    if K.shape != (m, n) or Q.shape != (n, n) or R.shape != (m, m) or x0.shape != (n,):
        raise ValueError("K, Q, R and x0 must have shapes (m,n), (n,n), (m,m), (n,)")
    terminal = jnp.zeros_like(Q) if Q_terminal is None else jnp.asarray(Q_terminal)
    if terminal.shape != (n, n):
        raise ValueError("Q_terminal must have shape (n, n)")
    A = sys.A - sys.B @ K
    W = Q + K.T @ R @ K
    phi, gramian = _compose_gramian(A, W, num_steps)
    gramian = gramian + phi.T @ terminal @ phi
    return x0 @ ((gramian + gramian.T) / 2) @ x0
