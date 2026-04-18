"""contrax.analysis — structural and transfer analysis helpers."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._precision import require_x64
from contrax.core import ContLTI, DiscLTI

__all__ = [
    "poles",
    "zeros",
    "ctrb",
    "obsv",
    "evalfr",
    "freqresp",
    "dcgain",
    "ctrb_gramian",
    "obsv_gramian",
    "lyap",
    "dlyap",
]


def _finite_horizon_gramian(A_left: Array, forcing: Array, horizon: float) -> Array:
    """Finite-horizon Gramian via Van Loan block-matrix exponential.

    Reference: Van Loan (1978), "Computing Integrals Involving the Matrix
    Exponential", IEEE TAC.
    """
    n = A_left.shape[0]
    block = jnp.block(
        [
            [A_left, forcing],
            [jnp.zeros_like(A_left), -A_left.T],
        ]
    )
    exp_block = jax.scipy.linalg.expm(block * horizon)
    phi11 = exp_block[:n, :n]
    phi12 = exp_block[:n, n:]
    return phi12 @ phi11.T


def poles(sys: DiscLTI | ContLTI) -> Array:
    """Return the poles of a state-space system.

    For the current state-space-only Contrax surface, poles are the eigenvalues
    of the state matrix `A`.

    This is a lightweight analysis helper that composes with `jit` and `vmap`
    for fixed-size systems. The return dtype is generally complex, even for
    real-valued systems with real poles.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: System poles. Shape: `(n,)`.
    """
    return jnp.linalg.eigvals(sys.A)


def ctrb(sys: DiscLTI | ContLTI) -> Array:
    """Construct the controllability matrix of a state-space system.

    Forms `[B, AB, A^2 B, ..., A^{n-1} B]` for a continuous or discrete linear
    system.

    This is a structural analysis primitive rather than a yes-or-no
    controllability test by itself. Users typically inspect the downstream rank
    or conditioning of the returned matrix. The implementation uses
    `jax.lax.scan` so it composes with transforms, but it is still intended
    primarily for design-time analysis rather than tight inner loops.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: Controllability matrix. Shape: `(n, n * m)`.
    """
    A, B = sys.A, sys.B
    n = A.shape[0]

    def step(block, _):
        return A @ block, block

    _, blocks = jax.lax.scan(step, B, None, length=n)
    return jnp.transpose(blocks, (1, 0, 2)).reshape(A.shape[0], -1)


def obsv(sys: DiscLTI | ContLTI) -> Array:
    """Construct the observability matrix of a state-space system.

    Forms `[C; CA; CA^2; ...; CA^{n-1}]` for a continuous or discrete linear
    system.

    This is a structural analysis primitive rather than a yes-or-no
    observability test by itself. Users typically inspect the downstream rank
    or conditioning of the returned matrix. The implementation uses
    `jax.lax.scan` so it composes with transforms, but it is still intended
    primarily for design-time analysis rather than tight inner loops.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: Observability matrix. Shape: `(n * p, n)`.
    """
    A, C = sys.A, sys.C
    n = A.shape[0]

    def step(block, _):
        return block @ A, block

    _, blocks = jax.lax.scan(step, C, None, length=n)
    return blocks.reshape(-1, A.shape[0])


def evalfr(sys: DiscLTI | ContLTI, point: complex | Array) -> Array:
    """Evaluate the transfer matrix at a complex frequency point.

    For continuous systems this computes `G(s) = C (sI - A)^-1 B + D`. For
    discrete systems it computes `G(z) = C (zI - A)^-1 B + D`.

    This is the first transfer-evaluation helper in Contrax rather than a full
    transfer-function algebra layer. It is intended for design-time analysis,
    spot checks, and building higher-level frequency-response tools.

    Args:
        sys: Continuous or discrete LTI system.
        point: Complex evaluation point `s` or `z`. Shape: scalar.

    Returns:
        Array: Transfer matrix evaluated at `point`. Shape: `(p, m)`.
    """
    point = jnp.asarray(point, dtype=jnp.result_type(sys.A.dtype, 1j))
    n = sys.A.shape[0]
    identity = jnp.eye(n, dtype=point.dtype)
    A = sys.A.astype(point.dtype)
    B = sys.B.astype(point.dtype)
    C = sys.C.astype(point.dtype)
    D = sys.D.astype(point.dtype)
    resolvent_rhs = jnp.linalg.solve(point * identity - A, B)
    return C @ resolvent_rhs + D


def freqresp(sys: DiscLTI | ContLTI, omega: Array) -> Array:
    """Evaluate the frequency response over angular frequencies.

    Continuous systems are evaluated on the imaginary axis at `s = j omega`.
    Discrete systems are evaluated on the unit circle at
    `z = exp(j omega dt)`.

    This is a basic state-space frequency-response helper. It returns the
    complex transfer matrix at each frequency but does not yet format Bode or
    Nyquist plots for you.

    Args:
        sys: Continuous or discrete LTI system.
        omega: Angular frequencies in rad/s. Shape: `(k,)`.

    Returns:
        Array: Complex frequency response. Shape: `(k, p, m)`.
    """
    omega = jnp.asarray(omega, dtype=float)
    if isinstance(sys, ContLTI):
        points = 1j * omega
    else:
        points = jnp.exp(1j * omega * sys.dt)
    return jax.vmap(lambda point: evalfr(sys, point))(points)


def dcgain(sys: DiscLTI | ContLTI) -> Array:
    """Return the zero-frequency or steady-state gain of an LTI system.

    For continuous systems this evaluates `G(0)`. For discrete systems this
    evaluates `G(1)`.

    This helper assumes the corresponding resolvent is nonsingular. It is a
    useful quick check for low-frequency behavior, but it is not by itself a
    stability certificate.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: DC gain matrix. Shape: `(p, m)`.
    """
    point = jnp.array(0.0) if isinstance(sys, ContLTI) else jnp.array(1.0)
    return evalfr(sys, point)


def ctrb_gramian(sys: ContLTI, t: float = 10.0) -> Array:
    """Compute the finite-horizon controllability Gramian.

    Uses the Van Loan block-matrix exponential construction to evaluate the
    finite-horizon controllability Gramian in closed form for continuous-time
    linear systems.

    Args:
        sys: Continuous-time linear system.
        t: Integration horizon. Shape: scalar.

    Returns:
        Array: Finite-horizon controllability Gramian. Shape: `(n, n)`.
    """
    require_x64("ctrb_gramian")
    A, B = sys.A, sys.B
    return _finite_horizon_gramian(A, B @ B.T, t)


def obsv_gramian(sys: ContLTI, t: float = 10.0) -> Array:
    """Compute the finite-horizon observability Gramian.

    Uses the same Van Loan block-matrix exponential construction as
    `ctrb_gramian()`, applied to the dual system.

    Args:
        sys: Continuous-time linear system.
        t: Integration horizon. Shape: scalar.

    Returns:
        Array: Finite-horizon observability Gramian. Shape: `(n, n)`.
    """
    require_x64("obsv_gramian")
    return _finite_horizon_gramian(sys.A.T, sys.C.T @ sys.C, t)


def _lyapunov_operator_matrix(A: Array, *, continuous: bool) -> Array:
    """Build the Kronecker-form linear system for a Lyapunov equation.

    Continuous: maps X → AX + XA^T.
    Discrete:   maps X → X - AXA^T.
    """
    n = A.shape[0]
    basis = jnp.eye(n * n, dtype=A.dtype).reshape(n * n, n, n)
    if continuous:
        op = lambda X: A @ X + X @ A.T  # noqa: E731
    else:
        op = lambda X: X - A @ X @ A.T  # noqa: E731
    return jax.vmap(lambda X: op(X).reshape(-1))(basis).T


def lyap(A: Array, Q: Array) -> Array:
    """Solve the continuous Lyapunov equation AX + XA^T + Q = 0.

    Returns the unique symmetric solution X when all eigenvalues of A satisfy
    Re(λ_i + λ_j) ≠ 0 (i.e. A is stable or the sum condition holds).

    Uses an explicit Kronecker-form linear solve. This is exact for small
    systems but scales as O(n^6) in the dense case; it is intended for
    design-time use, not inner-loop computation.

    Args:
        A: State matrix. Shape: `(n, n)`.
        Q: Right-hand-side matrix. Shape: `(n, n)`.

    Returns:
        Array: Solution X such that AX + XA^T = -Q. Shape: `(n, n)`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        >>> Q = jnp.eye(2)
        >>> X = cx.lyap(A, Q)
        >>> jnp.allclose(A @ X + X @ A.T + Q, 0.0, atol=1e-6)
        Array(True, dtype=bool)
    """
    require_x64("lyap")
    A = jnp.asarray(A)
    Q = jnp.asarray(Q)
    mat = _lyapunov_operator_matrix(A, continuous=True)
    X = jnp.linalg.solve(mat, (-Q).reshape(-1)).reshape(A.shape)
    return (X + X.T) / 2


def dlyap(A: Array, Q: Array) -> Array:
    """Solve the discrete Lyapunov equation AXA^T - X + Q = 0.

    Returns the unique symmetric solution X when all eigenvalues of A satisfy
    |λ_i λ_j| ≠ 1 (i.e. A is Schur-stable or the product condition holds).

    Uses an explicit Kronecker-form linear solve. This is exact for small
    systems but scales as O(n^6) in the dense case; it is intended for
    design-time use, not inner-loop computation.

    Args:
        A: State matrix. Shape: `(n, n)`.
        Q: Right-hand-side matrix. Shape: `(n, n)`.

    Returns:
        Array: Solution X such that AXA^T - X = -Q. Shape: `(n, n)`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> A = jnp.array([[0.5, 0.0], [0.0, 0.3]])
        >>> Q = jnp.eye(2)
        >>> X = cx.dlyap(A, Q)
        >>> jnp.allclose(A @ X @ A.T - X + Q, 0.0, atol=1e-6)
        Array(True, dtype=bool)
    """
    require_x64("dlyap")
    A = jnp.asarray(A)
    Q = jnp.asarray(Q)
    mat = _lyapunov_operator_matrix(A, continuous=False)
    X = jnp.linalg.solve(mat, Q.reshape(-1)).reshape(A.shape)
    return (X + X.T) / 2


def _zeros_d0(A: Array, B: Array, C: Array) -> Array:
    """Transmission zeros for D=0 via controlled-invariant subspace iteration.

    Computes the largest (A, B)-controlled invariant subspace V* contained in
    ker(C), then returns the eigenvalues of A restricted to V*. Works for
    SISO and MIMO square (p == m) systems with zero feed-through.

    Uses Python-level SVD iteration; not JIT-compilable.

    Reference: Basile & Marro (1992), "Controlled and Conditioned Invariants
    in Linear System Theory", Prentice-Hall.
    """
    n = A.shape[0]
    tol = float(jnp.finfo(A.dtype).eps) * 1e4

    # Initial subspace: null(C)
    _, sv_C, Vt_C = jnp.linalg.svd(C, full_matrices=True)
    rank_C = int(jnp.sum(sv_C > tol * (float(sv_C[0]) if sv_C.shape[0] > 0 else 1.0)))
    Z = Vt_C[rank_C:, :].T  # orthonormal basis for ker(C), shape (n, n - rank_C)

    if Z.shape[1] == 0:
        return jnp.array([], dtype=jnp.complex128)

    for _ in range(n):
        dim_old = Z.shape[1]

        # Orthonormal basis W for V + im(B)
        U_M, sv_M, _ = jnp.linalg.svd(jnp.hstack([Z, B]), full_matrices=False)
        rank_M = int(jnp.sum(sv_M > tol * float(sv_M[0])))
        W = U_M[:, :rank_M]  # (n, rank_M)

        # Pre-image of W under A: null space of (I - W W^T) A
        PA = A - W @ (W.T @ A)  # (n, n)
        _, sv_PA, Vt_PA = jnp.linalg.svd(PA, full_matrices=True)
        rank_PA = int(jnp.sum(sv_PA > tol * float(sv_PA[0])))
        pre_dim = n - rank_PA
        if pre_dim == 0:
            return jnp.array([], dtype=jnp.complex128)
        pre_basis = Vt_PA[rank_PA:, :].T  # (n, pre_dim)

        # Intersect pre-image with ker(C): null space of C @ pre_basis
        C_pre = C @ pre_basis  # (p, pre_dim)
        _, sv_CP, Vt_CP = jnp.linalg.svd(C_pre, full_matrices=True)
        rank_CP = (
            int(jnp.sum(sv_CP > tol * float(sv_CP[0]))) if sv_CP.shape[0] > 0 else 0
        )
        null_dim = pre_dim - rank_CP
        if null_dim == 0:
            return jnp.array([], dtype=jnp.complex128)
        coords = Vt_CP[rank_CP:, :].T  # (pre_dim, null_dim)
        Z = pre_basis @ coords  # (n, null_dim) — new basis for V_{i+1}

        # Re-orthonormalize
        Z, _ = jnp.linalg.qr(Z)
        Z = Z[:, :null_dim]

        if null_dim == dim_old:
            break

    # Zeros = eigenvalues of A restricted to V*
    Az = Z.T @ A @ Z
    return jnp.linalg.eigvals(Az).astype(jnp.complex128)


def zeros(sys: DiscLTI | ContLTI) -> Array:
    """Return the transmission zeros of a state-space system.

    Transmission zeros are the values s (continuous) or z (discrete) for
    which the Rosenbrock system matrix `M(s) = [[sI - A, -B], [C, D]]` loses rank.

    **D = 0** (most common case): uses a controlled-invariant subspace
    iteration to find V*, the largest (A, B)-controlled invariant subspace
    in ker(C). Returns the eigenvalues of A restricted to V*. Works for both
    SISO and square MIMO (p == m) systems.

    **D invertible** (square, full rank): uses the direct formula
    `zeros = eig(A - B @ solve(D, C))`.

    Non-square D and rank-deficient non-zero D are not yet supported.

    This is a design-time structural analysis primitive; it is not intended
    for use inside `jit`-compiled loops.

    Reference: Basile & Marro (1992), "Controlled and Conditioned Invariants
    in Linear System Theory", Prentice-Hall.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: Transmission zeros. Complex dtype. Number of zeros equals
        the dimension of V* (D = 0) or `n` (invertible D).

    Raises:
        NotImplementedError: For non-square D or rank-deficient non-zero D.
    """
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    m, p = B.shape[1], C.shape[0]

    if p != m:
        raise NotImplementedError(
            f"zeros() requires square D (p == m); got p={p}, m={m}. "
            "Non-square systems need a generalized eigenvalue solver."
        )

    d_is_zero = bool(jnp.max(jnp.abs(D)) < 1e-14)

    if d_is_zero:
        return _zeros_d0(A, B, C)

    return jnp.linalg.eigvals(A - B @ jnp.linalg.solve(D, C)).astype(jnp.complex128)
