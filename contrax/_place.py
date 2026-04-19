"""contrax._place — Pole placement internals and public place() function."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from contrax.core import ContLTI, DiscLTI
from contrax.types import PlaceResult


def _ctrb_matrix(A: Array, B: Array) -> Array:
    """Controllability matrix [B, AB, A²B, ...]."""
    n = A.shape[0]
    cols = [B]
    for _ in range(n - 1):
        cols.append(A @ cols[-1])
    return jnp.concatenate(cols, axis=1)


def _order_complex_poles(poles: ArrayLike) -> Array:
    poles_np = np.asarray(poles, dtype=complex)
    reals = sorted(
        [p for p in poles_np if np.isreal(p)],
        key=lambda z: (float(np.real(z)), float(np.imag(z))),
    )
    positive_imag = sorted(
        [p for p in poles_np if np.imag(p) > 0],
        key=lambda z: (float(np.real(z)), float(np.imag(z))),
    )
    ordered = list(reals)
    for pole in positive_imag:
        ordered.extend([pole, np.conj(pole)])
    if len(ordered) != len(poles_np):
        raise ValueError("Poles must be real or come in conjugate pairs.")
    return jnp.asarray(np.array(ordered))


def _place_knv0_update(ker_pole: list[Array], transfer_matrix: Array, j: int) -> Array:
    transfer_matrix_not_j = jnp.delete(transfer_matrix, j, axis=1)
    Q, _ = jnp.linalg.qr(transfer_matrix_not_j, mode="complete")
    yj = ker_pole[j] @ (ker_pole[j].T @ Q[:, -1:])
    if float(jnp.linalg.norm(yj)) > 1e-12:
        transfer_matrix = transfer_matrix.at[:, j].set((yj / jnp.linalg.norm(yj))[:, 0])
    return transfer_matrix


def _place_yt_real_update(
    ker_pole: list[Array], Q: Array, transfer_matrix: Array, i: int, j: int
) -> Array:
    u = Q[:, -2:]
    u0 = u[:, :1]
    u1 = u[:, 1:]
    m = ker_pole[i].T @ (u0 @ u1.T - u1 @ u0.T) @ ker_pole[j]
    um, sm, vmh = jnp.linalg.svd(m, full_matrices=True)
    mu1 = um[:, :1]
    mu2 = um[:, 1:2]
    nu1 = vmh.conj().T[:, :1]
    nu2 = vmh.conj().T[:, 1:2]

    tm_ij = jnp.vstack([transfer_matrix[:, i : i + 1], transfer_matrix[:, j : j + 1]])
    if abs(float(sm[0] - sm[1])) > 1e-12:
        ker_mu_nu = jnp.vstack([ker_pole[i] @ mu1, ker_pole[j] @ nu1])
    else:
        top = jnp.hstack([ker_pole[i], jnp.zeros_like(ker_pole[i])])
        bot = jnp.hstack([jnp.zeros_like(ker_pole[j]), ker_pole[j]])
        ker_ij = jnp.vstack([top, bot])
        mu_nu = jnp.vstack([jnp.hstack([mu1, mu2]), jnp.hstack([nu1, nu2])])
        ker_mu_nu = ker_ij @ mu_nu

    transfer_matrix_ij = ker_mu_nu @ (ker_mu_nu.T @ tm_ij)
    n = transfer_matrix.shape[0]
    if float(jnp.linalg.norm(transfer_matrix_ij)) > 1e-12:
        transfer_matrix_ij = (
            jnp.sqrt(jnp.asarray(2.0, dtype=transfer_matrix.dtype))
            * transfer_matrix_ij
            / jnp.linalg.norm(transfer_matrix_ij)
        )
        transfer_matrix = transfer_matrix.at[:, i].set(transfer_matrix_ij[:n, 0])
        transfer_matrix = transfer_matrix.at[:, j].set(transfer_matrix_ij[n:, 0])
    else:
        transfer_matrix = transfer_matrix.at[:, i].set(ker_mu_nu[:n, 0])
        transfer_matrix = transfer_matrix.at[:, j].set(ker_mu_nu[n:, 0])
    return transfer_matrix


def _place_yt_complex_update(
    ker_pole: list[Array], Q: Array, transfer_matrix: Array, i: int, j: int
) -> Array:
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=transfer_matrix.dtype))
    ur = sqrt2 * Q[:, -2:-1]
    ui = sqrt2 * Q[:, -1:]
    u = ur + 1j * ui
    ker = ker_pole[i]
    m = ker.conj().T @ (u @ u.conj().T - u.conj() @ u.T) @ ker
    evals, evecs = jnp.linalg.eig(m)
    order = jnp.argsort(jnp.abs(evals))
    mu1 = evecs[:, order[-1] : order[-1] + 1]
    mu2 = evecs[:, order[-2] : order[-2] + 1]
    tm_complex = transfer_matrix[:, i : i + 1] + 1j * transfer_matrix[:, j : j + 1]
    if abs(float(jnp.abs(evals[order[-1]]) - jnp.abs(evals[order[-2]]))) > 1e-12:
        ker_mu = ker @ mu1
    else:
        ker_mu = ker @ jnp.hstack([mu1, mu2])
    transfer_ij = ker_mu @ (ker_mu.conj().T @ tm_complex)
    if float(jnp.linalg.norm(transfer_ij)) > 1e-12:
        transfer_ij = transfer_ij / jnp.linalg.norm(transfer_ij)
        transfer_matrix = transfer_matrix.at[:, i].set(jnp.real(transfer_ij[:, 0]))
        transfer_matrix = transfer_matrix.at[:, j].set(jnp.imag(transfer_ij[:, 0]))
    else:
        transfer_matrix = transfer_matrix.at[:, i].set(jnp.real(ker_mu[:, 0]))
        transfer_matrix = transfer_matrix.at[:, j].set(jnp.imag(ker_mu[:, 0]))
    return transfer_matrix


def _place_yt_update_order(poles: Array) -> np.ndarray:
    poles_np = np.asarray(poles)
    nb_real = poles_np[np.isreal(poles_np)].shape[0]
    hnb = nb_real // 2
    if nb_real > 0:
        update_order = [[nb_real], [1]]
    else:
        update_order = [[], []]
    r_comp = np.arange(nb_real + 1, len(poles_np) + 1, 2)
    r_p = np.arange(1, hnb + nb_real % 2)
    update_order[0].extend(2 * r_p)
    update_order[1].extend(2 * r_p + 1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    r_p = np.arange(1, hnb + 1)
    update_order[0].extend(2 * r_p - 1)
    update_order[1].extend(2 * r_p)
    if hnb == 0 and np.isreal(poles_np[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    r_j = np.arange(2, hnb + nb_real % 2)
    for j in r_j:
        for i in range(1, hnb + 1):
            update_order[0].append(i)
            update_order[1].append(i + j)
    if hnb == 0 and np.isreal(poles_np[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    r_j = np.arange(2, hnb + nb_real % 2)
    for j in r_j:
        for i in range(hnb + 1, nb_real + 1):
            idx_1 = i + j
            if idx_1 > nb_real:
                idx_1 = i + j - nb_real
            update_order[0].append(i)
            update_order[1].append(idx_1)
    if hnb == 0 and np.isreal(poles_np[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    for i in range(1, hnb + 1):
        update_order[0].append(i)
        update_order[1].append(i + hnb)
    if hnb == 0 and np.isreal(poles_np[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp + 1)
    return np.array(update_order).T - 1


def _place_ackermann(A: Array, B: Array, poles: ArrayLike) -> Array:
    n = A.shape[0]
    poles_np = np.asarray(poles)
    coeffs = np.poly(poles_np)
    eye = jnp.eye(n, dtype=A.dtype)
    p_A = float(coeffs[0]) * eye
    for c in coeffs[1:]:
        p_A = p_A @ A + float(c) * eye
    C_ctrb = _ctrb_matrix(A, B)
    e_n = jnp.zeros(n, dtype=A.dtype).at[-1].set(1.0)
    x = jnp.linalg.solve(C_ctrb.T, e_n)
    L = x @ p_A
    return L.reshape(1, n)


def _place_robust(
    A: Array,
    B: Array,
    poles: ArrayLike,
    *,
    method: str,
    rtol: float,
    maxiter: int,
) -> Array:
    poles = _order_complex_poles(poles)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if len(poles) != A.shape[0]:
        raise ValueError("place() requires exactly n poles.")

    rankB = int(jnp.linalg.matrix_rank(B))
    poles_np = np.asarray(poles)
    for pole in poles_np:
        if np.sum(np.isclose(poles_np, pole)) > rankB:
            raise ValueError(
                "at least one requested pole is repeated more than rank(B) times"
            )
    if method == "KNV0" and not np.all(np.isreal(poles_np)):
        raise ValueError("Complex poles are not supported by KNV0.")

    U, Z = jnp.linalg.qr(B, mode="complete")
    u0 = U[:, :rankB]
    u1 = U[:, rankB:]
    Z = Z[:rankB, :]

    if B.shape[0] == rankB:
        diag_poles = jnp.zeros(A.shape, dtype=A.dtype)
        idx = 0
        while idx < poles.shape[0]:
            pole = poles[idx]
            diag_poles = diag_poles.at[idx, idx].set(jnp.real(pole))
            if not bool(jnp.isreal(pole)):
                diag_poles = diag_poles.at[idx, idx + 1].set(-jnp.imag(pole))
                diag_poles = diag_poles.at[idx + 1, idx + 1].set(jnp.real(pole))
                diag_poles = diag_poles.at[idx + 1, idx].set(jnp.imag(pole))
                idx += 1
            idx += 1
        gain_matrix, *_ = jnp.linalg.lstsq(B, diag_poles - A, rcond=None)
        return -jnp.real(gain_matrix)

    ker_pole: list[Array] = []
    transfer_cols: list[Array] = []
    skip_conjugate = False
    n = B.shape[0]
    identity = jnp.eye(n, dtype=A.dtype)

    for j in range(n):
        if skip_conjugate:
            skip_conjugate = False
            continue
        pole_space_j = (u1.T @ (A - poles[j] * identity)).T
        Q, _ = jnp.linalg.qr(pole_space_j, mode="complete")
        ker_j = Q[:, pole_space_j.shape[1] :]
        transfer_j = jnp.sum(ker_j, axis=1, keepdims=True)
        transfer_j = transfer_j / jnp.linalg.norm(transfer_j)
        if not bool(jnp.isreal(poles[j])):
            transfer_cols.extend(
                [jnp.real(transfer_j[:, 0]), jnp.imag(transfer_j[:, 0])]
            )
            ker_pole.extend([ker_j, ker_j])
            skip_conjugate = True
        else:
            transfer_cols.append(transfer_j[:, 0])
            ker_pole.append(ker_j)

    transfer_matrix = jnp.stack(transfer_cols, axis=1)

    if rankB > 1:
        if method == "KNV0":
            for _ in range(maxiter):
                det_before = max(
                    float(jnp.abs(jnp.linalg.det(transfer_matrix))),
                    float(np.sqrt(np.spacing(1))),
                )
                for j in range(len(poles)):
                    transfer_matrix = _place_knv0_update(ker_pole, transfer_matrix, j)
                det_after = max(
                    float(jnp.abs(jnp.linalg.det(transfer_matrix))),
                    float(np.sqrt(np.spacing(1))),
                )
                cur_rtol = abs((det_after - det_before) / det_after)
                if cur_rtol < rtol:
                    break
        elif method == "YT":
            update_order = _place_yt_update_order(poles)
            for _ in range(maxiter):
                det_before = max(
                    float(jnp.abs(jnp.linalg.det(transfer_matrix))),
                    float(np.sqrt(np.spacing(1))),
                )
                for i, j in update_order:
                    if i == j:
                        transfer_matrix = _place_knv0_update(
                            ker_pole, transfer_matrix, i
                        )
                    else:
                        tm_not = jnp.delete(transfer_matrix, jnp.array([i, j]), axis=1)
                        Q, _ = jnp.linalg.qr(tm_not, mode="complete")
                        if np.isreal(np.asarray(poles[i])):
                            transfer_matrix = _place_yt_real_update(
                                ker_pole, Q, transfer_matrix, i, j
                            )
                        else:
                            transfer_matrix = _place_yt_complex_update(
                                ker_pole, Q, transfer_matrix, i, j
                            )
                det_after = max(
                    float(jnp.abs(jnp.linalg.det(transfer_matrix))),
                    float(np.sqrt(np.spacing(1))),
                )
                cur_rtol = abs((det_after - det_before) / det_after)
                if cur_rtol < rtol:
                    break
        else:
            raise ValueError(f"Unknown pole-placement method: {method!r}")

    transfer_complex = transfer_matrix.astype(jnp.complex128)
    idx = 0
    while idx < poles.shape[0] - 1:
        if not bool(jnp.isreal(poles[idx])):
            rel = transfer_complex[:, idx].copy()
            img = transfer_complex[:, idx + 1]
            transfer_complex = transfer_complex.at[:, idx].set(rel - 1j * img)
            transfer_complex = transfer_complex.at[:, idx + 1].set(rel + 1j * img)
            idx += 1
        idx += 1

    m = jnp.linalg.solve(transfer_complex.T, (jnp.diag(poles) @ transfer_complex.T)).T
    gain_matrix = jnp.linalg.solve(Z, u0.T @ (m - A))
    return -jnp.real(gain_matrix)


# ── pole placement ─────────────────────────────────────────────────────────
# Uses Ackermann's formula for SISO systems (m=1).
# Reference: Ackermann (1972), "Der Entwurf linearer Regelungssysteme im
# Zustandsraum", Regelungstechnik 20(7), pp. 297-300.
# Note: numerically fragile for n > 5. A QR-based Sylvester equation mapping is
# the next likely implementation path for arbitrary n and MIMO.
# Reference: Kautsky, Nichols & Van Dooren (1985), "Robust Pole
# Assignment in Linear State Feedback", Int. J. Control 41(5), pp. 1129-1155.


def place(
    sys: DiscLTI | ContLTI,
    poles: ArrayLike,
    *,
    method: str | None = None,
    rtol: float = 1e-3,
    maxiter: int = 30,
) -> PlaceResult:
    """Place closed-loop poles for a state-space system.

    Uses a JAX-native robust assignment path based on KNV0 for real-pole cases
    and YT for complex-pole cases. Both methods are design-time helpers rather
    than traced differentiable primitives.

    If `method` is left as `None`, Contrax selects `KNV0` for all-real pole
    sets and `YT` when any complex-conjugate pair is present. For single-input
    systems, Ackermann remains available as a fallback if the robust path
    fails.

    Args:
        sys: Continuous or discrete-time system.
        poles: Desired closed-loop pole locations. Shape: `(n,)`.
        method: Optional robust assignment method. Supported values are
            `"KNV0"` and `"YT"`. When omitted, Contrax chooses based on
            whether the requested poles are all real.
        rtol: Determinant-improvement stopping tolerance for the robust update.
        maxiter: Maximum robust-assignment iterations.

    Returns:
        PlaceResult: Result with `.K` (state-feedback gain, shape `(m, n)`)
        and `.poles` (achieved closed-loop eigenvalues). Supports tuple
        unpacking: ``K, poles = cx.place(...)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[1.0, 0.1], [0.0, 0.9]]),
        ...     jnp.array([[0.0], [1.0]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ... )
        >>> result = cx.place(sys, jnp.array([0.6, 0.7]))
        >>> K = result.K  # or: K, achieved_poles = cx.place(...)
    """
    A, B = sys.A, sys.B
    poles_ordered = _order_complex_poles(poles)
    has_complex = bool(np.any(np.imag(np.asarray(poles_ordered)) != 0.0))
    chosen_method = method or ("YT" if has_complex else "KNV0")

    try:
        K = _place_robust(
            A,
            B,
            poles_ordered,
            method=chosen_method,
            rtol=rtol,
            maxiter=maxiter,
        )
    except Exception:
        if B.shape[1] == 1:
            K = _place_ackermann(A, B, poles_ordered)
        else:
            raise
    achieved_poles = jnp.linalg.eigvals(A - B @ K)
    return PlaceResult(K=K, poles=achieved_poles)
