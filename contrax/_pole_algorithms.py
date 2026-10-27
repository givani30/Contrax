"""Native linear algebra for design-time KNV0/YT and Ackermann assignment."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def _ctrb_matrix(A: Array, B: Array) -> Array:
    """Controllability matrix [B, AB, A²B, ...]."""
    n = A.shape[0]
    cols = [B]
    for _ in range(n - 1):
        cols.append(A @ cols[-1])
    return jnp.concatenate(cols, axis=1)


def _order_complex_poles(poles: ArrayLike) -> Array:
    poles_np = np.asarray(poles, dtype=complex)
    if poles_np.ndim != 1 or not np.all(np.isfinite(poles_np)):
        raise ValueError("Poles must be a finite one-dimensional sequence")
    reals = sorted(
        [p for p in poles_np if np.isreal(p)],
        key=lambda z: (float(np.real(z)), float(np.imag(z))),
    )
    positive_imag = sorted(
        [p for p in poles_np if np.imag(p) > 0],
        key=lambda z: (float(np.real(z)), float(np.imag(z))),
    )
    ordered = list(reals)
    negative = list(poles_np[np.imag(poles_np) < 0])
    for pole in positive_imag:
        matches = [
            i
            for i, other in enumerate(negative)
            if np.isclose(other, np.conj(pole), rtol=1e-12, atol=1e-14)
        ]
        if not matches:
            raise ValueError("Poles must be real or come in conjugate pairs.")
        negative.pop(matches[0])
        ordered.extend([pole, np.conj(pole)])
    if negative or len(ordered) != len(poles_np):
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


# Reference: Ackermann (1972), Regelungstechnik 20(7), pp. 297–300.
def _place_ackermann(A: Array, B: Array, poles: ArrayLike) -> Array:
    n = A.shape[0]
    coeffs = jnp.real(jnp.poly(jnp.asarray(poles)))
    eye = jnp.eye(n, dtype=A.dtype)
    p_A = coeffs[0] * eye
    for c in coeffs[1:]:
        p_A = p_A @ A + c * eye
    C_ctrb = _ctrb_matrix(A, B)
    e_n = jnp.zeros(n, dtype=A.dtype).at[-1].set(1.0)
    x = jnp.linalg.solve(C_ctrb.T, e_n)
    L = x @ p_A
    return L.reshape(1, n)


# References: Kautsky, Nichols & Van Dooren (1985), IJC 41, pp. 1129–1155;
# Tits & Yang, "Globally Convergent Algorithms for Robust Pole Assignment by
# State Feedback", UMD TR 95-3: https://hdl.handle.net/1903/5598.
def _place_robust(
    A: Array,
    B: Array,
    poles: ArrayLike,
    *,
    method: str,
    rtol: float,
    maxiter: int,
) -> tuple[Array, int, float]:
    poles = _order_complex_poles(poles)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if len(poles) != A.shape[0]:
        raise ValueError("place() requires exactly n poles.")

    U, singular, Vh = jnp.linalg.svd(B, full_matrices=True)
    rankB = int(jnp.sum(singular > singular[0] * max(B.shape) * jnp.finfo(B.dtype).eps))
    poles_np = np.asarray(poles)
    for pole in poles_np:
        if np.sum(poles_np == pole) > rankB:
            raise ValueError(
                "at least one requested pole is repeated more than rank(B) times"
            )
    if method == "KNV0" and not np.all(np.isreal(poles_np)):
        raise ValueError("Complex poles are not supported by KNV0.")

    u0 = U[:, :rankB]
    u1 = U[:, rankB:]

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
        gain_matrix = Vh[:rankB].T @ (
            (u0.T @ (diag_poles - A)) / singular[:rankB, None]
        )
        return -jnp.real(gain_matrix), 0, 0.0

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

    steps, cur_rtol = 0, (0.0 if rankB <= 1 else float("inf"))
    if rankB > 1:
        if method == "KNV0":
            for step in range(maxiter):
                log_before = jnp.linalg.slogdet(transfer_matrix)[1]
                for j in range(len(poles)):
                    transfer_matrix = _place_knv0_update(ker_pole, transfer_matrix, j)
                log_after = jnp.linalg.slogdet(transfer_matrix)[1]
                improvement = jnp.abs(jnp.expm1(log_before - log_after))
                cur_rtol = float(
                    jnp.where(jnp.isfinite(log_after), improvement, jnp.inf)
                )
                steps = step + 1
                if cur_rtol < rtol:
                    break
        elif method == "YT":
            update_order = _place_yt_update_order(poles)
            for step in range(maxiter):
                log_before = jnp.linalg.slogdet(transfer_matrix)[1]
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
                log_after = jnp.linalg.slogdet(transfer_matrix)[1]
                improvement = jnp.abs(jnp.expm1(log_before - log_after))
                cur_rtol = float(
                    jnp.where(jnp.isfinite(log_after), improvement, jnp.inf)
                )
                steps = step + 1
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
    gain_matrix = Vh[:rankB].T @ ((u0.T @ (m - A)) / singular[:rankB, None])
    return -jnp.real(gain_matrix), steps, cur_rtol
