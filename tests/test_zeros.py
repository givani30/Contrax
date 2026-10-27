"""Finite invariant zeros, including rectangular and nonminimal pencils.

The regular square-pencil oracle uses SciPy only in tests. Analytic fixtures
use scalar transfer factors and state/input/output equivalence transformations.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import eigvals

import contrax as cx


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("outputs,inputs", [(1, 1), (3, 1), (1, 3), (3, 2), (2, 3)])
@pytest.mark.parametrize("feedthrough", [False, True])
def test_rectangular_rank_one_transfer_factor(discrete, outputs, inputs, feedthrough):
    """G(s)=outer(c,b)*(1/(s+2)+d), zero -5 for d=1/3, none for d=0."""
    B = jnp.arange(1, inputs + 1, dtype=float)[None, :]
    C = jnp.arange(1, outputs + 1, dtype=float)[:, None]
    D = C @ B / 3 if feedthrough else jnp.zeros((outputs, inputs))
    if discrete:
        sys = cx.dss(jnp.array([[-2.0]]), B, C, D, dt=0.1)
    else:
        sys = cx.ss(jnp.array([[-2.0]]), B, C, D)
    actual = cx.zeros(sys)
    expected = [-5.0] if feedthrough else []
    np.testing.assert_allclose(np.sort_complex(actual), expected, atol=1e-10)


@pytest.mark.parametrize("rank_d", [0, 1, 2, 3])
def test_regular_square_pencil_matches_independent_generalized_eigensolver(rank_d):
    """Finite generalized eigenvalues of [[A,B],[C,D]]-z*diag(I,0)."""
    rng = np.random.default_rng(741 + rank_d)
    n, m = 5, 3
    A, B, C = rng.normal(size=(n, n)), rng.normal(size=(n, m)), rng.normal(size=(m, n))
    left, _ = np.linalg.qr(rng.normal(size=(m, m)))
    right, _ = np.linalg.qr(rng.normal(size=(m, m)))
    D = left @ np.diag([1.0] * rank_d + [0.0] * (m - rank_d)) @ right
    pencil_A = np.block([[A, B], [C, D]])
    pencil_E = np.diag([1.0] * n + [0.0] * m)
    reference = eigvals(pencil_A, pencil_E)
    reference = reference[np.isfinite(reference)]
    actual = np.asarray(cx.zeros(cx.ss(A, B, C, D)))
    assert actual.shape == reference.shape
    # Match unordered spectra without depending on the signs of tiny imaginary
    # roundoff in a real root or conjugate pair.
    for value in reference:
        index = np.argmin(abs(actual - value))
        np.testing.assert_allclose(actual[index], value, atol=1e-8, rtol=1e-8)
        actual = np.delete(actual, index)


def test_nonminimal_decoupling_zeros_are_retained():
    """Minimal transfer has zero -3.5; disconnected state adds invariant zero -7."""
    A = jnp.diag(jnp.array([-2.0, -7.0]))
    B, C, D = jnp.array([[1.0], [0.0]]), jnp.array([[3.0, 0.0]]), jnp.array([[2.0]])
    np.testing.assert_allclose(np.sort_complex(cx.zeros(cx.ss(A, B, C, D))), [-7, -3.5])


def test_slicot_ab08nd_rectangular_nonminimal_reference_and_dual():
    """SLICOT AB08ND published example: finite invariant zeros are {-1, 2}.

    https://www.slicot.org/objects/software/shared/doc/AB08ND.html
    The fixture includes unreachable and unobservable modes and a left
    Kronecker block; the dual exercises right singular structure as well.
    """
    A = jnp.diag(jnp.array([1.0, 1.0, 3.0, -4.0, -1.0, 3.0]))
    B = jnp.array(
        [[0.0, -1.0], [-1.0, 0.0], [1.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]
    )
    C = jnp.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    D = jnp.zeros((3, 2))
    for sys in (cx.ss(A, B, C, D), cx.ss(A.T, C.T, B.T, D.T)):
        np.testing.assert_allclose(np.sort_complex(cx.zeros(sys)), [-1, 2], atol=1e-10)


def test_identically_zero_transfer_removes_right_singular_structure():
    """Only the unreachable -7 mode drops the rank of [sI-A, B]."""
    A = jnp.diag(jnp.array([-2.0, -7.0]))
    B, C, D = jnp.array([[1.0], [0.0]]), jnp.zeros((2, 2)), jnp.zeros((2, 1))
    np.testing.assert_allclose(cx.zeros(cx.ss(A, B, C, D)), [-7], atol=1e-12)


def test_state_input_output_basis_changes_preserve_zeros():
    """Equivalence transformations of a rank-deficient rectangular pencil."""
    A = jnp.diag(jnp.array([-2.0, -7.0]))
    B, C = (
        jnp.array([[1.0, 2.0], [0.0, 0.0]]),
        jnp.array([[3.0, 0.0], [6.0, 0.0], [9.0, 0.0]]),
    )
    D = C @ B / 3
    T = jnp.array([[2.0, 0.5], [0.1, 1.0]])
    U = jnp.array([[1.0, 2.0], [0.0, 1.0]])
    V = jnp.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    sys = cx.ss(
        jnp.linalg.solve(T, A @ T), jnp.linalg.solve(T, B @ U), V @ C @ T, V @ D @ U
    )
    np.testing.assert_allclose(np.sort_complex(cx.zeros(sys)), [-7, -5], atol=1e-9)


def test_static_feedthrough_has_no_finite_zeros_and_bad_tolerances_raise():
    sys = cx.ss(
        jnp.empty((0, 0)), jnp.empty((0, 2)), jnp.empty((3, 0)), jnp.ones((3, 2))
    )
    assert cx.zeros(sys).shape == (0,)
    with pytest.raises(ValueError, match="tolerances"):
        cx.zeros(sys, rtol=-1)
