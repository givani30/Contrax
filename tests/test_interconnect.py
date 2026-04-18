"""Tests for contrax.interconnect — series and parallel composition."""

import jax
import jax.numpy as jnp
import pytest

import contrax as cx


def test_series_discrete_matches_manual_block_formula():
    sys1 = cx.dss(
        A=jnp.array([[0.9]]),
        B=jnp.array([[1.2]]),
        C=jnp.array([[1.5]]),
        D=jnp.array([[0.4]]),
        dt=0.1,
    )
    sys2 = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[0.7]]),
        C=jnp.array([[1.1]]),
        D=jnp.array([[0.2]]),
        dt=0.1,
    )

    series_sys = cx.series(sys2, sys1)

    expected_A = jnp.array([[0.9, 0.0], [1.05, 0.8]])
    expected_B = jnp.array([[1.2], [0.28]])
    expected_C = jnp.array([[0.3, 1.1]])
    expected_D = jnp.array([[0.08]])

    assert jnp.allclose(series_sys.A, expected_A)
    assert jnp.allclose(series_sys.B, expected_B)
    assert jnp.allclose(series_sys.C, expected_C)
    assert jnp.allclose(series_sys.D, expected_D)


def test_parallel_discrete_matches_manual_block_formula():
    sys1 = cx.dss(
        A=jnp.array([[0.9]]),
        B=jnp.array([[1.2]]),
        C=jnp.array([[1.5]]),
        D=jnp.array([[0.4]]),
        dt=0.1,
    )
    sys2 = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[0.7]]),
        C=jnp.array([[1.1]]),
        D=jnp.array([[0.2]]),
        dt=0.1,
    )

    parallel_sys = cx.parallel(sys1, sys2)

    expected_A = jnp.array([[0.9, 0.0], [0.0, 0.8]])
    expected_B = jnp.array([[1.2], [0.7]])
    expected_C = jnp.array([[1.5, 1.1]])
    expected_D = jnp.array([[0.6]])

    assert jnp.allclose(parallel_sys.A, expected_A)
    assert jnp.allclose(parallel_sys.B, expected_B)
    assert jnp.allclose(parallel_sys.C, expected_C)
    assert jnp.allclose(parallel_sys.D, expected_D)


def test_subtraction_matches_parallel_with_negative_sign():
    sys1 = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[2.0]]),
        D=jnp.array([[0.5]]),
    )
    sys2 = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[0.3]]),
        C=jnp.array([[1.2]]),
        D=jnp.array([[0.1]]),
    )

    sub_sys = sys1 - sys2
    par_sys = cx.parallel(sys1, sys2, sign=-1.0)

    assert jnp.allclose(sub_sys.A, par_sys.A)
    assert jnp.allclose(sub_sys.B, par_sys.B)
    assert jnp.allclose(sub_sys.C, par_sys.C)
    assert jnp.allclose(sub_sys.D, par_sys.D)


def test_matmul_overload_matches_series():
    sys1 = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[2.0]]),
        D=jnp.array([[0.5]]),
    )
    sys2 = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[0.3]]),
        C=jnp.array([[1.2]]),
        D=jnp.array([[0.1]]),
    )

    overloaded = sys2 @ sys1
    named = cx.series(sys2, sys1)

    assert jnp.allclose(overloaded.A, named.A)
    assert jnp.allclose(overloaded.B, named.B)
    assert jnp.allclose(overloaded.C, named.C)
    assert jnp.allclose(overloaded.D, named.D)


def test_series_rejects_dimension_mismatch():
    sys1 = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0], [0.0]]),
        D=jnp.zeros((2, 1)),
    )
    sys2 = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[0.3]]),
        C=jnp.array([[1.2]]),
        D=jnp.array([[0.1]]),
    )

    with pytest.raises(ValueError, match="output dimension"):
        cx.series(sys2, sys1)


def test_parallel_rejects_dt_mismatch():
    sys1 = cx.dss(
        A=jnp.array([[0.9]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
        dt=0.1,
    )
    sys2 = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
        dt=0.2,
    )

    with pytest.raises(ValueError, match="matching dt"):
        cx.parallel(sys1, sys2)


def test_interconnect_rejects_mixed_continuous_and_discrete():
    sys_c = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
    )
    sys_d = cx.dss(
        A=jnp.array([[0.9]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
        dt=0.1,
    )

    with pytest.raises(TypeError, match="both systems"):
        cx.series(sys_c, sys_d)


def test_interconnect_is_jittable():
    sys1 = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[2.0]]),
        D=jnp.array([[0.5]]),
    )
    sys2 = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[0.3]]),
        C=jnp.array([[1.2]]),
        D=jnp.array([[0.1]]),
    )

    build = jax.jit(lambda s2, s1: cx.parallel(cx.series(s2, s1), s1))
    result = build(sys2, sys1)

    assert result.A.shape == (3, 3)
    assert result.B.shape == (3, 1)
    assert result.C.shape == (1, 3)


def test_discrete_interconnect_is_jittable_with_traced_dt():
    sys1 = cx.dss(
        A=jnp.array([[0.9]]),
        B=jnp.array([[1.2]]),
        C=jnp.array([[1.5]]),
        D=jnp.array([[0.4]]),
        dt=0.1,
    )
    sys2 = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[0.7]]),
        C=jnp.array([[1.1]]),
        D=jnp.array([[0.2]]),
        dt=0.1,
    )

    result = jax.jit(cx.series)(sys2, sys1)

    assert isinstance(result, cx.DiscLTI)
    assert result.A.shape == (2, 2)
    assert jnp.allclose(result.dt, sys1.dt)
