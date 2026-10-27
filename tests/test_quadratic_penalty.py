"""Scalar and matrix weights denote the same quadratic cost convention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


def test_scalar_and_isotropic_matrix_weights_agree_and_differentiate():
    residuals = jnp.array([[1.0, -2.0], [0.5, 1.0]])
    scalar = cx.soft_quadratic_penalty(residuals, 0.5)
    matrix = jax.jit(cx.soft_quadratic_penalty)(residuals, 0.5 * jnp.eye(2))
    np.testing.assert_allclose(scalar, matrix)
    gradient = jax.grad(
        lambda r: cx.soft_quadratic_penalty(r, jnp.diag(jnp.array([2.0, 4.0])))
    )(residuals)
    np.testing.assert_allclose(gradient, [[4.0, -16.0], [2.0, 8.0]])
    weight_gradient = jax.grad(lambda W: cx.soft_quadratic_penalty(residuals, W))(
        jnp.eye(2)
    )
    np.testing.assert_allclose(weight_gradient, [[1.25, -1.5], [-1.5, 5.0]])
    assert cx.soft_quadratic_penalty(jnp.empty((0, 2)), jnp.eye(2)) == 0
    with pytest.raises(ValueError, match="dimension"):
        cx.soft_quadratic_penalty(residuals, jnp.eye(3))
