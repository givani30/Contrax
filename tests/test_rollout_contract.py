"""Generic trajectories retain state/input pytrees and scalar leaves."""

import jax
import jax.numpy as jnp
import numpy as np

import contrax as cx


def test_rollout_state_and_input_pytrees_match_analytic_trajectory():
    initial = {"position": jnp.ones(1), "energy": jnp.array(0.0)}
    inputs = {"force": jnp.zeros((4, 1))}

    def transition(state, control, a):
        return {
            "position": a * state["position"] + control["force"],
            "energy": state["energy"] + jnp.sum(state["position"] ** 2),
        }

    solve = jax.jit(lambda a: cx.rollout(transition, initial, inputs, a))
    result = solve(jnp.array(0.8))
    np.testing.assert_allclose(result["position"][:, 0], 0.8 ** jnp.arange(5))
    expected_energy = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.8 ** (2 * jnp.arange(4)))]
    )
    np.testing.assert_allclose(result["energy"], expected_energy)
    assert jnp.isfinite(jax.grad(lambda a: solve(a)["energy"][-1])(jnp.array(0.8)))
    empty = cx.rollout(transition, initial, {"force": jnp.empty((0, 1))}, 0.8)
    assert empty["position"].shape == (1, 1) and empty["energy"].shape == (1,)
