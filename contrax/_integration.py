"""Map native integration outcomes to Contrax status codes."""

import jax.numpy as jnp


def _diffrax_status(solution):
    import diffrax

    return jnp.where(
        solution.result == diffrax.RESULTS.successful,
        0,
        jnp.where(solution.result == diffrax.RESULTS.max_steps_reached, 1, 3),
    )
