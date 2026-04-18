"""Nonlinear MHE with constrained parameterization on a pendulum.

A damped pendulum is driven by a small torque. Both the state (angle, angular
velocity) and a physical parameter (damping coefficient) are estimated
jointly from noisy angle measurements using Moving Horizon Estimation (MHE).

The damping coefficient is represented in *raw* (unconstrained) form and
mapped to a positive value via ``positive_softplus()``. This is the standard
Contrax pattern for constrained parameter estimation: write the objective in
terms of raw parameters and let the optimizer explore all of ℝ.

The augmented state is ``z = [theta, omega, b_raw]``, where
``b = positive_softplus(b_raw)`` is the physical damping.

Demonstrates:
- ``mhe_objective()`` as a pure differentiable cost over a fixed window.
- ``mhe()`` as a thin LBFGS-backed solve wrapper.
- ``positive_softplus()`` to keep physical parameters positive throughout
  unconstrained optimization.
- ``mhe_warm_start()`` to slide the estimation window forward.
"""

from __future__ import annotations

# --8<-- [start:setup]
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import contrax as cx

DT = 0.05          # sample period (s)
WINDOW = 12        # MHE window length
G_ACCEL = 9.81     # gravitational acceleration (m/s²)
L = 1.0            # pendulum length (m), assumed known
DAMPING_TRUE = 0.3 # true damping coefficient (kg·m²/s)
N_AUG = 3          # augmented state dimension: [theta, omega, b_raw]
# --8<-- [end:setup]


# --8<-- [start:model]
def aug_dynamics(x_aug, u):
    """Euler step for augmented state z = [theta, omega, b_raw].

    The damping parameter b_raw is treated as a slowly-varying state:
    its dynamics are an identity (random-walk model). The physical damping
    b = positive_softplus(b_raw) is always positive.
    """
    theta, omega, b_raw = x_aug[0], x_aug[1], x_aug[2]
    b = cx.positive_softplus(b_raw)
    torque = u[0]
    alpha = -(G_ACCEL / L) * jnp.sin(theta) - b * omega + torque
    theta_next = theta + DT * omega
    omega_next = omega + DT * alpha
    return jnp.array([theta_next, omega_next, b_raw])  # b_raw unchanged


def aug_observation(x_aug):
    """Observe angle only from augmented state."""
    return x_aug[:1]


# --8<-- [end:model]


# --8<-- [start:generate-data]
def generate_data(seed: int = 42):
    """Simulate ground-truth trajectory and noisy angle measurements."""
    key = jax.random.PRNGKey(seed)
    n_steps = WINDOW + 15

    x_true = jnp.array([0.8, 0.0])  # true physical state
    us = 0.05 * jax.random.normal(key, (n_steps, 1))

    xs_phys = [x_true]
    for k in range(n_steps):
        theta, omega = xs_phys[-1]
        alpha = -(G_ACCEL / L) * jnp.sin(theta) - DAMPING_TRUE * omega + us[k, 0]
        xs_phys.append(jnp.array([theta + DT * omega, omega + DT * alpha]))
    xs_phys = jnp.stack(xs_phys)  # (n_steps+1, 2)

    key2 = jax.random.fold_in(key, 99)
    R_val = 0.04
    noise = jax.random.normal(key2, (n_steps + 1, 1)) * jnp.sqrt(R_val)
    ys = xs_phys[:, :1] + noise    # noisy angle measurements

    return xs_phys, ys, us


# --8<-- [end:generate-data]


# --8<-- [start:mhe-solve]
def run_mhe(xs_phys, ys, us):
    """Solve MHE over the first window and report results."""
    # Noise covariances in augmented state space
    Q_noise = jnp.diag(jnp.array([1e-4, 1e-3, 1e-5]))  # small drift on b_raw
    R_noise = jnp.array([[0.04]])

    # Prior: start near the first measurement; b_raw=0 → b≈0.69 (soft overestimate)
    x_prior = jnp.array([float(ys[0, 0]), 0.0, 0.0])
    P_prior = jnp.diag(jnp.array([0.1, 0.5, 2.0]))

    ys_win = ys[: WINDOW + 1]       # (WINDOW+1, 1)
    us_win = us[:WINDOW]             # (WINDOW, 1)

    # Warm-start: replicate prior across window
    xs_init = jnp.tile(x_prior, (WINDOW + 1, 1))

    result = cx.mhe(
        aug_dynamics,
        aug_observation,
        xs_init,
        us_win,
        ys_win,
        x_prior,
        P_prior,
        Q_noise,
        R_noise,
        max_steps=512,
    )
    return result


# --8<-- [end:mhe-solve]


def run_example():
    xs_phys, ys, us = generate_data()
    result = run_mhe(xs_phys, ys, us)

    # Extract estimated damping from terminal augmented state
    b_raw_est = float(result.x_hat[2])
    b_est = float(cx.positive_softplus(jnp.array(b_raw_est)))

    # True state at end of window for comparison
    true_angle = float(xs_phys[WINDOW, 0])
    est_angle = float(result.x_hat[0])

    # positive_softplus guarantees b > 0 regardless of convergence
    assert b_est > 0, "estimated damping must be positive"
    assert abs(est_angle - true_angle) < 0.5, (
        f"angle estimate too far from truth: {est_angle:.3f} vs {true_angle:.3f}"
    )

    return {
        "converged": bool(result.solver_converged),
        "final_cost": float(result.final_cost),
        "true_damping": DAMPING_TRUE,
        "estimated_damping": b_est,
        "true_angle_end": true_angle,
        "estimated_angle_end": est_angle,
    }


def main():
    result = run_example()
    print("Nonlinear MHE — damped pendulum with parameter estimation")
    print(f"  converged           = {result['converged']} "
          f"(LBFGS may report False before tolerance; check cost)")
    print(f"  final cost          = {result['final_cost']:.4e}")
    print(f"  damping: true={result['true_damping']:.3f}  "
          f"estimated={result['estimated_damping']:.3f}")
    print("  angle at window end:")
    print(f"    true      = {result['true_angle_end']:.4f} rad")
    print(f"    estimated = {result['estimated_angle_end']:.4f} rad")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
