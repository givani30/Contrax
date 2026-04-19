"""Continuous-model EKF with first-order-hold input interpolation.

A Van der Pol oscillator is driven by a piecewise-linear input. States are
estimated by an EKF that uses the continuous nonlinear model sampled with
first-order-hold (FOH) interpolation between measurement steps.

Demonstrates:
- ``sample_system()`` with ``input_interpolation="foh"`` to build a discrete
  transition model that linearly interpolates inputs within each sample period.
- ``foh_inputs()`` to package a raw input sequence into (u_k, u_{k+1}) pairs.
- Online EKF step helpers (``ekf_predict``, ``ekf_update``) for a loop that
  updates the covariance at each measurement.
"""

from __future__ import annotations

# --8<-- [start:setup]
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import contrax as cx

# Van der Pol oscillator: ẋ = [x2, μ(1-x1²)x2 - x1 + u]
MU = 1.0
DT = 0.1  # sample period
T_STEPS = 50  # number of steps
# --8<-- [end:setup]


# --8<-- [start:model]
def vdp_dynamics(t, x, u):
    """Continuous-time Van der Pol dynamics with additive scalar input."""
    x1, x2 = x[0], x[1]
    dx1 = x2
    dx2 = MU * (1 - x1**2) * x2 - x1 + u[0]
    return jnp.array([dx1, dx2])


def vdp_observation(t, x, u):
    """Observe the first state (position) only."""
    return x[:1]


nl = cx.nonlinear_system(
    vdp_dynamics, vdp_observation, dt=None, state_dim=2, input_dim=1, output_dim=1
)

# Build a discrete transition model via FOH integration
discrete_foh = cx.sample_system(nl, DT, input_interpolation="foh")
# --8<-- [end:model]


# --8<-- [start:generate-data]
def generate_trajectory(seed: int = 0):
    """Simulate ground-truth trajectory and noisy measurements."""
    key = jax.random.PRNGKey(seed)

    # Piecewise-linear input: slowly varying
    t_axis = jnp.linspace(0.0, T_STEPS * DT, T_STEPS)
    us_raw = 0.3 * jnp.sin(2 * jnp.pi * 0.5 * t_axis)[:, None]  # (T, 1)
    us_foh = cx.foh_inputs(us_raw)  # (T, 2, 1)

    # Simulate true trajectory using FOH discrete model
    x0_true = jnp.array([0.5, 0.0])
    xs_true = cx.rollout(
        lambda x, u: discrete_foh.dynamics(0.0, x, u), x0_true, us_foh
    )  # (T+1, 2)

    # Noisy measurements of first state only
    R_noise = jnp.array([[0.05]])
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T_STEPS + 1, 1)) * jnp.sqrt(R_noise[0, 0])
    ys = xs_true[:, :1] + noise  # (T+1, 1)

    return xs_true, ys, us_raw, us_foh


# --8<-- [end:generate-data]


# --8<-- [start:filter]
def run_ekf(xs_true, ys, us_foh):
    """Run an online EKF using FOH-sampled discrete model."""
    Q_noise = jnp.diag(jnp.array([1e-4, 1e-3]))
    R_noise = jnp.array([[0.05]])

    def h(x):
        return x[:1]

    def f_foh(x, u_pair):
        return discrete_foh.dynamics(0.0, x, u_pair)

    x = ys[0, :1]  # initialise from first measurement
    x = jnp.concatenate([x, jnp.zeros(1)])
    P = jnp.eye(2) * 0.5

    xs_est = [x]
    for k in range(T_STEPS):
        x, P = cx.ekf_predict(f_foh, x, P, us_foh[k], Q_noise)
        x, P, _ = cx.ekf_update(h, x, P, ys[k + 1], R_noise)
        xs_est.append(x)

    return jnp.stack(xs_est)  # (T+1, 2)


# --8<-- [end:filter]


def run_example():
    xs_true, ys, us_raw, us_foh = generate_trajectory()
    xs_est = run_ekf(xs_true, ys, us_foh)

    rmse_pos = float(jnp.sqrt(jnp.mean((xs_true[:, 0] - xs_est[:, 0]) ** 2)))
    rmse_vel = float(jnp.sqrt(jnp.mean((xs_true[:, 1] - xs_est[:, 1]) ** 2)))

    assert rmse_pos < 0.5, f"position RMSE too large: {rmse_pos:.4f}"
    assert rmse_vel < 1.0, f"velocity RMSE too large: {rmse_vel:.4f}"

    return {
        "rmse_position": rmse_pos,
        "rmse_velocity": rmse_vel,
        "n_steps": T_STEPS,
        "final_true": np.asarray(xs_true[-1]),
        "final_estimated": np.asarray(xs_est[-1]),
    }


def main():
    result = run_example()
    print("Continuous-model EKF with FOH input interpolation")
    print(f"  steps              = {result['n_steps']}")
    print(f"  position RMSE      = {result['rmse_position']:.4f}")
    print(f"  velocity RMSE      = {result['rmse_velocity']:.4f}")
    print(f"  true  final state  = {result['final_true']}")
    print(f"  estim final state  = {result['final_estimated']}")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
