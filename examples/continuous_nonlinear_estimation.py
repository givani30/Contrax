# --8<-- [start:script]
"""Continuous-time nonlinear estimation with FOH-sampled inputs."""

from __future__ import annotations

import jax

# --8<-- [start:setup]
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import contrax as cx

DT = 0.1
DURATION = 4.0
NUM_STEPS = int(DURATION / DT)
NOISE_KEY = jr.PRNGKey(7)
X0_TRUE = jnp.array([0.7, -0.1])
X0_EST = jnp.array([0.45, 0.2])
P0 = jnp.diag(jnp.array([0.08, 0.12]))
Q_NOISE = 6e-4 * jnp.eye(2)
R_NOISE = jnp.array([[6e-3]])
MEASUREMENT_STD = jnp.sqrt(R_NOISE[0, 0])
SAMPLE_TIMES = jnp.arange(NUM_STEPS, dtype=jnp.float64) * DT
INPUT_SAMPLES = (
    0.55 * jnp.sin(1.25 * SAMPLE_TIMES) + 0.2 * jnp.cos(0.55 * SAMPLE_TIMES + 0.3)
)[:, None]
INPUTS_FOH = cx.foh_inputs(INPUT_SAMPLES)


def pendulum_dynamics(t, x, u):
    theta, theta_dot = x
    torque = u[0]
    return jnp.array([theta_dot, -0.35 * theta_dot - jnp.sin(theta) + torque])


def angle_sensor(t, x, u):
    del t, u
    return x[:1]


SYS_CONT = cx.nonlinear_system(
    pendulum_dynamics,
    observation=angle_sensor,
    state_dim=2,
    input_dim=1,
    output_dim=1,
)
SYS_DISC = cx.sample_system(SYS_CONT, DT, input_interpolation="foh")
# --8<-- [end:setup]


def torque_profile(t):
    t_clipped = jnp.clip(t, 0.0, DURATION - 1e-9)
    step = jnp.minimum(jnp.floor(t_clipped / DT).astype(jnp.int32), NUM_STEPS - 1)
    tau = t_clipped - step.astype(jnp.float64) * DT
    u0, u1 = INPUTS_FOH[step]
    return (u0 + (tau / DT) * (u1 - u0)).reshape(1)


def sample_true_trajectory():
    ts_cont, xs_cont, _ = cx.simulate(
        SYS_CONT,
        X0_TRUE,
        lambda t, x: torque_profile(t),
        duration=DURATION,
        dt=DT / 5.0,
    )
    xs_sampled = jnp.stack(
        [
            jnp.interp(SAMPLE_TIMES, ts_cont, xs_cont[:, 0]),
            jnp.interp(SAMPLE_TIMES, ts_cont, xs_cont[:, 1]),
        ],
        axis=1,
    )
    return xs_sampled


def sample_measurements(x_true):
    noise = MEASUREMENT_STD * jr.normal(NOISE_KEY, (NUM_STEPS, 1), dtype=x_true.dtype)
    return x_true[:, :1] + noise


def summarize_rmse(filtered, smoothed, truth):
    filtered_theta_rmse = float(
        jnp.sqrt(jnp.mean((filtered.x_hat[:, 0] - truth[:, 0]) ** 2))
    )
    smoothed_theta_rmse = float(
        jnp.sqrt(jnp.mean((smoothed.x_smooth[:, 0] - truth[:, 0]) ** 2))
    )
    filtered_rate_rmse = float(
        jnp.sqrt(jnp.mean((filtered.x_hat[:, 1] - truth[:, 1]) ** 2))
    )
    smoothed_rate_rmse = float(
        jnp.sqrt(jnp.mean((smoothed.x_smooth[:, 1] - truth[:, 1]) ** 2))
    )
    return {
        "filtered_theta_rmse": filtered_theta_rmse,
        "smoothed_theta_rmse": smoothed_theta_rmse,
        "filtered_rate_rmse": filtered_rate_rmse,
        "smoothed_rate_rmse": smoothed_rate_rmse,
    }


# --8<-- [start:run-example]
def run_example():
    x_true = sample_true_trajectory()
    ys = sample_measurements(x_true)

    filtered = cx.ukf(
        SYS_DISC,
        Q_noise=Q_NOISE,
        R_noise=R_NOISE,
        ys=ys,
        us=INPUTS_FOH,
        x0=X0_EST,
        P0=P0,
        alpha=0.45,
    )
    smoothed = cx.uks(
        SYS_DISC,
        filtered,
        Q_noise=Q_NOISE,
        us=INPUTS_FOH,
        alpha=0.45,
    )
    innovation_diag, likelihood_diag = cx.ukf_diagnostics(filtered)
    rmse = summarize_rmse(filtered, smoothed, x_true)

    assert filtered.x_hat.shape == (NUM_STEPS, 2)
    assert filtered.predicted_measurements.shape == (NUM_STEPS, 1)
    assert smoothed.x_smooth.shape == (NUM_STEPS, 2)
    assert rmse["smoothed_theta_rmse"] < rmse["filtered_theta_rmse"]
    assert rmse["smoothed_rate_rmse"] < rmse["filtered_rate_rmse"]

    return {
        "times": SAMPLE_TIMES,
        "inputs": INPUT_SAMPLES[:, 0],
        "measurements": ys[:, 0],
        "true_theta": x_true[:, 0],
        "filtered_theta": filtered.x_hat[:, 0],
        "smoothed_theta": smoothed.x_smooth[:, 0],
        "true_rate": x_true[:, 1],
        "filtered_rate": filtered.x_hat[:, 1],
        "smoothed_rate": smoothed.x_smooth[:, 1],
        **rmse,
        "mean_nis": float(innovation_diag.mean_nis),
        "max_condition_number": float(
            innovation_diag.max_innovation_cov_condition_number
        ),
        "total_log_likelihood": float(likelihood_diag.total_log_likelihood),
    }


# --8<-- [end:run-example]


def main():
    result = run_example()
    print("Continuous nonlinear estimation")
    print(f"filtered theta rmse     = {result['filtered_theta_rmse']:.6f}")
    print(f"smoothed theta rmse     = {result['smoothed_theta_rmse']:.6f}")
    print(f"filtered rate rmse      = {result['filtered_rate_rmse']:.6f}")
    print(f"smoothed rate rmse      = {result['smoothed_rate_rmse']:.6f}")
    print(f"mean NIS                = {result['mean_nis']:.6f}")
    print(f"max innovation cond     = {result['max_condition_number']:.6f}")
    print(f"total log likelihood    = {result['total_log_likelihood']:.6f}")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
