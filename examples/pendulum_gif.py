"""
Pendulum stabilization: gradient descent through a differentiable Contrax loop.

Differentiates through ss → c2d → lqr → simulate to tune LQR cost weights
automatically. The system is a lightly damped pendulum (omega=3 rad/s, zeta=0.05)
displaced 57° from equilibrium. The initial controller (large R, small Q) barely
damps the oscillations. Gradient descent drives the design toward crisp active
damping — settling from 6+ seconds of oscillation to under 1 second.

Renders a side-by-side GIF:
  left  — pendulum stick figure replayed at each captured gradient step
  right — cost convergence curve with a moving marker

Run:   uv run python examples/pendulum_gif.py
Out:   docs/assets/images/pendulum_lqr.gif

Dependencies: matplotlib, pillow (dev deps; not required by the library itself).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import contrax as cx

# ── system parameters ─────────────────────────────────────────────────────────
#
# Lightly damped pendulum near stable equilibrium (hanging down).
# State: [theta (rad), theta_dot (rad/s)], theta=0 is equilibrium.
# Natural frequency 3 rad/s, damping ratio 0.05 → visible multi-second oscillations.

OMEGA = 3.0  # natural frequency (rad/s)
ZETA = 0.05  # damping ratio
DT = 0.05  # discretization period (s)
T_SIM = 60  # simulation horizon (steps → 3 seconds)
X0 = jnp.array([1.0, 0.0])  # 1 rad ≈ 57° displacement, at rest


# ── system ────────────────────────────────────────────────────────────────────

A_cont = jnp.array([[0.0, 1.0], [-(OMEGA**2), -2.0 * ZETA * OMEGA]])
B_cont = jnp.array([[0.0], [1.0]])

sys_c = cx.ss(A_cont, B_cont, jnp.eye(2), jnp.zeros((2, 1)))
sys_d = cx.c2d(sys_c, dt=DT)


# ── differentiable cost ───────────────────────────────────────────────────────


def closed_loop_cost(log_q_diag: jnp.ndarray, log_r: jnp.ndarray) -> jnp.ndarray:
    """LQR design cost: sum of squared angle deviation over the simulated trajectory.

    log_q_diag: shape (2,) — log diagonal entries of Q.
    log_r:      scalar    — log of the scalar R weight.
    """
    Q = jnp.diag(jnp.exp(log_q_diag))
    R = jnp.exp(log_r)[None, None]  # scalar → (1, 1)
    K = cx.lqr(sys_d, Q, R).K
    _, xs, _ = cx.simulate(sys_d, X0, lambda t, x: -K @ x, num_steps=T_SIM)
    return jnp.sum(xs[:, 0] ** 2)


def run_example(
    n_steps: int = 50,
    lr: float = 0.15,
    snapshot_at: tuple[int, ...] = (0, 3, 7, 12, 49),
) -> dict:
    """Run the gradient descent loop and collect snapshots.

    Returns a dict with cost history, snapshot trajectories, and snapshot
    iteration indices — suitable for rendering or for the test harness.
    """
    # Start passive: large R (expensive control), small Q (weak state penalty).
    # This gives a barely-damped response — the pendulum oscillates for 6+ seconds.
    log_q = jnp.array([-2.0, -2.0])  # Q ≈ 0.135 · I
    log_r = jnp.array(3.0)  # R ≈ 20  (scalar)

    val_and_grad = jax.jit(jax.value_and_grad(closed_loop_cost, argnums=(0, 1)))

    @jax.jit
    def _simulate_trajectory(lq, lr_):
        Q = jnp.diag(jnp.exp(lq))
        R = jnp.exp(lr_)[None, None]
        K = cx.lqr(sys_d, Q, R).K
        _, xs, _ = cx.simulate(sys_d, X0, lambda t, x: -K @ x, num_steps=T_SIM)
        return xs

    costs: list[float] = []
    snapshots: list[np.ndarray] = []

    for step in range(n_steps):
        if step in snapshot_at:
            snapshots.append(np.array(_simulate_trajectory(log_q, log_r)))
        cost, (gq, gr) = val_and_grad(log_q, log_r)
        costs.append(float(cost))
        log_q = log_q - lr * gq
        log_r = log_r - lr * gr

    costs.append(float(jax.jit(closed_loop_cost)(log_q, log_r)))

    return {
        "costs": costs,
        "snapshots": snapshots,
        "snapshot_at": list(snapshot_at),
        "n_steps": n_steps,
        "log_q_final": np.array(log_q),
        "log_r_final": np.array(log_r),
    }


# ── animation ─────────────────────────────────────────────────────────────────


def render_gif(result: dict, out_path: str, fps: int = 20, dpi: int = 110) -> None:
    """Render the pendulum GIF from a run_example() result dict."""
    costs = result["costs"]
    snapshots = result["snapshots"]
    snapshot_at = result["snapshot_at"]
    n_steps = result["n_steps"]

    # ── frame map: (snapshot_idx, sim_step) ──────────────────────────────────
    FRAMES_PER_SIM = T_SIM + 1
    PAUSE_FRAMES = 8  # hold last frame before next replay

    frame_map: list[tuple[int, int]] = []
    for i in range(len(snapshots)):
        for s in range(FRAMES_PER_SIM):
            frame_map.append((i, s))
        for _ in range(PAUSE_FRAMES):
            frame_map.append((i, FRAMES_PER_SIM - 1))

    # ── style ─────────────────────────────────────────────────────────────────
    BG = "#111318"
    ACCENT = "#4fc3f7"
    GRID_COLOR = "#2a2d35"
    TEXT_COLOR = "#c8ccd4"
    CURVE_COLOR = "#3a6b80"
    ROD_COLOR = "#8ab4c8"

    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor=BG)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.88, wspace=0.42)

    # ── left: pendulum ────────────────────────────────────────────────────────
    ax_p.set_facecolor(BG)
    ax_p.set_xlim(-1.25, 1.25)
    ax_p.set_ylim(-1.15, 0.2)
    ax_p.set_aspect("equal")
    ax_p.axis("off")
    ax_p.set_title("pendulum angle trajectory", color=TEXT_COLOR, fontsize=10, pad=6)

    # Pivot at origin, pendulum hangs downward
    ax_p.plot(0, 0, "o", color="#555", markersize=7, zorder=3)
    ax_p.axhline(0, color=GRID_COLOR, linewidth=1.2, xmin=0.1, xmax=0.9)

    (rod_line,) = ax_p.plot(
        [], [], "-", color=ROD_COLOR, linewidth=3.5, solid_capstyle="round", zorder=4
    )
    (bob_dot,) = ax_p.plot([], [], "o", color=ACCENT, markersize=14, zorder=5)
    iter_label = ax_p.text(
        0.5,
        0.04,
        "",
        transform=ax_p.transAxes,
        ha="center",
        color="#888",
        fontsize=8.5,
    )

    # ── right: cost curve ─────────────────────────────────────────────────────
    ax_c.set_facecolor(BG)
    iters = np.arange(len(costs))
    ax_c.plot(iters, costs, color=CURVE_COLOR, linewidth=1.8, zorder=1)
    ax_c.set_xlim(0, n_steps)
    ax_c.set_ylim(0, costs[0] * 1.08)
    ax_c.set_xlabel("gradient step", color=TEXT_COLOR, fontsize=9)
    ax_c.set_ylabel(r"trajectory cost  $\sum\theta^2$", color=TEXT_COLOR, fontsize=9)
    ax_c.set_title("differentiable LQR tuning", color=TEXT_COLOR, fontsize=10, pad=6)
    ax_c.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax_c.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    (cost_marker,) = ax_c.plot([], [], "o", color=ACCENT, markersize=8, zorder=3)
    cost_vline = ax_c.axvline(x=0, color=ACCENT, linewidth=0.9, alpha=0.35)

    # ── update function ───────────────────────────────────────────────────────
    def update(frame_idx: int):
        snap_idx, sim_step = frame_map[frame_idx]
        theta = float(snapshots[snap_idx][sim_step, 0])
        grad_iter = snapshot_at[snap_idx]

        # Pendulum hangs from pivot at origin: bob at (sin θ, -cos θ)
        bx = np.sin(theta)
        by = -np.cos(theta)
        rod_line.set_data([0, bx], [0, by])
        bob_dot.set_data([bx], [by])
        iter_label.set_text(f"gradient step {grad_iter} / {n_steps - 1}")

        cost_marker.set_data([grad_iter], [costs[grad_iter]])
        cost_vline.set_xdata([grad_iter])

        return rod_line, bob_dot, iter_label, cost_marker, cost_vline

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_map),
        interval=1000 // fps,
        blit=True,
    )
    ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close()
    print(f"Saved {out_path}  ({len(frame_map)} frames, {fps} fps)")


def main():
    result = run_example()
    render_gif(result, out_path="docs/assets/images/pendulum_lqr.gif")
    costs = result["costs"]
    print(f"initial cost: {costs[0]:.3f}")
    print(f"final cost:   {costs[-1]:.3f}")
    print(f"reduction:    {100 * (1 - costs[-1] / costs[0]):.1f}%")


if __name__ == "__main__":
    main()
