# --8<-- [start:script]
"""Quadratic optimal execution as a discrete LQR example."""

from __future__ import annotations

import jax

# --8<-- [start:setup]
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import contrax as cx

DT = 1.0
HORIZON = 20
X0 = jnp.array([1.0])


def build_execution_system(dt: float = DT) -> cx.DiscLTI:
    """Inventory dynamics with signed inventory-change control.

    State:
        x_k = remaining inventory, normalized so x_0 = 1 means 100%.

    Control:
        u_k = signed inventory change. Selling corresponds to u_k < 0.

    Dynamics:
        x_{k+1} = x_k + u_k
    """

    A = jnp.array([[1.0]])
    B = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    D = jnp.zeros((1, 1))
    return cx.dss(A, B, C, D, dt=dt)


SYS = build_execution_system()
# --8<-- [end:setup]


# --8<-- [start:baseline-execution]
def execution_schedule(
    inventory_risk: jax.Array,
    trading_cost: jax.Array,
    *,
    x0: jax.Array = X0,
    horizon: int = HORIZON,
):
    """Solve the execution problem and return the resulting liquidation path."""

    Q = jnp.array([[inventory_risk]])
    R = jnp.array([[trading_cost]])
    result = cx.lqr(SYS, Q, R)

    def controller(t, x):
        return -result.K @ x

    ts, xs, _ = cx.simulate(SYS, x0, controller, num_steps=horizon)
    inventory = xs[:, 0]
    # With x[k+1] = x[k] + u[k], a sell quantity is -u[k] = x[k] - x[k+1].
    sell_quantity = inventory[:-1] - inventory[1:]
    return result, ts, inventory, sell_quantity


# --8<-- [end:baseline-execution]


# --8<-- [start:differentiable-tuning]
def target_inventory_curve(horizon: int = HORIZON) -> jax.Array:
    """Reference curve: liquidate most of the position over the horizon."""

    steps = jnp.arange(horizon + 1, dtype=jnp.float64)
    return jnp.exp(-0.22 * steps)


def execution_tracking_loss(log_inventory_risk, log_trading_cost):
    """Tune LQR weights so the inventory path matches a desired urgency."""

    inventory_risk = jnp.exp(log_inventory_risk)
    trading_cost = jnp.exp(log_trading_cost)
    _, _, inventory, sell_quantity = execution_schedule(
        inventory_risk,
        trading_cost,
    )
    target = target_inventory_curve()
    inventory_error = jnp.mean((inventory - target) ** 2)
    turnover_penalty = 1e-2 * jnp.mean(sell_quantity**2)
    terminal_penalty = 10.0 * inventory[-1] ** 2
    return inventory_error + turnover_penalty + terminal_penalty


def tune_execution_weights(num_steps: int = 50, learning_rate: float = 0.15):
    params = (jnp.array(-2.0), jnp.array(-2.0))
    objective_and_grad = jax.jit(
        jax.value_and_grad(execution_tracking_loss, argnums=(0, 1))
    )

    initial_loss, _ = objective_and_grad(*params)
    history = [float(initial_loss)]

    for _ in range(num_steps):
        loss, grads = objective_and_grad(*params)
        dq, dr = grads
        params = (
            params[0] - learning_rate * dq,
            params[1] - learning_rate * dr,
        )
        history.append(float(loss))

    final_loss = float(execution_tracking_loss(*params))
    return {
        "initial_loss": history[0],
        "final_loss": final_loss,
        "inventory_risk": float(jnp.exp(params[0])),
        "trading_cost": float(jnp.exp(params[1])),
        "loss_history": np.asarray(history),
    }


# --8<-- [end:differentiable-tuning]


# --8<-- [start:batched-design]
@jax.jit
def batched_first_trade(inventory_risks, trading_costs):
    def solve_one(q, r):
        _, _, _, sells = execution_schedule(q, r, horizon=HORIZON)
        return sells[0]

    return jax.vmap(solve_one)(inventory_risks, trading_costs)


# --8<-- [end:batched-design]


def run_example():
    baseline, _, inventory, sell_quantity = execution_schedule(
        inventory_risk=jnp.array(2.5),
        trading_cost=jnp.array(0.4),
    )
    tuned = tune_execution_weights()

    batched_sells = batched_first_trade(
        jnp.array([1.0, 2.5, 5.0]),
        jnp.array([0.8, 0.4, 0.2]),
    )

    assert np.all(np.diff(np.asarray(inventory)) <= 1e-12)
    assert inventory[-1] < 0.05
    assert np.all(np.asarray(sell_quantity) >= 0.0)
    assert tuned["final_loss"] < tuned["initial_loss"]

    return {
        "baseline_gain": np.asarray(baseline.K),
        "inventory_path": np.asarray(inventory),
        "sell_schedule": np.asarray(sell_quantity),
        "tuned_inventory_risk": tuned["inventory_risk"],
        "tuned_trading_cost": tuned["trading_cost"],
        "initial_loss": tuned["initial_loss"],
        "final_loss": tuned["final_loss"],
        "batched_first_sells": np.asarray(batched_sells),
    }


def main():
    result = run_example()
    print("LQR optimal execution")
    print(f"baseline gain          = {result['baseline_gain']}")
    print(f"initial tuning loss    = {result['initial_loss']:.6f}")
    print(f"final tuning loss      = {result['final_loss']:.6f}")
    print(f"tuned inventory risk   = {result['tuned_inventory_risk']:.6f}")
    print(f"tuned trading cost     = {result['tuned_trading_cost']:.6f}")
    print(f"first sell quantities  = {result['sell_schedule'][:5]}")
    print(f"batched first sells    = {result['batched_first_sells']}")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
