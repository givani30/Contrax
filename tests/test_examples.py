"""Smoke tests for runnable example scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sc42095_reference_example():
    module = _load_module("examples/sc42095_reference.py")
    result = module.run_example()
    assert result["phi00"] > 0.0


def test_linearize_lqr_simulate_example():
    module = _load_module("examples/linearize_lqr_simulate.py")
    result = module.run_example()
    assert result["final_norm"] < result["initial_norm"]


def test_differentiable_lqr_example():
    module = _load_module("examples/differentiable_lqr.py")
    result = module.run_example(num_steps=30, learning_rate=0.08)
    assert result["final_cost"] < result["initial_cost"]


def test_kalman_filtering_example():
    module = _load_module("examples/kalman_filtering.py")
    result = module.run_example()
    assert abs(result["final_filtered_position"] - result["final_measurement"]) < 0.1
    assert result["innovation_norm"] < 1.5


def test_continuous_nonlinear_estimation_example():
    module = _load_module("examples/continuous_nonlinear_estimation.py")
    result = module.run_example()
    assert result["smoothed_theta_rmse"] < result["filtered_theta_rmse"]
    assert result["smoothed_rate_rmse"] < result["filtered_rate_rmse"]
    assert result["max_condition_number"] < 1e3


def test_structured_nonlinear_estimation_example():
    module = _load_module("examples/structured_nonlinear_estimation.py")
    result = module.run_example()
    assert result["smoothed_q_rmse"] < result["filtered_q_rmse"]
    assert result["smoothed_p_rmse"] < result["filtered_p_rmse"]
    assert result["mean_nis"] > 0.0
    assert result["dissipation_min_eigenvalue"] >= -1e-9


def test_continuous_lqr_example():
    import numpy as np

    module = _load_module("examples/continuous_lqr.py")
    result = module.run_example()
    assert result["stable"]
    assert np.allclose(result["final_state"], 0.0, atol=1e-2)
    assert np.isfinite(result["gradient"])


def test_lqr_optimal_execution_example():
    module = _load_module("examples/lqr_optimal_execution.py")
    result = module.run_example()
    assert result["final_loss"] < result["initial_loss"]
    assert result["inventory_path"][-1] < 0.05
    assert result["sell_schedule"][0] > result["sell_schedule"][-1]


def test_pendulum_gif_example():
    module = _load_module("examples/pendulum_gif.py")
    # 30 steps captures the fast initial convergence without full run time.
    # No GIF is rendered — only the gradient loop is exercised.
    result = module.run_example(n_steps=30, snapshot_at=(0, 15, 29))
    assert result["costs"][-1] < result["costs"][0]
    assert len(result["snapshots"]) == 3
    # Final snapshot should accumulate substantially less theta² than the initial one.
    import numpy as np

    initial_sq = float(np.sum(result["snapshots"][0][:, 0] ** 2))
    final_sq = float(np.sum(result["snapshots"][-1][:, 0] ** 2))
    assert final_sq < 0.5 * initial_sq, (
        f"Expected >50% reduction in trajectory cost, got {final_sq / initial_sq:.2f}"
    )
