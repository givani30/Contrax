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


def test_linearize_lqr_simulate_example():
    module = _load_module("examples/linearize_lqr_simulate.py")
    result = module.run_example()
    assert result["final_norm"] < result["initial_norm"]


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


def test_lqr_optimal_execution_example():
    module = _load_module("examples/lqr_optimal_execution.py")
    result = module.run_example()
    assert result["final_loss"] < result["initial_loss"]
    assert result["inventory_path"][-1] < 0.05
    assert result["sell_schedule"][0] > result["sell_schedule"][-1]
