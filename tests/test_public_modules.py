"""Tests for public namespace modules."""

import contrax as cx
import contrax.simulation as sim_api
import contrax.systems as systems_api
import contrax.types as types_api


def test_systems_namespace_exports_public_modeling_surface():
    assert systems_api.ss is cx.ss
    assert systems_api.dss is cx.dss
    assert systems_api.c2d is cx.c2d
    assert systems_api.linearize is cx.linearize
    assert systems_api.linearize_ss is cx.linearize_ss
    assert systems_api.nonlinear_system is cx.nonlinear_system
    assert systems_api.series is cx.series
    assert systems_api.parallel is cx.parallel
    assert systems_api.ContLTI is cx.ContLTI
    assert systems_api.DiscLTI is cx.DiscLTI
    assert systems_api.NonlinearSystem is cx.NonlinearSystem
    assert systems_api.PHSSystem is cx.PHSSystem
    assert systems_api.partition_state is cx.partition_state
    assert systems_api.block_observation is cx.block_observation
    assert systems_api.block_matrix is cx.block_matrix
    assert systems_api.symmetrize_matrix is cx.symmetrize_matrix
    assert systems_api.project_psd is cx.project_psd
    assert systems_api.phs_diagnostics is cx.phs_diagnostics


def test_simulation_namespace_exports_public_simulation_surface():
    assert sim_api.rollout is cx.rollout
    assert sim_api.foh_inputs is cx.foh_inputs
    assert sim_api.sample_system is cx.sample_system
    assert sim_api.lsim is cx.lsim
    assert sim_api.simulate is cx.simulate
    assert sim_api.step_response is cx.step_response
    assert sim_api.impulse_response is cx.impulse_response
    assert sim_api.initial_response is cx.initial_response
    assert sim_api.as_ode_term is cx.as_ode_term


def test_top_level_exports_parameterization_helpers():
    assert callable(cx.positive_exp)
    assert callable(cx.positive_softplus)
    assert callable(cx.lower_triangular)
    assert callable(cx.spd_from_cholesky_raw)
    assert callable(cx.diagonal_spd)


def test_types_namespace_exports_result_bundles():
    assert types_api.LQRResult is cx.LQRResult
    assert types_api.KalmanGainResult is cx.KalmanGainResult
    assert types_api.KalmanResult is cx.KalmanResult
    assert types_api.UKFResult is cx.UKFResult
    assert types_api.RTSResult is cx.RTSResult
    assert types_api.MHEResult is cx.MHEResult
    assert types_api.PHSStructureDiagnostics is cx.PHSStructureDiagnostics
    assert types_api.InnovationDiagnostics is cx.InnovationDiagnostics
    assert types_api.LikelihoodDiagnostics is cx.LikelihoodDiagnostics
