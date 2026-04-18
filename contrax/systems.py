"""Public systems namespace.

This module gathers the system-model concepts that users typically look for
first: LTI containers and constructors, nonlinear model containers, basic
interconnection, and linearization helpers.
"""

from contrax.core import ContLTI, DiscLTI, c2d, dss, linearize, linearize_ss, ss
from contrax.interconnect import parallel, series
from contrax.nonlinear import NonlinearSystem, nonlinear_system
from contrax.phs import (
    PHSSystem,
    block_matrix,
    block_observation,
    canonical_J,
    partition_state,
    phs_diagnostics,
    phs_system,
    phs_to_ss,
    project_psd,
    schedule_phs,
    symmetrize_matrix,
)

__all__ = [
    "ContLTI",
    "DiscLTI",
    "NonlinearSystem",
    "PHSSystem",
    "partition_state",
    "block_observation",
    "block_matrix",
    "symmetrize_matrix",
    "project_psd",
    "phs_diagnostics",
    "ss",
    "dss",
    "c2d",
    "linearize",
    "linearize_ss",
    "nonlinear_system",
    "phs_system",
    "phs_to_ss",
    "canonical_J",
    "schedule_phs",
    "series",
    "parallel",
]
