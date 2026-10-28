"""Configuration objects for numerical solvers.

Solver objects hold static algorithm choices and tolerances. Close over a
configuration in ``jax.jit``; model matrices remain dynamic inputs.
"""

import math
from enum import IntEnum

import equinox as eqx

from contrax.types import KalmanGainResult


class SolverStatus(IntEnum):
    """JAX-compatible integer status codes shared by numerical results."""

    successful = 0
    max_steps_reached = 1
    invalid_problem = 2
    numerical_failure = 3


class Lyapunov(eqx.Module):
    """Configure a continuous- or discrete-time Lyapunov solve.

    ``auto`` uses squared Smith iteration with dense fallback when the state
    count is at most ``dense_limit``. ``doubling`` applies to stable systems;
    ``dense`` handles any uniquely solvable equation, including unstable A.
    Explicit dense selection permits O(n^4) storage and O(n^6) work.

    Attributes:
        method: ``auto``, ``doubling``, or ``dense``.
        rtol: Scale-aware equation residual tolerance.
        atol: Absolute equation residual tolerance.
        max_steps: Maximum doubling steps, including residual correction solves.
        dense_limit: Largest state dimension allowed for automatic fallback.
    """

    method: str = eqx.field(static=True, default="auto")
    rtol: float = eqx.field(static=True, default=1e-12)
    atol: float = eqx.field(static=True, default=1e-14)
    max_steps: int = eqx.field(static=True, default=64)
    dense_limit: int = eqx.field(static=True, default=8)

    def __check_init__(self):
        if self.method not in ("auto", "doubling", "dense"):
            raise ValueError("Lyapunov method must be auto, doubling, or dense")
        if (
            not math.isfinite(self.rtol)
            or not math.isfinite(self.atol)
            or self.rtol < 0
            or self.atol < 0
            or self.rtol + self.atol == 0
        ):
            raise ValueError(
                "Lyapunov tolerances must be nonnegative and not both zero"
            )
        if not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        if not isinstance(self.dense_limit, int) or self.dense_limit < 0:
            raise ValueError("dense_limit must be a nonnegative integer")


__all__ = [
    "Lyapunov",
    "DARE",
    "CARE",
    "Kalman",
    "SteadyStateKalman",
    "InformationKalman",
    "PolePlacement",
    "EKF",
    "UKF",
    "SolverStatus",
]


class DARE(eqx.Module):
    """Configure structured doubling for the discrete algebraic Riccati equation.

    Attributes:
        max_steps: Maximum structured doubling iterations.
        rtol: Relative iterate and equation residual tolerance.
        atol: Absolute iterate and equation residual tolerance.
    """

    max_steps: int = eqx.field(static=True, default=64)
    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-12)

    def __check_init__(self):
        Lyapunov(max_steps=self.max_steps, rtol=self.rtol, atol=self.atol)


class CARE(eqx.Module):
    """Configure a Hamiltonian stable-subspace CARE solve with Newton refinement.

    Refinement accepts corrections that reduce the residual and retain stable
    feedback. The Hamiltonian must provide a well-separated stable subspace.

    Attributes:
        refinement_steps: Maximum Newton corrections of the stabilizing candidate.
        rtol: Relative equation residual tolerance.
        atol: Absolute equation residual tolerance.
    """

    refinement_steps: int = eqx.field(static=True, default=3)
    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-12)

    def __check_init__(self):
        Lyapunov(max_steps=self.refinement_steps, rtol=self.rtol, atol=self.atol)


class Kalman(eqx.Module):
    """Covariance-form linear Kalman filter configuration.

    Attributes:
        covariance_update: ``joseph`` (default) or ``compact``. Joseph form
            retains the measurement-noise term explicitly and better preserves
            covariance symmetry and positive semidefiniteness under roundoff.
    """

    covariance_update: str = eqx.field(static=True, default="joseph")

    def __check_init__(self):
        if self.covariance_update not in ("joseph", "compact"):
            raise ValueError("covariance_update must be joseph or compact")


class SteadyStateKalman(eqx.Module):
    """Explicitly reuse a kalman_gain design for a fully observed LTI stream.

    Uses the gain design as differentiable data for a fully observed LTI stream.
    Batch filtering defaults P0 to design.P, the predicted covariance before
    the first update. A different P0, a missing observation, or an inconsistent
    model/noise pair reports `invalid_problem`; use covariance-form filtering
    when those cases are expected.

    Attributes:
        design: KalmanGainResult returned by `kalman_gain`. The gain and
            predicted covariance remain differentiable inputs.
        rtol: Relative tolerance for gain/fixed-point and initialization checks.
        atol: Absolute tolerance for those checks.
    """

    design: KalmanGainResult
    rtol: float = eqx.field(static=True, default=1e-8)
    atol: float = eqx.field(static=True, default=1e-10)

    def __check_init__(self):
        if not isinstance(self.design, KalmanGainResult):
            raise TypeError("SteadyStateKalman requires a kalman_gain design")
        Lyapunov(rtol=self.rtol, atol=self.atol)


class InformationKalman(eqx.Module):
    """Whitened state-space Kalman update for diagonal observation noise.

    Use a positive observation-variance vector `R` with shape `(p,)`. Prior
    whitening and state-size Cholesky solves make this formulation useful when
    observations outnumber states. Predicted covariances must be positive
    definite; controls and missing observations follow covariance-form Kalman
    conventions.
    """


class EKF(eqx.Module):
    """Extended Kalman measurement-update configuration.

    Attributes:
        covariance_update: ``joseph`` (default) or ``compact``.
        num_iter: Measurement linearization passes; one gives ordinary EKF.
            The prediction covariance stays fixed during iterated updates.
    """

    covariance_update: str = eqx.field(static=True, default="joseph")
    num_iter: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        Kalman(covariance_update=self.covariance_update)
        if not isinstance(self.num_iter, int) or self.num_iter < 1:
            raise ValueError("num_iter must be a positive integer")


class UKF(eqx.Module):
    """Scaled unscented transform for additive-noise filtering.

    Attributes:
        alpha: Positive sigma-point spread; defaults to 1.
        beta: Prior-distribution correction; 2 is standard for Gaussian priors.
        kappa: Secondary scaling; n+kappa must be positive for n states.

    Sigma-point covariances and innovation covariances must be positive
    definite. Supply covariances that satisfy this condition before filtering.
    """

    alpha: float = eqx.field(static=True, default=1.0)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if not all(math.isfinite(a) for a in (self.alpha, self.beta, self.kappa)):
            raise ValueError("UKF scaling parameters must be finite")
        if self.alpha <= 0:
            raise ValueError("UKF alpha must be positive")


class PolePlacement(eqx.Module):
    """Configure design-time pole assignment and acceptance tolerances.

    Attributes:
        method: auto chooses KNV0 for real poles and YT for complex pairs.
            Explicit KNV0 and YT selections are also accepted.
        rtol: Relative determinant-improvement convergence tolerance.
        max_steps: Maximum eigenvector-update sweeps.
        assignment_rtol: Scale-aware achieved-pole tolerance.
        assignment_atol: Absolute achieved-pole tolerance.
        condition_limit: Maximum controllability/eigenvector condition estimate.
        fallback_limit: Largest SISO dimension allowed for Ackermann fallback;
            zero disables fallback. Every fallback is validated independently.
    """

    method: str = eqx.field(static=True, default="auto")
    rtol: float = eqx.field(static=True, default=1e-3)
    max_steps: int = eqx.field(static=True, default=30)
    assignment_rtol: float = eqx.field(static=True, default=1e-6)
    assignment_atol: float = eqx.field(static=True, default=1e-8)
    condition_limit: float = eqx.field(static=True, default=1e12)
    fallback_limit: int = eqx.field(static=True, default=8)

    def __check_init__(self):
        if self.method not in ("auto", "KNV0", "YT"):
            raise ValueError("PolePlacement method must be auto, KNV0, or YT")
        Lyapunov(
            rtol=self.rtol,
            atol=0,
            max_steps=self.max_steps,
            dense_limit=self.fallback_limit,
        )
        Lyapunov(rtol=self.assignment_rtol, atol=self.assignment_atol)
        if not math.isfinite(self.condition_limit) or self.condition_limit < 1:
            raise ValueError("condition_limit must be finite and at least one")
