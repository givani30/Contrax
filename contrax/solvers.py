"""Algorithm configuration for Contrax numerical solvers.

Solver objects contain static algorithm choices. Close over them in ``jax.jit``
(or pass them through Equinox transforms); model matrices remain dynamic data.
Native Diffrax and Optimistix solvers retain their own interfaces.
"""

import math
from enum import IntEnum

import equinox as eqx


class SolverStatus(IntEnum):
    """JAX-compatible integer status codes shared by numerical results."""

    successful = 0
    max_steps_reached = 1
    invalid_problem = 2
    numerical_failure = 3


class Lyapunov(eqx.Module):
    """Configure a continuous or discrete Lyapunov solve.

    ``auto`` tries squared Smith iteration, then a dense solve only for systems
    with at most ``dense_limit`` states. ``doubling`` requires a stable system;
    ``dense`` supports any uniquely solvable equation, including unstable A.
    Explicit dense selection permits O(n^4) storage and O(n^6) work.

    Attributes:
        method: ``auto``, ``doubling``, or ``dense``.
        rtol: Scale-aware equation residual tolerance.
        atol: Absolute equation residual tolerance.
        max_steps: Maximum doubling steps.
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


__all__ = ["Lyapunov", "DARE", "CARE", "Kalman", "EKF", "UKF", "SolverStatus"]


class DARE(eqx.Module):
    """Structured doubling for the discrete algebraic Riccati equation.

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
    """Hamiltonian stable-subspace CARE solve with Newton refinement.

    Refinement retains a candidate only if its residual decreases and its
    feedback remains stabilizing. It cannot repair a missing stable subspace.

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

    Sigma-point covariances and innovation covariances must be positive definite.
    The filter adds no diagonal jitter or other covariance repair.
    """

    alpha: float = eqx.field(static=True, default=1.0)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if not all(math.isfinite(a) for a in (self.alpha, self.beta, self.kappa)):
            raise ValueError("UKF scaling parameters must be finite")
        if self.alpha <= 0:
            raise ValueError("UKF alpha must be positive")
