"""Solvers and adaptive training loop (implementation plan Component 6)."""

from vpinns_jax.solvers.adaptive_loop import AdaptiveIterationResult, run_adaptive_loop
from vpinns_jax.solvers.trainer import TrainingHistory, train, train_adam, train_lbfgs

__all__ = [
    "TrainingHistory",
    "train_adam",
    "train_lbfgs",
    "train",
    "AdaptiveIterationResult",
    "run_adaptive_loop",
]
