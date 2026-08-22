"""Neural trial representations and boundary conditions (implementation plan Component 2)."""

from vpinns_jax.models.ansatz import FourierFeatureMLP, MultiLayerPerceptron
from vpinns_jax.models.boundary import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    InterpolationData,
    interpolated_trial_dofs,
    interpolated_trial_function,
    precompute_interpolation_data,
)

__all__ = [
    "MultiLayerPerceptron",
    "FourierFeatureMLP",
    "ExactDistanceFunction",
    "ConstrainedAnsatz",
    "InterpolationData",
    "precompute_interpolation_data",
    "interpolated_trial_dofs",
    "interpolated_trial_function",
]
