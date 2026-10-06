"""
Neural trial functions, their evaluation, and Dirichlet imposition.
"""

from vpinns_jax.models.ansatz import FourierFeatureMLP, MultiLayerPerceptron
from vpinns_jax.models.boundary import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    interpolate_network,
)
from vpinns_jax.models.evaluation import (
    Network,
    evaluate_network,
    evaluate_network_flux_divergence,
    evaluate_network_values,
)

__all__ = [
    "Network",
    "MultiLayerPerceptron",
    "FourierFeatureMLP",
    "evaluate_network_values",
    "evaluate_network",
    "evaluate_network_flux_divergence",
    "ExactDistanceFunction",
    "ConstrainedAnsatz",
    "interpolate_network",
]
