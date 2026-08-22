"""Neural trial-function architectures.

Every ansatz here is a scalar map ``x: (d,) -> u_theta(x): ()``: a *single-
point* evaluator (not batched), matching the convention
``jax.vmap(model)(points)`` used throughout the rest of the library (models,
losses, patch quadrature) to batch it over quadrature points or DOF
coordinates.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class MultiLayerPerceptron(eqx.Module):
    """Scalar MLP trial function ``u_theta: R^d -> R``, configurable depth/width/activation."""

    net: eqx.nn.MLP

    def __init__(
        self,
        *,
        input_dimension: int = 2,
        width: int = 32,
        depth: int = 3,
        activation: Callable[[jax.Array], jax.Array] = jnp.tanh,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(
            in_size=input_dimension,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)[0]


class FourierFeatureMLP(eqx.Module):
    r"""MLP over Fourier features ``[cos(2 pi B x), sin(2 pi B x)]``.

    Lifts the input through a random (by default fixed, optionally
    trainable) frequency matrix ``B`` before the MLP, which resolves
    high-frequency solution components an ordinary MLP is spectrally biased
    against (Tancik et al., 2020).
    """

    frequency_matrix: jax.Array  # (num_fourier_features, input_dimension)
    net: eqx.nn.MLP
    trainable_frequencies: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_dimension: int = 2,
        num_fourier_features: int = 64,
        frequency_scale: float = 1.0,
        trainable_frequencies: bool = False,
        width: int = 32,
        depth: int = 3,
        activation: Callable[[jax.Array], jax.Array] = jnp.tanh,
        key: jax.Array,
    ):
        frequency_key, net_key = jax.random.split(key)
        self.frequency_matrix = frequency_scale * jax.random.normal(
            frequency_key, (num_fourier_features, input_dimension)
        )
        self.trainable_frequencies = trainable_frequencies
        self.net = eqx.nn.MLP(
            in_size=2 * num_fourier_features,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=activation,
            key=net_key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        frequencies = self.frequency_matrix
        if not self.trainable_frequencies:
            frequencies = jax.lax.stop_gradient(frequencies)
        projection = 2.0 * jnp.pi * (frequencies @ x)
        features = jnp.concatenate((jnp.cos(projection), jnp.sin(projection)))
        return self.net(features)[0]
