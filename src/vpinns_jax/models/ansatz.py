"""
Scalar neural network trial functions.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class MultiLayerPerceptron(eqx.Module):
    """
    Scalar multilayer perceptron.

    Attributes
    ----------
    net : eqx.nn.MLP
        Perceptron with a single output.

    Notes
    -----
    The network maps one point of shape ``(d,)`` to a scalar of shape ``()``.
    ``vpinns_jax.models.evaluation`` evaluates it at batches of row-vector
    points.
    """

    net: eqx.nn.MLP

    def __init__(
        self,
        *,
        input_dimension: int = 2,
        width: int = 32,
        depth: int = 3,
        activation: Callable[[jax.Array], jax.Array] = jnp.tanh,
        key: jax.Array,
    ) -> None:
        """
        Initialize the network.

        Parameters
        ----------
        input_dimension : int
            Dimension ``d`` of the input points.
        width : int
            Number of neurons in each hidden layer.
        depth : int
            Number of hidden layers.
        activation : Callable[[jax.Array], jax.Array]
            Activation of the hidden layers.
        key : jax.Array
            PRNG key for the initial weights.
        """
        self.net = eqx.nn.MLP(
            in_size=input_dimension,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the network at one point.

        Parameters
        ----------
        x : jax.Array
            Point, shape ``(d,)``.

        Returns
        -------
        value : jax.Array
            Network value, shape ``()``.
        """
        return jnp.squeeze(self.net(x), axis=-1)


class FourierFeatureMLP(eqx.Module):
    """
    Scalar multilayer perceptron on random Fourier features.

    Attributes
    ----------
    frequency_matrix : jax.Array
        Frequencies ``B``, shape ``(number_of_features, d)``.
    net : eqx.nn.MLP
        Perceptron acting on the features, with a single output.
    trainable_frequencies : bool
        Whether ``frequency_matrix`` receives gradients.

    Notes
    -----
    The input is lifted to ``[cos(2 pi B x), sin(2 pi B x)]`` before the
    perceptron, which counters the spectral bias of a plain perceptron
    against high-frequency components (Tancik et al., 2020).
    """

    frequency_matrix: jax.Array
    net: eqx.nn.MLP
    trainable_frequencies: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_dimension: int = 2,
        number_of_features: int = 64,
        frequency_scale: float = 1.0,
        trainable_frequencies: bool = False,
        width: int = 32,
        depth: int = 3,
        activation: Callable[[jax.Array], jax.Array] = jnp.tanh,
        key: jax.Array,
    ) -> None:
        """
        Initialize the network.

        Parameters
        ----------
        input_dimension : int
            Dimension ``d`` of the input points.
        number_of_features : int
            Number of frequencies; the feature vector has twice this length.
        frequency_scale : float
            Standard deviation of the Gaussian frequencies.
        trainable_frequencies : bool
            Whether the frequencies are trained.
        width : int
            Number of neurons in each hidden layer.
        depth : int
            Number of hidden layers.
        activation : Callable[[jax.Array], jax.Array]
            Activation of the hidden layers.
        key : jax.Array
            PRNG key for the frequencies and the initial weights.
        """
        frequency_key, net_key = jax.random.split(key)
        self.frequency_matrix = frequency_scale * jax.random.normal(
            frequency_key, (number_of_features, input_dimension)
        )
        self.trainable_frequencies = trainable_frequencies
        self.net = eqx.nn.MLP(
            in_size=2 * number_of_features,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=activation,
            key=net_key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the network at one point.

        Parameters
        ----------
        x : jax.Array
            Point, shape ``(d,)``.

        Returns
        -------
        value : jax.Array
            Network value, shape ``()``.
        """
        frequencies = self.frequency_matrix
        if not self.trainable_frequencies:
            frequencies = jax.lax.stop_gradient(frequencies)
        phases = 2.0 * jnp.pi * (frequencies @ x)
        features = jnp.concatenate((jnp.cos(phases), jnp.sin(phases)), axis=-1)
        return jnp.squeeze(self.net(features), axis=-1)
