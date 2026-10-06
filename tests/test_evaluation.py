"""Tests for the batched evaluation of networks and their flux divergence."""

import jax
import jax.numpy as jnp
import jax.random as jr

from vpinns_jax.models import (
    MultiLayerPerceptron,
    evaluate_network,
    evaluate_network_flux_divergence,
    evaluate_network_values,
)


def _cubic(x: jnp.ndarray) -> jnp.ndarray:
    return x[0] ** 2 * x[1] + 3.0 * x[1]


def _points(*batch_shape: int) -> jnp.ndarray:
    return jr.uniform(jr.PRNGKey(1), (*batch_shape, 1, 2))


def _linear_diffusion(points: jnp.ndarray) -> jnp.ndarray:
    scale = 1.0 + jnp.expand_dims(points[..., 0, 0], axis=(-2, -1))
    return scale * jnp.eye(2)


def test_shapes_with_leading_batch_dimensions() -> None:
    points = _points(2, 3, 4)
    evaluation = evaluate_network(_cubic, points)

    assert evaluate_network_values(_cubic, points).shape == (2, 3, 4, 1, 1)
    assert evaluation.values.shape == (2, 3, 4, 1, 1)
    assert evaluation.gradients.shape == (2, 3, 4, 1, 2)
    assert evaluation.divergence is None
    assert evaluate_network_flux_divergence(
        _cubic, _linear_diffusion, points
    ).shape == (2, 3, 4, 1, 1)


def test_values_and_gradients_match_closed_form() -> None:
    points = _points(5, 6)
    x, y = points[..., 0:1], points[..., 1:2]
    evaluation = evaluate_network(_cubic, points)

    assert jnp.allclose(evaluation.values, x**2 * y + 3.0 * y)
    expected_gradients = jnp.concatenate((2.0 * x * y, x**2 + 3.0), axis=-1)
    assert jnp.allclose(evaluation.gradients, expected_gradients)


def test_flux_divergence_with_constant_diffusion_is_scaled_laplacian() -> None:
    points = _points(5, 6)

    def diffusion(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(2.0 * jnp.eye(2), points.shape[:-2] + (2, 2))

    divergence = evaluate_network_flux_divergence(_cubic, diffusion, points)
    assert jnp.allclose(divergence, 4.0 * points[..., 1:2])


def test_flux_divergence_includes_the_diffusion_gradient() -> None:
    points = _points(5, 6)
    x, y = points[..., 0:1], points[..., 1:2]
    divergence = evaluate_network_flux_divergence(_cubic, _linear_diffusion, points)
    # grad(1 + x) . grad w + (1 + x) lap w with grad w = (2xy, x^2 + 3), lap w = 2y.
    assert jnp.allclose(divergence, 2.0 * x * y + (1.0 + x) * 2.0 * y)


def test_network_evaluation_matches_explicit_vmap(prng_key) -> None:
    network = MultiLayerPerceptron(width=8, depth=2, key=prng_key)
    points = _points(7, 3)
    values, gradients = jax.vmap(jax.vmap(jax.value_and_grad(network)))(
        jnp.squeeze(points, axis=-2)
    )
    evaluation = evaluate_network(network, points)

    assert jnp.allclose(jnp.squeeze(evaluation.values, axis=(-2, -1)), values)
    assert jnp.allclose(jnp.squeeze(evaluation.gradients, axis=-2), gradients)
