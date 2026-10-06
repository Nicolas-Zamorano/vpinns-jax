"""Tests for the Adam and L-BFGS trainers."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from vpinns_jax.solvers import train, train_adam, train_lbfgs

_TARGET = jnp.array([1.0, -2.0, 0.5])


class _Point(eqx.Module):
    value: jnp.ndarray
    label: str = eqx.field(static=True, default="point")


def _quadratic(point: _Point) -> jnp.ndarray:
    return jnp.sum((point.value - _TARGET) ** 2)


def _rosenbrock(point: _Point) -> jnp.ndarray:
    x, y = point.value[0], point.value[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def test_adam_minimizes_a_quadratic() -> None:
    trained, history = train_adam(
        _quadratic, _Point(jnp.zeros(3)), num_steps=200, learning_rate=0.1
    )
    assert history.loss.shape == (200,)
    assert jnp.allclose(history.loss[0], _quadratic(_Point(jnp.zeros(3))))
    assert _quadratic(trained) < 1e-3 * history.loss[0]
    assert trained.label == "point"


def test_lbfgs_minimizes_the_rosenbrock_function() -> None:
    trained, history = train_lbfgs(
        _rosenbrock, _Point(jnp.array([-1.2, 1.0])), num_steps=100
    )
    assert history.loss.shape == (100,)
    assert jnp.allclose(trained.value, jnp.array([1.0, 1.0]), atol=1e-6)


def test_early_stopping_ends_training() -> None:
    _, history = train_lbfgs(
        _quadratic, _Point(jnp.zeros(3)), num_steps=100, early_stopping_patience=3
    )
    assert history.loss.shape[0] < 100


def test_train_concatenates_adam_and_lbfgs_histories() -> None:
    trained, history = train(
        _quadratic, _Point(jnp.zeros(3)), adam_steps=5, lbfgs_steps=10
    )
    assert history.loss.shape == (15,)
    assert _quadratic(trained) < 1e-12


def test_zero_steps_are_rejected() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        train_adam(_quadratic, _Point(jnp.zeros(3)), num_steps=0)


def _value(point: _Point) -> dict[str, jnp.ndarray]:
    return {"value": point.value}


def test_monitor_is_recorded_every_few_steps_and_after_the_last() -> None:
    trained, history = train_adam(
        _quadratic,
        _Point(jnp.zeros(3)),
        num_steps=10,
        learning_rate=0.1,
        monitor=_value,
        monitor_every=4,
    )
    assert jnp.array_equal(history.monitored_steps, jnp.array([0, 4, 8, 10]))
    assert history.monitored["value"].shape == (4, 3)
    assert jnp.array_equal(history.monitored["value"][0], jnp.zeros(3))
    assert jnp.array_equal(history.monitored["value"][-1], trained.value)


def test_monitor_follows_early_stopping() -> None:
    trained, history = train_lbfgs(
        _quadratic,
        _Point(jnp.zeros(3)),
        num_steps=100,
        early_stopping_patience=3,
        monitor=_value,
        monitor_every=2,
    )
    assert history.monitored_steps[-1] == history.loss.shape[0]
    assert jnp.array_equal(history.monitored["value"][-1], trained.value)


def test_train_joins_monitor_records_without_duplicates() -> None:
    _, history = train(
        _quadratic,
        _Point(jnp.zeros(3)),
        adam_steps=4,
        lbfgs_steps=6,
        monitor=_value,
        monitor_every=2,
    )
    assert jnp.array_equal(history.monitored_steps, jnp.array([0, 2, 4, 6, 8, 10]))
    assert history.monitored["value"].shape == (6, 3)


def test_history_without_monitor_has_no_records() -> None:
    _, history = train_adam(_quadratic, _Point(jnp.zeros(3)), num_steps=3)
    assert history.monitored is None
    assert history.monitored_steps.shape == (0,)
