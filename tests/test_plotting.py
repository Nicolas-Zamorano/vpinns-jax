"""Tests for the training, solution, and cellwise error plots."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax_fem.diagnostics import energy_error_functional, l2_error_functional
from jax_fem.element import LagrangeTriangleP2
from jax_fem.function import FiniteElementFunction
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_finite_element_space
from matplotlib.tri import Triangulation, UniformTriRefiner

from vpinns_jax.analysis import (
    TrainingRecord,
    compute_network_error,
    plot_cell_errors,
    plot_fe_solution,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
)
from vpinns_jax.models import evaluate_network_values
from vpinns_jax.solvers import TrainingHistory


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _affine(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 2.0 * x[0] - 0.5 * x[1]


def _history() -> TrainingHistory:
    loss = jnp.array([4.0, 2.0, 1.0])
    error = jnp.array([1.0, 0.5])
    return TrainingHistory(
        loss=jnp.array([4.0, 3.0, 2.0, 1.5]),
        monitored_steps=jnp.array([0, 4]),
        monitored=TrainingRecord(
            loss=loss[1:],
            terms={"first": jnp.array([1.5, 0.5]), "second": jnp.array([0.5, 0.5])},
            energy_error=error,
            relative_loss=jnp.sqrt(loss[1:]) / 2.0,
            relative_energy_error=error / 2.0,
            robustness=jnp.sqrt(loss[1:]) / error,
        ),
    )


def _line_data(axis, index: int) -> tuple[np.ndarray, np.ndarray]:
    line = axis.get_lines()[index]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def test_training_quantities_plot_the_loss_and_relative_quantities() -> None:
    history = _history()
    loss_axis, relative_axis = plot_training_quantities(history).axes

    assert np.array_equal(_line_data(loss_axis, 0)[1], history.loss)
    steps, relative_loss = _line_data(relative_axis, 0)
    assert np.array_equal(steps, history.monitored_steps)
    assert np.allclose(relative_loss, history.monitored.relative_loss)
    assert np.allclose(
        _line_data(relative_axis, 1)[1], history.monitored.relative_energy_error
    )


def test_loss_terms_plot_the_loss_and_every_term() -> None:
    history = _history()
    (axis,) = plot_loss_terms(history).axes

    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "loss $L$",
        "first",
        "second",
    ]
    assert np.allclose(_line_data(axis, 2)[1], history.monitored.terms["second"])


def test_robustness_plot_shows_the_ratio() -> None:
    history = _history()
    (axis,) = plot_robustness(history).axes
    assert np.allclose(_line_data(axis, 1)[1], history.monitored.robustness)


@pytest.mark.parametrize(
    "plot", [plot_training_quantities, plot_loss_terms, plot_robustness]
)
def test_training_plots_require_a_training_record(plot) -> None:
    with pytest.raises(ValueError, match="TrainingRecord"):
        plot(TrainingHistory(loss=jnp.ones(3)))


def test_network_solution_plot_shows_the_pointwise_error(
    unit_square_mesh, sin_sin_problem
) -> None:
    figure = plot_network_solution(_affine, unit_square_mesh, sin_sin_problem)
    exact, approximation, error = (axis.collections[0] for axis in figure.axes[:3])

    assert len(figure.axes) == 6
    assert np.allclose(
        error.get_array(), np.abs(exact.get_array() - approximation.get_array())
    )
    assert exact.get_clim() == approximation.get_clim()


def test_fe_solution_plot_interpolates_the_p1_function_exactly(
    p1_basis, sin_sin_problem
) -> None:
    """The P1 function with the nodal values of an affine function equals it."""
    space = p1_basis.space
    nodal_values = evaluate_network_values(_affine, space.dof_coordinates)
    solution = FiniteElementFunction(
        space=space, dof_values=jnp.squeeze(nodal_values, axis=-2)
    )
    figure = plot_fe_solution(solution, sin_sin_problem, subdivisions=2)
    coordinates = np.asarray(space.mesh.vertex_coordinates)[..., 0, :]
    refined = UniformTriRefiner(
        Triangulation(
            coordinates[:, 0],
            coordinates[:, 1],
            np.asarray(space.mesh.cells_to_vertices),
        )
    ).refine_triangulation(subdiv=2)
    expected = 1.0 + 2.0 * refined.x - 0.5 * refined.y

    assert np.allclose(figure.axes[1].collections[0].get_array(), expected)


def test_fe_solution_plot_rejects_p2(unit_square_mesh, sin_sin_problem) -> None:
    space = create_finite_element_space(
        unit_square_mesh, LagrangeTriangleP2(reference_cell=ReferenceTriangle())
    )
    solution = FiniteElementFunction(
        space=space, dof_values=jnp.zeros((space.number_of_dofs, 1))
    )
    with pytest.raises(NotImplementedError, match="P1"):
        plot_fe_solution(solution, sin_sin_problem)


def test_cell_errors_plot_one_panel_per_norm(p1_basis, sin_sin_problem) -> None:
    cell_errors = {
        "L2": compute_network_error(
            _affine, p1_basis, sin_sin_problem, l2_error_functional, reduce=False
        ),
        "energy": compute_network_error(
            _affine, p1_basis, sin_sin_problem, energy_error_functional, reduce=False
        ),
    }
    figure = plot_cell_errors(p1_basis.space.mesh, cell_errors)
    panels = figure.axes[: len(cell_errors)]

    assert [axis.get_title() for axis in panels] == ["L2", "energy"]
    for axis, errors in zip(panels, cell_errors.values(), strict=True):
        assert np.allclose(axis.collections[0].get_array(), np.ravel(errors))
