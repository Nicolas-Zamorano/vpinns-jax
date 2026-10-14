"""Tests for the training, solution, and cellwise error plots."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax_fem.diagnostics import energy_error_functional, l2_error_functional
from jax_fem.mesh import create_gmsh_l_shaped_mesh
from jax_fem.problem import create_poisson_l_shaped_singular_problem

from vpinns_jax.analysis import (
    TrainingRecord,
    compute_network_error,
    plot_cell_errors,
    plot_loss_terms,
    plot_network_solution,
    plot_robustness,
    plot_training_quantities,
)
from vpinns_jax.analysis.plotting import _evaluate_on_grid
from vpinns_jax.solvers import TrainingHistory

_OKABE_ITO = {
    "#000000",
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _affine(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 2.0 * x[0] - 0.5 * x[1]


def _history() -> TrainingHistory:
    loss = jnp.array([4.0, 1.0])
    error = jnp.array([1.0, 0.5])
    return TrainingHistory(
        loss=jnp.array([4.0, 3.0, 2.0, 1.5]),
        monitored_steps=jnp.array([0, 4]),
        monitored=TrainingRecord(
            loss=loss,
            terms={"first": jnp.array([3.5, 0.5]), "second": jnp.array([0.5, 0.5])},
            energy_error=error,
            relative_loss=jnp.sqrt(loss) / 2.0,
            relative_energy_error=error / 2.0,
            robustness=jnp.sqrt(loss) / error,
        ),
    )


def _line_data(axis, index: int) -> np.ndarray:
    return np.asarray(axis.get_lines()[index].get_ydata())


def _assert_readable(figure) -> None:
    """Okabe-Ito colors, one linestyle per line, grids, and tick labels."""
    lines = [line for axis in figure.axes for line in axis.get_lines()]
    styles = [line.get_linestyle() for line in lines]
    for line in lines:
        assert line.get_color().lower() in _OKABE_ITO
    assert len(set(styles)) == len(styles)
    for axis in figure.axes:
        assert axis.xaxis.get_gridlines()[0].get_visible()
        assert axis.yaxis.get_gridlines()[0].get_visible()
        assert any(label.get_visible() for label in axis.get_xticklabels())
        assert any(label.get_visible() for label in axis.get_yticklabels())


@pytest.mark.parametrize(
    "plot", [plot_training_quantities, plot_loss_terms, plot_robustness]
)
def test_training_plots_are_readable(plot) -> None:
    _assert_readable(plot(_history()))


def test_training_quantities_plot_the_loss_and_relative_quantities() -> None:
    history = _history()
    loss_axis, relative_axis = plot_training_quantities(history).axes

    assert np.array_equal(_line_data(loss_axis, 0), history.loss)
    assert np.array_equal(
        relative_axis.get_lines()[0].get_xdata(), history.monitored_steps
    )
    assert np.allclose(_line_data(relative_axis, 0), history.monitored.relative_loss)
    assert np.allclose(
        _line_data(relative_axis, 1), history.monitored.relative_energy_error
    )


def test_loss_terms_plot_the_loss_and_every_term() -> None:
    history = _history()
    (axis,) = plot_loss_terms(history).axes

    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "loss $L$",
        "first",
        "second",
    ]
    assert np.allclose(_line_data(axis, 0), history.monitored.loss)
    assert np.allclose(_line_data(axis, 2), history.monitored.terms["second"])


def test_robustness_plot_shows_the_ratio() -> None:
    history = _history()
    (axis,) = plot_robustness(history).axes
    assert np.allclose(_line_data(axis, 0), history.monitored.robustness)


@pytest.mark.parametrize(
    "plot", [plot_training_quantities, plot_loss_terms, plot_robustness]
)
def test_training_plots_require_a_training_record(plot) -> None:
    with pytest.raises(ValueError, match="TrainingRecord"):
        plot(TrainingHistory(loss=jnp.ones(3)))


def test_network_solution_surfaces_on_a_uniform_grid(
    unit_square_mesh, sin_sin_problem
) -> None:
    figure = plot_network_solution(
        _affine, unit_square_mesh, sin_sin_problem, points_per_side=11
    )
    panels = [axis for axis in figure.axes if axis.name == "3d"]
    exact, approximation, error = (axis.collections[0] for axis in panels)
    grid = np.linspace(0.0, 1.0, 11)

    assert len(panels) == 3
    assert [axis.get_zlabel() for axis in panels] == ["$u$", "$w$", "$|u - w|$"]
    # A surface on an 11 x 11 grid has 10 x 10 faces.
    assert exact.get_array().size == 100
    assert np.allclose(
        error.get_array(), np.abs(exact.get_array() - approximation.get_array())
    )
    assert np.allclose(panels[0].get_xlim(), (grid[0], grid[-1]), atol=0.1)


def test_network_solution_leaves_points_outside_the_domain_blank() -> None:
    mesh = create_gmsh_l_shaped_mesh(0.5)
    x, y, exact, approximation = _evaluate_on_grid(
        _affine, mesh, create_poisson_l_shaped_singular_problem(mesh), 21
    )
    outside = (x > 1e-12) & (y < -1e-12)

    assert np.all(np.isnan(approximation[outside]))
    assert np.all(np.isnan(exact[outside]))
    assert np.allclose(
        approximation[~outside], 1.0 + 2.0 * x[~outside] - 0.5 * y[~outside]
    )


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
        assert axis.xaxis.get_gridlines()[0].get_visible()


@pytest.mark.parametrize(
    "plot", [plot_training_quantities, plot_loss_terms, plot_robustness]
)
def test_training_plots_span_every_step(plot) -> None:
    """Four steps but two records: the axis follows the steps, not the records."""
    history = _history()
    for axis in plot(history).axes:
        assert axis.get_xlim() == (0.0, float(history.loss.shape[0]))


def test_network_solution_limits(unit_square_mesh, sin_sin_problem) -> None:
    figure = plot_network_solution(
        _affine, unit_square_mesh, sin_sin_problem, points_per_side=11
    )
    _, _, exact, approximation = _evaluate_on_grid(
        _affine, unit_square_mesh, sin_sin_problem, 11
    )
    exact_range = (np.nanmin(exact), np.nanmax(exact))
    error_range = (0.0, np.nanmax(np.abs(exact - approximation)))
    panels = [axis for axis in figure.axes if axis.name == "3d"]

    for axis, value_range in zip(
        panels, (exact_range, exact_range, error_range), strict=True
    ):
        assert np.allclose(axis.get_xlim(), (0.0, 1.0))
        assert np.allclose(axis.get_ylim(), (0.0, 1.0))
        assert np.allclose(axis.get_zlim(), value_range)
        assert np.allclose(axis.collections[0].get_clim(), value_range)


def test_cell_error_limits(p1_basis, sin_sin_problem) -> None:
    errors = compute_network_error(
        _affine, p1_basis, sin_sin_problem, l2_error_functional, reduce=False
    )
    (axis, _) = plot_cell_errors(p1_basis.space.mesh, {"L2": errors}).axes

    assert np.allclose(axis.get_xlim(), (0.0, 1.0))
    assert np.allclose(axis.get_ylim(), (0.0, 1.0))
    assert np.allclose(axis.collections[0].get_clim(), (0.0, np.max(errors)))
