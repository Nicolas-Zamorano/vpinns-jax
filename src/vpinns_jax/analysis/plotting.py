"""
Plots of training histories, network solutions, and cellwise errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.mesh import DfnMesh, TriangleMesh, create_fracture_frames
from jax_fem.problem import EllipticProblem

from vpinns_jax.analysis.monitoring import TrainingRecord
from vpinns_jax.models.evaluation import Network, evaluate_network_values
from vpinns_jax.solvers.trainer import TrainingHistory

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Okabe-Ito colorblind-safe palette in its usual order.
_OKABE_ITO = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)
_LINESTYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1, 1, 1, 1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 5)),
)


def _pyplot() -> Any:
    """
    Import ``matplotlib.pyplot`` on first use.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    import matplotlib.pyplot as pyplot

    return pyplot


def line_style(index: int) -> dict[str, Any]:
    """
    Color and linestyle of the ``index``-th line of a figure.

    Notes
    -----
    All line plots of the package use it, so custom plots that follow it
    match them: Okabe-Ito colors and one linestyle per line.

    Parameters
    ----------
    index : int
        Position of the line in the figure.

    Returns
    -------
    style : dict[str, Any]
        ``color`` from the Okabe-Ito palette and ``linestyle``, distinct for
        the first eight lines.
    """
    return {
        "color": _OKABE_ITO[index % len(_OKABE_ITO)],
        "linestyle": _LINESTYLES[index % len(_LINESTYLES)],
    }


def show_grid(axis: Axes) -> None:
    """
    Draw major and minor gridlines.

    Parameters
    ----------
    axis : Axes
        Two-dimensional axes.
    """
    axis.grid(True, which="major")
    axis.grid(True, which="minor", linestyle=":", alpha=0.4)


def _training_record(history: TrainingHistory) -> TrainingRecord:
    """
    Recorded training quantities of a history.

    Parameters
    ----------
    history : TrainingHistory
        History of a training with a ``create_training_monitor`` monitor.

    Returns
    -------
    record : TrainingRecord
        The quantities, each of shape ``(number_of_records,)``.

    Raises
    ------
    ValueError
        If the history was recorded without such a monitor.
    """
    if not isinstance(history.monitored, TrainingRecord):
        raise ValueError(
            "The history has no TrainingRecord; train with "
            "monitor=create_training_monitor(...)."
        )
    return history.monitored


def plot_training_quantities(history: TrainingHistory) -> Figure:
    """
    Loss, relative loss, and relative energy error over the training steps.

    Parameters
    ----------
    history : TrainingHistory
        History of a training with a ``create_training_monitor`` monitor.

    Returns
    -------
    figure : Figure
        Two panels sharing the step axis, from zero to the number of steps:
        the loss ``L`` at every step, and the relative loss
        ``sqrt(L) / |u|_A`` with the relative energy error
        ``|u - w|_A / |u|_A`` at the monitored steps.

    Notes
    -----
    The loss and the relative quantities have different scales, so they are
    drawn on separate axes; the two relative quantities share theirs, and
    their ratio is the robustness of ``plot_robustness``.
    """
    record = _training_record(history)
    steps = np.asarray(history.monitored_steps)
    pyplot = _pyplot()
    figure, (loss_axis, relative_axis) = pyplot.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True
    )
    loss_axis.semilogy(np.asarray(history.loss), label="loss $L$", **line_style(0))
    loss_axis.set_ylabel("loss")
    loss_axis.set_title("Loss")
    loss_axis.set_xlim(0, history.loss.shape[-1])
    loss_axis.legend()

    relative_axis.semilogy(
        steps,
        np.asarray(record.relative_loss),
        label=r"relative loss $\sqrt{L} / \|u\|_A$",
        **line_style(1),
    )
    relative_axis.semilogy(
        steps,
        np.asarray(record.relative_energy_error),
        label=r"relative energy error $\|u - w\|_A / \|u\|_A$",
        **line_style(2),
    )
    relative_axis.set_xlabel("step")
    relative_axis.set_ylabel(r"relative to $\|u\|_A$")
    relative_axis.set_title("Relative loss and relative energy error")
    relative_axis.legend()
    for axis in (loss_axis, relative_axis):
        show_grid(axis)
        axis.tick_params(labelbottom=True)
    return figure


def plot_loss_terms(history: TrainingHistory) -> Figure:
    """
    Loss and its terms over the training steps.

    Parameters
    ----------
    history : TrainingHistory
        History of a training with a ``create_training_monitor`` monitor.

    Returns
    -------
    figure : Figure
        The loss and each of its terms at the monitored steps, with the step
        axis from zero to the number of steps.
    """
    record = _training_record(history)
    steps = np.asarray(history.monitored_steps)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    axis.semilogy(steps, np.asarray(record.loss), label="loss $L$", **line_style(0))
    for index, (name, values) in enumerate(record.terms.items(), start=1):
        axis.semilogy(steps, np.asarray(values), label=name, **line_style(index))
    axis.set_xlabel("step")
    axis.set_ylabel("value")
    axis.set_title("Loss terms")
    axis.set_xlim(0, history.loss.shape[-1])
    axis.legend()
    show_grid(axis)
    return figure


def plot_robustness(history: TrainingHistory) -> Figure:
    """
    Ratio of the relative loss to the relative energy error over training.

    Parameters
    ----------
    history : TrainingHistory
        History of a training with a ``create_training_monitor`` monitor.

    Returns
    -------
    figure : Figure
        ``sqrt(L) / |u - w|_A`` at the monitored steps, with a reference line
        at one and the step axis from zero to the number of steps.

    Notes
    -----
    A loss that is a robust estimator of the energy error keeps this ratio
    bounded above and below during training. A ratio drifting to zero means
    the loss decreases while the error does not.
    """
    record = _training_record(history)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    axis.semilogy(
        np.asarray(history.monitored_steps),
        np.asarray(record.robustness),
        label=r"$\sqrt{L} / \|u - w\|_A$",
        **line_style(0),
    )
    axis.axhline(1.0, label="1", linewidth=1.0, **line_style(1))
    axis.set_xlabel("step")
    axis.set_ylabel(r"$\sqrt{L} / \|u - w\|_A$")
    axis.set_title("Robustness: relative loss / relative energy error")
    axis.set_xlim(0, history.loss.shape[-1])
    axis.legend()
    show_grid(axis)
    return figure


def _evaluate_on_grid(
    network: Network,
    mesh: TriangleMesh,
    problem: EllipticProblem,
    points_per_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact solution and network on a uniform grid over a planar domain.

    Parameters
    ----------
    network : Network
        Trial function of one point.
    mesh : TriangleMesh
        Planar mesh; it only defines the domain.
    problem : EllipticProblem
        Problem providing the exact solution.
    points_per_side : int
        Number of grid points along each side of the bounding box.

    Returns
    -------
    x : np.ndarray
        Abscissae of the grid, shape ``(points_per_side, points_per_side)``.
    y : np.ndarray
        Ordinates of the grid, same shape.
    exact : np.ndarray
        Exact solution, ``NaN`` outside the domain, same shape.
    approximation : np.ndarray
        Network values, ``NaN`` outside the domain, same shape.

    Raises
    ------
    NotImplementedError
        If the mesh is not planar.
    ValueError
        If the problem has no exact solution.
    """
    from matplotlib.tri import Triangulation

    if mesh.geometric_dimension != 2:
        raise NotImplementedError(
            "plot_network_solution needs a planar domain; for fracture networks "
            "use jax_fem.diagnostics.plot_fracture_solutions."
        )
    if problem.exact_solution is None:
        raise ValueError("problem.exact_solution is required for the plot.")
    coordinates = np.asarray(mesh.vertex_coordinates)[..., 0, :]
    lower = np.min(coordinates, axis=0)
    upper = np.max(coordinates, axis=0)
    x, y = np.meshgrid(
        np.linspace(lower[0], upper[0], points_per_side),
        np.linspace(lower[1], upper[1], points_per_side),
    )
    triangulation = Triangulation(
        coordinates[:, 0], coordinates[:, 1], np.asarray(mesh.cells_to_vertices)
    )
    inside = triangulation.get_trifinder()(x, y) >= 0
    points = jnp.expand_dims(jnp.stack((x, y), axis=-1), axis=-2)
    exact = jnp.squeeze(problem.exact_solution(points), axis=(-2, -1))
    approximation = jnp.squeeze(evaluate_network_values(network, points), axis=(-2, -1))
    return (
        x,
        y,
        np.where(inside, np.asarray(exact), np.nan),
        np.where(inside, np.asarray(approximation), np.nan),
    )


def plot_network_solution(
    network: Network,
    mesh: TriangleMesh,
    problem: EllipticProblem,
    *,
    points_per_side: int = 101,
) -> Figure:
    """
    Exact solution, network solution, and pointwise absolute error as surfaces.

    Parameters
    ----------
    network : Network
        Trial function of one point, e.g. a ``ConstrainedAnsatz``.
    mesh : TriangleMesh
        Planar mesh; it only defines the domain.
    problem : EllipticProblem
        Problem providing the exact solution.
    points_per_side : int
        Number of grid points along each side of the bounding box.

    Returns
    -------
    figure : Figure
        Three 3D panels over the bounding box of the domain: ``u`` and ``w``
        with the range of ``u`` as vertical and color limits, and
        ``|u - w|`` from zero to its maximum.

    Notes
    -----
    The functions are evaluated on a uniform grid over the bounding box of
    the domain, independent of the mesh, so the plot shows the network
    itself rather than a mesh interpolant. Grid points outside the domain,
    located through the mesh triangulation, are left blank, and the parts of
    ``w`` beyond the range of ``u`` are clipped.
    """
    x, y, exact, approximation = _evaluate_on_grid(
        network, mesh, problem, points_per_side
    )
    error = np.abs(exact - approximation)
    exact_range = (float(np.nanmin(exact)), float(np.nanmax(exact)))
    error_range = (0.0, float(np.nanmax(error)))
    panels = (
        ("Exact solution", "$u$", exact, exact_range),
        ("VPINN solution", "$w$", approximation, exact_range),
        ("Pointwise absolute error", "$|u - w|$", error, error_range),
    )
    pyplot = _pyplot()
    figure = pyplot.figure(figsize=(18.0, 5.5), constrained_layout=True)
    for index, (title, label, values, (low, high)) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        # Draw every grid point; the default subsampling to 50 rows and
        # columns produces staircase artifacts on non-convex domains.
        axis.plot_surface(
            x,
            y,
            values,
            cmap="viridis",
            vmin=low,
            vmax=high,
            linewidth=0.0,
            rcount=points_per_side,
            ccount=points_per_side,
            axlim_clip=True,
        )
        axis.set_title(title)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.set_zlabel(label)
        axis.set_xlim(np.min(x), np.max(x))
        axis.set_ylim(np.min(y), np.max(y))
        if high > low:
            axis.set_zlim(low, high)
    return figure


def _cell_error_panels(
    mesh: TriangleMesh | DfnMesh,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Planar coordinates and cells of every panel row of a cellwise plot.

    Parameters
    ----------
    mesh : TriangleMesh | DfnMesh
        Planar mesh, or fracture network drawn fracture by fracture.

    Returns
    -------
    rows : list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]
        For each row: a title prefix, vertex coordinates in the plane of the
        row, shape ``(N_vertices, 2)``, the cells of the row, shape
        ``(K_row, 3)``, and their indices in the mesh, shape ``(K_row,)``.

    Notes
    -----
    A fracture is drawn in its own plane, at the coordinates ``x @ Q_i`` of
    its tangent frame from ``create_fracture_frames``.
    """
    coordinates = np.asarray(mesh.vertex_coordinates)[..., 0, :]
    cells = np.asarray(mesh.cells_to_vertices)
    if not isinstance(mesh, DfnMesh):
        return [("", coordinates, cells, np.arange(cells.shape[0]))]
    frames = np.asarray(create_fracture_frames(mesh))
    cells_to_fractures = np.asarray(mesh.cells_to_fractures)
    rows = []
    for fracture, tag in enumerate(np.asarray(mesh.fracture_tags).tolist()):
        fracture_cells = np.flatnonzero(cells_to_fractures == fracture)
        rows.append(
            (
                f"fracture {tag}: ",
                coordinates @ frames[fracture],
                cells[fracture_cells],
                fracture_cells,
            )
        )
    return rows


def plot_cell_errors(
    mesh: TriangleMesh | DfnMesh, cell_errors: Mapping[str, jax.Array]
) -> Figure:
    """
    Cellwise error norms over the mesh.

    Parameters
    ----------
    mesh : TriangleMesh | DfnMesh
        Mesh whose cells carry the errors.
    cell_errors : Mapping[str, jax.Array]
        Panel title to cellwise errors of shape ``(K, 1, 1, 1)``, e.g. from
        the ``jax_fem`` ``compute_*_error(..., reduce=False)`` of a finite
        element solution or ``compute_network_error(..., reduce=False)``.

    Returns
    -------
    figure : Figure
        One panel per entry, each cell colored by its error on a scale from
        zero to the largest error, with the axes tight around the domain; on
        a fracture network, one row per fracture, drawn in its plane, with a
        color scale shared by the fractures.
    """
    from matplotlib.tri import Triangulation

    rows = _cell_error_panels(mesh)
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(
        len(rows),
        len(cell_errors),
        figsize=(5.5 * len(cell_errors), 4.8 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    for column, (title, errors) in enumerate(cell_errors.items()):
        values = np.ravel(np.asarray(errors))
        for row, (prefix, coordinates, cells, cell_indices) in enumerate(rows):
            axis = axes[row, column]
            image = axis.tripcolor(
                Triangulation(coordinates[:, 0], coordinates[:, 1], cells),
                facecolors=values[cell_indices],
                shading="flat",
                edgecolors="k",
                linewidth=0.1,
                vmin=0.0,
                vmax=np.max(values),
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            row_points = coordinates[np.unique(cells)]
            axis.set_xlim(np.min(row_points[:, 0]), np.max(row_points[:, 0]))
            axis.set_ylim(np.min(row_points[:, 1]), np.max(row_points[:, 1]))
            axis.set_aspect("equal")
            axis.set_title(prefix + title)
            axis.set_xlabel("$x$" if not prefix else "$s_1$")
            axis.set_ylabel("$y$" if not prefix else "$s_2$")
            show_grid(axis)
    return figure
