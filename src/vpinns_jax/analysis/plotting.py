"""
Plots of training histories, solutions, and cellwise errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.element import LagrangeTriangleP1
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import TriangleMesh
from jax_fem.problem import EllipticProblem

from vpinns_jax.analysis.monitoring import TrainingRecord
from vpinns_jax.models.evaluation import Network, evaluate_network_values
from vpinns_jax.solvers.trainer import TrainingHistory

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation

_SURFACE = "#fcfcfb"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
# Categorical slots in their fixed order: blue, orange, aqua, yellow, magenta,
# green, violet, red.
_SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
# One-hue ramps, light to dark: blue for solution values, orange for errors.
_VALUE_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
_ERROR_RAMP = ("#fce3d8", "#f6b294", "#eb6834", "#c4501f", "#8f3612", "#5c210a")

_STYLE = {
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "axes.edgecolor": _BASELINE,
    "axes.labelcolor": _SECONDARY_INK,
    "axes.titlecolor": _PRIMARY_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": _GRIDLINE,
    "grid.linewidth": 0.6,
    "xtick.color": _MUTED_INK,
    "ytick.color": _MUTED_INK,
    "xtick.labelcolor": _SECONDARY_INK,
    "ytick.labelcolor": _SECONDARY_INK,
    "text.color": _PRIMARY_INK,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "legend.labelcolor": _SECONDARY_INK,
}


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


def _colormap(name: str, ramp: tuple[str, ...]) -> Any:
    """
    Continuous colormap through a one-hue ramp.

    Parameters
    ----------
    name : str
        Name of the colormap.
    ramp : tuple[str, ...]
        Colors from light to dark.

    Returns
    -------
    colormap : matplotlib.colors.LinearSegmentedColormap
        The colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, ramp)


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


def _label_line_end(axis: Axes, x: np.ndarray, y: np.ndarray, text: str) -> None:
    """
    Label a line next to its last point.

    Parameters
    ----------
    axis : Axes
        Axes holding the line.
    x : np.ndarray
        Abscissae of the line.
    y : np.ndarray
        Ordinates of the line.
    text : str
        Label, drawn in secondary ink.
    """
    axis.annotate(
        text,
        (x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=_SECONDARY_INK,
        fontsize="small",
    )


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
        Two panels sharing the step axis: the loss ``L`` at every step, and
        the relative loss ``sqrt(L) / |u|_A`` with the relative energy error
        ``|u - w|_A / |u|_A`` at the monitored steps.

    Notes
    -----
    The loss and the relative quantities have different scales, so they are
    drawn on separate axes; the two relative quantities share theirs, and
    their ratio is the robustness of ``plot_robustness``.
    """
    record = _training_record(history)
    steps = np.asarray(history.monitored_steps)
    relative_loss = np.asarray(record.relative_loss)
    relative_error = np.asarray(record.relative_energy_error)
    pyplot = _pyplot()
    with pyplot.rc_context(_STYLE):
        figure, (loss_axis, relative_axis) = pyplot.subplots(
            2, 1, figsize=(7.5, 6.0), sharex=True, constrained_layout=True
        )
        loss_axis.semilogy(np.asarray(history.loss), color=_SERIES[0], linewidth=1.0)
        loss_axis.set_ylabel("loss $L$")
        loss_axis.set_title("Loss")

        relative_axis.semilogy(
            steps, relative_loss, color=_SERIES[0], label=r"$\sqrt{L} / \|u\|_A$"
        )
        relative_axis.semilogy(
            steps,
            relative_error,
            color=_SERIES[1],
            label=r"$\|u - w\|_A / \|u\|_A$",
        )
        _label_line_end(relative_axis, steps, relative_loss, "relative loss")
        _label_line_end(relative_axis, steps, relative_error, "relative error")
        relative_axis.set_xlabel("step")
        relative_axis.set_ylabel("relative to $\\|u\\|_A$")
        relative_axis.set_title("Relative loss and relative energy error")
        relative_axis.legend(loc="upper right")
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
        The loss and each of its terms at the monitored steps.
    """
    record = _training_record(history)
    steps = np.asarray(history.monitored_steps)
    pyplot = _pyplot()
    with pyplot.rc_context(_STYLE):
        figure, axis = pyplot.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        axis.semilogy(
            steps,
            np.asarray(record.loss),
            color=_SECONDARY_INK,
            linestyle="--",
            label="loss $L$",
        )
        for color, (name, values) in zip(_SERIES, record.terms.items(), strict=False):
            axis.semilogy(steps, np.asarray(values), color=color, label=name)
        axis.set_xlabel("step")
        axis.set_ylabel("value")
        axis.set_title("Loss terms")
        axis.legend(loc="upper right")
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
        at one.

    Notes
    -----
    A loss that is a robust estimator of the energy error keeps this ratio
    bounded above and below during training. A ratio drifting to zero means
    the loss decreases while the error does not.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    record = _training_record(history)
    pyplot = _pyplot()
    with pyplot.rc_context(_STYLE):
        figure, axis = pyplot.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        axis.axhline(1.0, color=_MUTED_INK, linestyle=":", linewidth=1.0)
        axis.semilogy(
            np.asarray(history.monitored_steps),
            np.asarray(record.robustness),
            color=_SERIES[0],
        )
        # Ratios usually span less than a decade, so label 1, 2, and 5 in every
        # decade, as plain numbers.
        axis.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("step")
        axis.set_ylabel(r"$\sqrt{L} / \|u - w\|_A$")
        axis.set_title("Robustness: relative loss / relative energy error")
    return figure


def _triangulation(mesh: TriangleMesh) -> Triangulation:
    """
    Matplotlib triangulation of a planar mesh.

    Parameters
    ----------
    mesh : TriangleMesh
        Planar triangular mesh.

    Returns
    -------
    triangulation : Triangulation
        Triangulation of the mesh vertices.
    """
    from matplotlib.tri import Triangulation

    coordinates = np.asarray(mesh.vertex_coordinates)[..., 0, :]
    return Triangulation(
        coordinates[:, 0], coordinates[:, 1], np.asarray(mesh.cells_to_vertices)
    )


def _points(triangulation: Triangulation) -> jax.Array:
    """
    Points of a triangulation as row vectors.

    Parameters
    ----------
    triangulation : Triangulation
        A triangulation.

    Returns
    -------
    points : jax.Array
        Its points, shape ``(N_points, 1, 2)``.
    """
    return jnp.expand_dims(
        jnp.stack((triangulation.x, triangulation.y), axis=-1), axis=-2
    )


def _style_field_axis(axis: Axes, title: str) -> None:
    """
    Format an axis showing a field over the domain.

    Parameters
    ----------
    axis : Axes
        Axes of a field plot.
    title : str
        Title of the panel.
    """
    axis.grid(False)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")


def _plot_solution_panels(
    triangulation: Triangulation, exact: np.ndarray, approximation: np.ndarray
) -> Figure:
    """
    Exact solution, approximation, and pointwise absolute error side by side.

    Parameters
    ----------
    triangulation : Triangulation
        Triangulation of the evaluation points.
    exact : np.ndarray
        Exact solution at the points, shape ``(N_points,)``.
    approximation : np.ndarray
        Approximation at the points, shape ``(N_points,)``.

    Returns
    -------
    figure : Figure
        Three panels; the two solutions share a color scale.
    """
    low = min(float(np.min(exact)), float(np.min(approximation)))
    high = max(float(np.max(exact)), float(np.max(approximation)))
    value_colormap = _colormap("solution", _VALUE_RAMP)
    pyplot = _pyplot()
    with pyplot.rc_context(_STYLE):
        figure, axes = pyplot.subplots(
            1, 3, figsize=(15.0, 4.6), constrained_layout=True
        )
        for axis, values, title in zip(
            axes[:2],
            (exact, approximation),
            ("Exact solution", "VPINN solution"),
            strict=True,
        ):
            image = axis.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap=value_colormap,
                vmin=low,
                vmax=high,
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            _style_field_axis(axis, title)
        image = axes[2].tripcolor(
            triangulation,
            np.abs(exact - approximation),
            shading="gouraud",
            cmap=_colormap("error", _ERROR_RAMP),
        )
        figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
        _style_field_axis(axes[2], "Pointwise absolute error")
    return figure


def _exact_values(problem: EllipticProblem, points: jax.Array) -> np.ndarray:
    """
    Exact solution at points.

    Parameters
    ----------
    problem : EllipticProblem
        Problem with an exact solution.
    points : jax.Array
        Points, shape ``(N_points, 1, d)``.

    Returns
    -------
    values : np.ndarray
        Exact solution, shape ``(N_points,)``.

    Raises
    ------
    ValueError
        If the problem has no exact solution.
    """
    if problem.exact_solution is None:
        raise ValueError("problem.exact_solution is required for the plot.")
    return np.asarray(jnp.squeeze(problem.exact_solution(points), axis=(-2, -1)))


def plot_network_solution(
    network: Network,
    mesh: TriangleMesh,
    problem: EllipticProblem,
    *,
    subdivisions: int = 3,
) -> Figure:
    """
    Exact solution, network solution, and pointwise absolute error.

    Parameters
    ----------
    network : Network
        Trial function of one point, e.g. a ``ConstrainedAnsatz``.
    mesh : TriangleMesh
        Planar mesh of the domain.
    problem : EllipticProblem
        Problem providing the exact solution.
    subdivisions : int
        Number of uniform refinements of the mesh for the evaluation points.

    Returns
    -------
    figure : Figure
        Three panels over the domain.

    Notes
    -----
    Both functions are evaluated exactly at the vertices of the refined
    triangulation and shaded linearly between them.
    """
    from matplotlib.tri import UniformTriRefiner

    refined = UniformTriRefiner(_triangulation(mesh)).refine_triangulation(
        subdiv=subdivisions
    )
    points = _points(refined)
    approximation = jnp.squeeze(evaluate_network_values(network, points), axis=(-2, -1))
    return _plot_solution_panels(
        refined, _exact_values(problem, points), np.asarray(approximation)
    )


def plot_fe_solution(
    solution: FiniteElementFunction,
    problem: EllipticProblem,
    *,
    subdivisions: int = 3,
) -> Figure:
    """
    Exact solution, P1 solution, and pointwise absolute error.

    Parameters
    ----------
    solution : FiniteElementFunction
        P1 function on a planar mesh, e.g. from ``interpolate_network``.
    problem : EllipticProblem
        Problem providing the exact solution.
    subdivisions : int
        Number of uniform refinements of the mesh for the evaluation points.

    Returns
    -------
    figure : Figure
        Three panels over the domain.

    Raises
    ------
    NotImplementedError
        If the space is not P1.

    Notes
    -----
    The P1 function is interpolated linearly to the refined points, which is
    exact, so the error panel shows the true pointwise error there.
    """
    from matplotlib.tri import LinearTriInterpolator, UniformTriRefiner

    space = solution.space
    if not isinstance(space.element, LagrangeTriangleP1):
        raise NotImplementedError("plot_fe_solution supports P1 functions only.")
    mesh = space.mesh
    vertex_values = np.zeros(mesh.vertex_coordinates.shape[-3])
    vertex_values[np.asarray(mesh.cells_to_vertices)] = np.asarray(solution.dof_values)[
        np.asarray(space.cells_to_dofs), 0
    ]
    triangulation = _triangulation(mesh)
    refined, approximation = UniformTriRefiner(triangulation).refine_field(
        vertex_values,
        triinterpolator=LinearTriInterpolator(triangulation, vertex_values),
        subdiv=subdivisions,
    )
    return _plot_solution_panels(
        refined, _exact_values(problem, _points(refined)), np.asarray(approximation)
    )


def plot_cell_errors(
    mesh: TriangleMesh, cell_errors: Mapping[str, jax.Array]
) -> Figure:
    """
    Cellwise error norms over the mesh.

    Parameters
    ----------
    mesh : TriangleMesh
        Planar mesh whose cells carry the errors.
    cell_errors : Mapping[str, jax.Array]
        Panel title to cellwise errors of shape ``(K, 1, 1, 1)``, e.g. from
        ``compute_network_error(..., reduce=False)`` or the ``jax_fem``
        ``compute_*_error(..., reduce=False)``.

    Returns
    -------
    figure : Figure
        One panel per entry, each cell colored by its error.
    """
    triangulation = _triangulation(mesh)
    error_colormap = _colormap("error", _ERROR_RAMP)
    pyplot = _pyplot()
    with pyplot.rc_context(_STYLE):
        figure, axes = pyplot.subplots(
            1,
            len(cell_errors),
            figsize=(5.2 * len(cell_errors), 4.6),
            constrained_layout=True,
            squeeze=False,
        )
        for axis, (title, errors) in zip(axes[0], cell_errors.items(), strict=True):
            image = axis.tripcolor(
                triangulation,
                facecolors=np.ravel(np.asarray(errors)),
                shading="flat",
                cmap=error_colormap,
                edgecolors=_SURFACE,
                linewidth=0.3,
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            _style_field_axis(axis, title)
    return figure
