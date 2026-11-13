"""Quadrature error of the normalised weights of the ``l^inf`` family.

The losses are integrated on the triangles of the criss-cross triangulation
``T_s``. On each of them the raw weights ``phi_i`` and the covering ``S`` are
affine, so the normalised weights ``psi_i = phi_i / S`` are quotients of
affine functions: no rule integrates them exactly, whereas a rule of degree
``q`` integrates the products of the raw weights and their gradients
exactly. For every tabulated degree ``q``, this script measures the relative
errors of

* the energy Gram matrices ``G_psi`` and ``G_phi`` of the interior test
  functions, which do not depend on the problem;
* ``eta_psi``, ``sqrt(L_MF)`` and ``rho`` for three trial functions:

  - the fixed error ``e = sin(pi x) sin(pi y)``: for ``w = u - e`` the
    residual ``R(w) = a(e, .)`` and the strong residual ``f + Lap w = -Lap e``
    do not depend on the data;
  - the initial network ``u_theta0 = zeta N_theta0``, for
    ``u = sin(pi x) sin(pi y)``;
  - the network trained with ``L_MF + rho^2`` with the settings of
    Experiment 3, at ``n = train_cells_per_side`` only.

The reference values use the degree-8 rule on the local triangles split into
``4^k`` triangles, a composite rule on ``T_s``; the relative difference to the
reference with ``k - 1`` splits is reported as its accuracy. For the
networks, ``eta_psi / ||grad(u - u_theta)||_0`` is reported for every ``q``
as well: the lower bound ``eta_psi <= ||grad(u - u_theta)||_0`` holds
exactly, so a ratio above one reveals the quadrature error. The residual of
a trained network is a small difference of integrals of size one, so the
relative error of ``eta_psi`` grows as the network improves.

Outputs, under ``experiments/13_paper_quadrature_error_<time>/``:

* ``quadrature.csv``: every case, ``n``, rule and quantity;
* ``gram.tex``: relative errors of the Gram matrices, a LaTeX ``tabular``;
* ``estimators_<case>.tex``: relative errors of the estimators of each
  trial function, LaTeX ``tabular`` environments.

Run with:

    uv run python examples/13_paper_quadrature_error.py
"""

import csv
import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax_fem.diagnostics import (
    create_timestamped_output_directory,
    energy_error_functional,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.problem import EllipticProblem, create_poisson_sin_sin_problem
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

from vpinns_jax.analysis import compute_network_error
from vpinns_jax.losses import (
    compute_meshfree_patch_residual,
    compute_meshfree_patch_residual_terms,
    mf_vpinn_loss,
    restrict_to_free,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
    evaluate_network,
    evaluate_network_flux_divergence,
)
from vpinns_jax.solvers import train_adam
from vpinns_jax.test_spaces import (
    MeshfreePatchFamily,
    MeshfreePatchTestSpace,
    create_crisscross_unit_square_mesh,
    create_linf_vertex_family,
    create_meshfree_patch_test_space,
    subdivide_meshfree_family,
)

# Estimators whose quadrature error is measured, with their LaTeX names.
ESTIMATORS = {
    "eta_psi": r"\eta_\psi",
    "sqrt_L_MF": r"\L_{\mathrm{MF}}^{1/2}",
    "rho": r"\rho",
}
GRAMS = {"gram_psi": r"\bm{G}_\psi", "gram_phi": r"\bm{G}_\varphi"}
CASES = {
    "fixed_error": r"fixed error \(e=\sin(\pi x)\sin(\pi y)\)",
    "initial_network": r"initial network",
    "trained_network": r"trained network",
}


class Rule(NamedTuple):
    """
    A quadrature rule on the local triangles.

    Attributes
    ----------
    label : str
        Name of the rule in the tables.
    degree : int
        Exactness degree of the rule on each triangle.
    splits : int
        Number of uniform splits of the local triangles.
    """

    label: str
    degree: int
    splits: int


class Spaces(NamedTuple):
    """
    Raw and normalised test spaces of the ``l^inf`` family with one rule.

    Attributes
    ----------
    raw : MeshfreePatchTestSpace
        Test functions ``phi_i``.
    normalised : MeshfreePatchTestSpace
        Test functions ``psi_i``, at the points of ``raw``.
    gram_phi : np.ndarray
        Interior energy Gram matrix of ``raw``.
    gram_psi : np.ndarray
        Interior energy Gram matrix of ``normalised``.
    """

    raw: MeshfreePatchTestSpace
    normalised: MeshfreePatchTestSpace
    gram_phi: np.ndarray
    gram_psi: np.ndarray


def create_spaces(
    family: MeshfreePatchFamily, problem: EllipticProblem, rule: Rule
) -> Spaces:
    """
    Raw and normalised spaces of a family integrated with a rule.

    Parameters
    ----------
    family : MeshfreePatchFamily
        The ``l^inf`` family.
    problem : EllipticProblem
        Problem providing the coefficients.
    rule : Rule
        The rule.

    Returns
    -------
    spaces : Spaces
        The spaces and their interior Gram matrices.
    """
    family = subdivide_meshfree_family(family, rule.splits)
    spaces = [
        create_meshfree_patch_test_space(
            family, problem, quadrature_degree=rule.degree, normalise=normalise
        )
        for normalise in (False, True)
    ]
    grams = [
        np.asarray(restrict_to_free(space.gram, family.interior_patches))
        for space in spaces
    ]
    return Spaces(
        raw=spaces[0],
        normalised=spaces[1],
        gram_phi=0.5 * (grams[0] + grams[0].T),
        gram_psi=0.5 * (grams[1] + grams[1].T),
    )


def without_source(space: MeshfreePatchTestSpace) -> MeshfreePatchTestSpace:
    """
    The same test space for the problem with zero source.

    Parameters
    ----------
    space : MeshfreePatchTestSpace
        Test space.

    Returns
    -------
    space : MeshfreePatchTestSpace
        Its copy with ``f = 0``, so the residual of a trial ``e`` is
        ``a(e, psi_i)`` and its strong residual ``Lap e``.
    """
    coefficients = space.coefficients
    return dataclasses.replace(
        space,
        coefficients=coefficients._replace(source=jnp.zeros_like(coefficients.source)),
    )


def dual_norm(residual: np.ndarray, gram: np.ndarray) -> float:
    """
    Dual norm of a residual vector with a pseudo-inverse.

    Parameters
    ----------
    residual : np.ndarray
        Residual vector, shape ``(N,)``.
    gram : np.ndarray
        Symmetric Gram matrix, shape ``(N, N)``; the pseudo-inverse keeps the
        low-degree rules, whose matrices may be singular, usable.

    Returns
    -------
    norm : float
        ``(r^T G^+ r)^(1/2)``.
    """
    return float(
        np.sqrt(max(residual @ np.linalg.pinv(gram, hermitian=True) @ residual, 0.0))
    )


def estimators(
    spaces: Spaces,
    trial_function: Callable[[jax.Array], jax.Array],
    problem: EllipticProblem,
    zero_source: bool,
) -> dict[str, float]:
    """
    ``eta_psi``, ``sqrt(L_MF)`` and ``rho`` of a trial function.

    Parameters
    ----------
    spaces : Spaces
        Test spaces of the rule.
    trial_function : Callable[[jax.Array], jax.Array]
        Trial function of one point.
    problem : EllipticProblem
        Problem providing the diffusion.
    zero_source : bool
        Use ``f = 0``, for the fixed error.

    Returns
    -------
    estimators : dict[str, float]
        The three estimators.
    """
    raw, normalised = spaces.raw, spaces.normalised
    if zero_source:
        raw, normalised = without_source(raw), without_source(normalised)
    interior = raw.family.interior_patches
    trial = evaluate_network(trial_function, raw.points)
    raw_residual = jnp.take(compute_meshfree_patch_residual(raw, trial), interior, -2)
    normalised_residual = np.asarray(
        jnp.take(compute_meshfree_patch_residual(normalised, trial), interior, -2)
    )[:, 0]
    flux_divergence = evaluate_network_flux_divergence(
        trial_function, problem.diffusion, raw.points
    )
    rho_squared = jnp.sum(
        compute_meshfree_patch_residual_terms(raw, trial, flux_divergence)
    )
    areas = jnp.take(raw.family.patch_areas, interior)
    return {
        "eta_psi": dual_norm(normalised_residual, spaces.gram_psi),
        "sqrt_L_MF": float(jnp.sqrt(mf_vpinn_loss(raw_residual, areas))),
        "rho": float(jnp.sqrt(rho_squared)),
    }


def train_network(
    cells_per_side: int,
    problem: EllipticProblem,
    network: eqx.Module,
    *,
    num_steps: int,
    quadrature_degree: int,
) -> eqx.Module:
    """
    Train a network with ``L_MF + rho^2`` on the ``l^inf`` family.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side ``n``.
    problem : EllipticProblem
        Problem of the training.
    network : eqx.Module
        Initial free network.
    num_steps : int
        Number of Adam steps, with the learning rate decaying from ``1e-3`` to
        ``1e-4``.
    quadrature_degree : int
        Exactness degree of the training rule on the local triangles.

    Returns
    -------
    network : eqx.Module
        Trained free network.
    """
    family = create_linf_vertex_family(cells_per_side)
    space = create_meshfree_patch_test_space(
        family, problem, quadrature_degree=quadrature_degree
    )
    interior = family.interior_patches
    areas = jnp.take(family.patch_areas, interior)
    distance = ExactDistanceFunction("unit_square")

    def loss_fn(model: eqx.Module) -> jax.Array:
        """
        ``L_MF + rho^2`` of the constrained network.

        Parameters
        ----------
        model : eqx.Module
            Free network.

        Returns
        -------
        loss : jax.Array
            Loss, shape ``()``.
        """
        trial_function = ConstrainedAnsatz(model, distance)
        trial = evaluate_network(trial_function, space.points)
        residual = jnp.take(compute_meshfree_patch_residual(space, trial), interior, -2)
        flux_divergence = evaluate_network_flux_divergence(
            trial_function, problem.diffusion, space.points
        )
        rho_squared = jnp.sum(
            compute_meshfree_patch_residual_terms(space, trial, flux_divergence)
        )
        return mf_vpinn_loss(residual, areas) + rho_squared

    trained, _ = train_adam(
        loss_fn,
        network,
        num_steps=num_steps,
        learning_rate=1e-3,
        final_learning_rate=1e-4,
    )
    return trained


def scientific(value: float) -> str:
    """
    A number in LaTeX scientific notation with two significant digits.

    Parameters
    ----------
    value : float
        Number.

    Returns
    -------
    text : str
        ``0``, or ``m\\cdot10^{e}`` in math mode.
    """
    if value == 0.0 or not math.isfinite(value):
        return "0" if value == 0.0 else "--"
    exponent = math.floor(math.log10(abs(value)))
    return f"\\({value / 10**exponent:.1f}\\cdot10^{{{exponent}}}\\)"


def plain(label: str) -> str:
    """
    A table label without its LaTeX math delimiters.

    Parameters
    ----------
    label : str
        Label, possibly with ``\\(`` and ``\\)``.

    Returns
    -------
    text : str
        The label for the console.
    """
    return label.replace("\\(", "").replace("\\)", "")


def relative_error(value: float, reference: float) -> float:
    """
    ``|value - reference| / |reference|``.

    Parameters
    ----------
    value : float
        Approximation.
    reference : float
        Reference value, nonzero.

    Returns
    -------
    error : float
        The relative error.
    """
    return abs(value - reference) / abs(reference)


def write_table(
    path: Path,
    labels: Sequence[str],
    grid_sizes: Sequence[int],
    columns: Mapping[str, str],
    cell: Callable[[str, int, str], str],
) -> None:
    """
    LaTeX table with one row per rule and a column group per ``n``.

    Parameters
    ----------
    path : Path
        Output file.
    labels : Sequence[str]
        Row labels, the rules and the reference check.
    grid_sizes : Sequence[int]
        Values of ``n``, one column group each.
    columns : Mapping[str, str]
        Quantities of each group by key, with their LaTeX headers.
    cell : Callable[[str, int, str], str]
        Text of the cell of a row label, ``n`` and quantity key.
    """
    width = len(columns)
    lines = [
        r"\begin{tabular}{|l|" + ("c" * width + "|") * len(grid_sizes) + "}",
        r"\hline",
        " & "
        + " & ".join(rf"\multicolumn{{{width}}}{{c|}}{{\(n={n}\)}}" for n in grid_sizes)
        + r"\\",
        "rule & " + " & ".join(list(columns.values()) * len(grid_sizes)) + r"\\",
        r"\hline",
    ]
    for label in labels:
        if label == labels[-1]:
            lines.append(r"\hline")
        lines.append(
            f"{label} & "
            + " & ".join(cell(label, n, key) for n in grid_sizes for key in columns)
            + r"\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(
    grid_sizes: tuple[int, ...] = (4, 8, 16),
    degrees: tuple[int, ...] = (1, 2, 4, 5, 6, 8),
    reference_degree: int = 8,
    reference_splits: int = 2,
    width: int = 32,
    depth: int = 3,
    seed: int = 0,
    train_cells_per_side: int = 8,
    train_steps: int = 6000,
    train_quadrature_degree: int = 4,
    error_cells_per_side: int = 32,
    error_quadrature_degree: int = 6,
) -> None:
    """
    Measure the quadrature errors and write the tables.

    Parameters
    ----------
    grid_sizes : tuple[int, ...]
        Numbers of grid cells per side ``n``.
    degrees : tuple[int, ...]
        Exactness degrees ``q`` of the rules on the local triangles.
    reference_degree : int
        Exactness degree of the reference rule.
    reference_splits : int
        Number of uniform splits ``k`` of the local triangles of the
        reference, at least one.
    width : int
        Hidden-layer width of the network.
    depth : int
        Number of hidden layers.
    seed : int
        Seed of the initial network weights.
    train_cells_per_side : int
        ``n`` of the trained network, one of ``grid_sizes``; ``0`` skips it.
    train_steps : int
        Number of Adam steps of the training.
    train_quadrature_degree : int
        Exactness degree of the training rule.
    error_cells_per_side : int
        Cells per side of the criss-cross mesh of the energy errors.
    error_quadrature_degree : int
        Exactness degree of the rule on that mesh.
    """
    if reference_splits < 1:
        raise ValueError("reference_splits must be at least one.")
    output_directory = create_timestamped_output_directory(
        "experiments", "13_paper_quadrature_error"
    )
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    error_mesh = create_crisscross_unit_square_mesh(error_cells_per_side)
    error_basis = create_cell_basis(
        create_finite_element_space(error_mesh, element),
        element.reference_cell.create_quadrature(error_quadrature_degree),
    )
    problem = create_poisson_sin_sin_problem(error_mesh)
    distance = ExactDistanceFunction("unit_square")
    initial = MultiLayerPerceptron(width=width, depth=depth, key=jr.PRNGKey(seed))
    trial_functions = {
        "fixed_error": (
            lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
            True,
            {n: math.pi / math.sqrt(2.0) for n in grid_sizes},
        ),
        "initial_network": (ConstrainedAnsatz(initial, distance), False, {}),
    }
    if train_cells_per_side:
        start = time.perf_counter()
        trained = train_network(
            train_cells_per_side,
            problem,
            initial,
            num_steps=train_steps,
            quadrature_degree=train_quadrature_degree,
        )
        print(
            f"Trained with L_MF + rho^2 at n = {train_cells_per_side}, "
            f"{train_steps} steps ({time.perf_counter() - start:.1f}s)."
        )
        trial_functions["trained_network"] = (
            ConstrainedAnsatz(trained, distance),
            False,
            {},
        )
    # Energy errors of the networks, the same for every n.
    for name, (trial_function, zero_source, errors) in trial_functions.items():
        if not zero_source:
            error = float(
                compute_network_error(
                    trial_function, error_basis, problem, energy_error_functional
                )
            )
            errors.update({n: error for n in grid_sizes})
            print(f"{name}: ||grad(u - u_theta)||_0 = {error:.3e}")

    rules = [Rule(f"\\(q={q}\\)", q, 0) for q in degrees]
    references = [
        Rule("reference", reference_degree, reference_splits),
        Rule("reference check", reference_degree, reference_splits - 1),
    ]
    rows = []
    values = {}
    for n in grid_sizes:
        family = create_linf_vertex_family(n)
        for rule in rules + references:
            start = time.perf_counter()
            spaces = create_spaces(family, problem, rule)
            values[n, rule.label, "gram_psi"] = spaces.gram_psi
            values[n, rule.label, "gram_phi"] = spaces.gram_phi
            for name, (trial_function, zero_source, _) in trial_functions.items():
                if name == "trained_network" and n != train_cells_per_side:
                    continue
                for key, value in estimators(
                    spaces, trial_function, problem, zero_source
                ).items():
                    values[n, rule.label, name, key] = value
            print(
                f"n = {n:3d}, {plain(rule.label):16s}: "
                f"{spaces.raw.points.shape[-3] * family.number_of_patches} points "
                f"({time.perf_counter() - start:.1f}s)",
                flush=True,
            )

    reference = references[0].label
    labels = [rule.label for rule in rules] + [references[1].label]

    def gram_error(label: str, n: int, key: str) -> float:
        """
        Relative Frobenius error of a Gram matrix.

        Parameters
        ----------
        label : str
            Rule.
        n : int
            Number of grid cells per side.
        key : str
            ``"gram_psi"`` or ``"gram_phi"``.

        Returns
        -------
        error : float
            ``|G - G_ref|_F / |G_ref|_F``.
        """
        approximation = values[n, label, key]
        exact = values[n, reference, key]
        return float(np.linalg.norm(approximation - exact) / np.linalg.norm(exact))

    for n in grid_sizes:
        for label in labels:
            for key in GRAMS:
                rows.append(
                    {
                        "case": "gram",
                        "cells_per_side": n,
                        "rule": label,
                        "quantity": key,
                        "value": "",
                        "reference": "",
                        "relative_error": gram_error(label, n, key),
                    }
                )
            for name, (_, _, errors) in trial_functions.items():
                if (n, label, name, "eta_psi") not in values:
                    continue
                for key in ESTIMATORS:
                    rows.append(
                        {
                            "case": name,
                            "cells_per_side": n,
                            "rule": label,
                            "quantity": key,
                            "value": values[n, label, name, key],
                            "reference": values[n, reference, name, key],
                            "relative_error": relative_error(
                                values[n, label, name, key],
                                values[n, reference, name, key],
                            ),
                        }
                    )
                rows.append(
                    {
                        "case": name,
                        "cells_per_side": n,
                        "rule": label,
                        "quantity": "eta_psi_over_error",
                        "value": values[n, label, name, "eta_psi"] / errors[n],
                        "reference": values[n, reference, name, "eta_psi"] / errors[n],
                        "relative_error": "",
                    }
                )
    with open(output_directory / "quadrature.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_table(
        output_directory / "gram.tex",
        labels,
        grid_sizes,
        {key: f"\\({latex}\\)" for key, latex in GRAMS.items()},
        lambda label, n, key: scientific(gram_error(label, n, key)),
    )
    for name, (_, _, errors) in trial_functions.items():
        sizes = [n for n in grid_sizes if (n, reference, name, "eta_psi") in values]
        columns = {key: f"\\({latex}\\)" for key, latex in ESTIMATORS.items()}
        columns["ratio"] = r"\(\eta_\psi/\norm{\nabla(u-w)}_0\)"

        def cell(label: str, n: int, key: str, name=name, errors=errors) -> str:
            """
            Text of a cell of the estimator table.

            Parameters
            ----------
            label : str
                Rule of the row.
            n : int
                Number of grid cells per side.
            key : str
                Estimator, or ``"ratio"`` for ``eta_psi / error``.
            name : str
                Trial function.
            errors : dict[int, float]
                Its energy error by ``n``.

            Returns
            -------
            text : str
                The relative error in LaTeX, or the ratio.
            """
            if key == "ratio":
                return f"{values[n, label, name, 'eta_psi'] / errors[n]:.3f}"
            return scientific(
                relative_error(
                    values[n, label, name, key], values[n, reference, name, key]
                )
            )

        write_table(
            output_directory / f"estimators_{name}.tex", labels, sizes, columns, cell
        )
        print(f"\n{plain(CASES[name])} (relative errors; last column eta_psi / error):")
        for n in sizes:
            for label in labels:
                relative_errors = [
                    relative_error(
                        values[n, label, name, key], values[n, reference, name, key]
                    )
                    for key in ESTIMATORS
                ]
                print(
                    f"  n = {n:3d} {plain(label):16s} "
                    + " ".join(f"{error:.2e}" for error in relative_errors)
                    + f"  {values[n, label, name, 'eta_psi'] / errors[n]:.3f}"
                )
    print(f"Saved quadrature.csv, gram.tex and estimators_*.tex to {output_directory}/")


if __name__ == "__main__":
    main()
