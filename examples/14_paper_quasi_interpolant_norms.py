"""Operator norms of the quasi-interpolants ``Pi_psi`` and ``Pi_P``.

For the ``l^inf`` family with the normalised weights and for the criss-cross
family, whose weights sum to one, the quasi-interpolant
``Pi v = sum_(i in I_Omega) v_i psi_i``, with the weighted means
``v_i = (phi_i, v) / (phi_i, 1)``, has the operator norm
``sup |grad Pi v|_0 / |grad v|_0`` over ``H^1_0``. Its discrete value
``||Pi||_h`` is the supremum over the continuous piecewise linear functions
on the criss-cross mesh of size ``s / r`` that vanish on the boundary, the
square root of the largest eigenvalue of a generalized eigenvalue problem.
It bounds ``||Pi||`` from below and increases with ``r``; the table lists it
for two refinements and every ``n``, next to the a priori bound
``N_ov K_psi`` of the paper, with ``K_psi = ||grad psi||_inf s max C_P h / s``
from the computed constants of (W3).

Outputs, under ``experiments/14_paper_quasi_interpolant_norms_<time>/``:

* ``norms.csv``: every ``n`` and refinement, with the computing time;
* ``norms.tex``: the table, a LaTeX ``tabular``.

Run with:

    uv run python examples/14_paper_quasi_interpolant_norms.py
"""

import csv
import math
import time

from jax_fem.diagnostics import create_timestamped_output_directory
from jax_fem.problem import create_poisson_sin_sin_problem

from vpinns_jax.analysis import compute_quasi_interpolant_norm
from vpinns_jax.test_spaces import (
    LINF_CONFIGURATIONS,
    compute_linf_poincare_constants,
    create_crisscross_family,
    create_crisscross_unit_square_mesh,
    create_linf_vertex_family,
)
from vpinns_jax.test_spaces.poincare_constants import round_up

# Overlap N_ov and ||grad psi_c||_inf s of the l^inf family (Lemma 5.1).
LINF_OVERLAP = 4
LINF_GRADIENT_BOUND = math.sqrt(5.0)
OPERATORS = {
    "psi": (create_linf_vertex_family, True),
    "P": (create_crisscross_family, False),
}


def main(
    grid_sizes: tuple[int, ...] = (2, 4, 8, 16, 24, 32),
    refinements: tuple[int, ...] = (4, 8),
    quadrature_degree: int = 8,
) -> None:
    """
    Compute the norms of both quasi-interpolants and write the table.

    Parameters
    ----------
    grid_sizes : tuple[int, ...]
        Numbers of grid cells per side ``n``.
    refinements : tuple[int, ...]
        Cells of the fine mesh per grid cell, ``r``.
    quadrature_degree : int
        Exactness degree of the rule of the Gram matrices of the test
        functions on the local triangles.
    """
    output_directory = create_timestamped_output_directory(
        "experiments", "14_paper_quasi_interpolant_norms"
    )
    # Only the coefficients of the energy norm are used.
    problem = create_poisson_sin_sin_problem(create_crisscross_unit_square_mesh(2))
    poincare_bound = max(
        round_up(compute_linf_poincare_constants(name).unweighted, 3)
        for name in LINF_CONFIGURATIONS
    )
    a_priori = LINF_OVERLAP * LINF_GRADIENT_BOUND * poincare_bound
    print(f"A priori bound N_ov K_psi = 4 sqrt(5) {poincare_bound} = {a_priori:.3f}")

    rows = []
    norms = {}
    for n in grid_sizes:
        for operator, (create_family, normalise) in OPERATORS.items():
            family = create_family(n)
            for refinement in refinements:
                start = time.perf_counter()
                norm = compute_quasi_interpolant_norm(
                    family,
                    problem,
                    n,
                    normalise=normalise,
                    refinement=refinement,
                    quadrature_degree=quadrature_degree,
                )
                elapsed = time.perf_counter() - start
                norms[operator, n, refinement] = norm
                rows.append(
                    {
                        "operator": operator,
                        "cells_per_side": n,
                        "refinement": refinement,
                        "norm": norm,
                        "time_s": elapsed,
                    }
                )
                print(
                    f"n = {n:3d}, Pi_{operator:3s} r = {refinement}: "
                    f"||Pi||_h = {norm:.4f} ({elapsed:.1f}s)",
                    flush=True,
                )
        with open(output_directory / "norms.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    columns = [
        (operator, refinement) for operator in OPERATORS for refinement in refinements
    ]
    symbols = {"psi": r"\Pi_\psi", "P": r"\Pi_\P"}
    lines = [
        r"\begin{tabular}{|r|"
        + "c" * len(refinements)
        + "|"
        + "c" * len(refinements)
        + "|}",
        r"\hline",
        r"\(n\) & "
        + " & ".join(
            rf"\(\norm{{{symbols[operator]}}}_h\), \(s/{refinement}\)"
            for operator, refinement in columns
        )
        + r"\\",
        r"\hline",
    ]
    for n in grid_sizes:
        lines.append(
            f"{n} & "
            + " & ".join(f"{norms[operator, n, r]:.4f}" for operator, r in columns)
            + r"\\"
        )
    a_priori_row = [f"{a_priori:.2f}"] * len(refinements) + ["--"] * len(refinements)
    lines += [
        r"\hline",
        r"\(N_{ov}K\) & " + " & ".join(a_priori_row) + r"\\",
        r"\hline",
        r"\end{tabular}",
    ]
    (output_directory / "norms.tex").write_text("\n".join(lines) + "\n")
    print(f"Saved norms.csv and norms.tex to {output_directory}/")


if __name__ == "__main__":
    main()
