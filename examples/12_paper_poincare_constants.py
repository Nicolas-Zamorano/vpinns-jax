"""Constants of (W2) and (W3) of the ``l^inf`` family, with their convergence.

For the interior, side and corner configurations of the ``l^inf`` family, the
constant ``c_2 h`` of the weighted inequality (W2) and the constant ``C_P h``
of (W3) are computed as P1 eigenvalue problems on criss-cross meshes with
``r`` cells per grid spacing ``s``, for increasing ``r``. Both constants are
invariant under scaling, so they are computed once on the reference
configurations at ``s = 1`` and hold for every ``n``.

The meshes are nested and the P1 functions form a subspace, so the discrete
constants increase to the exact ones. For each refinement the table lists the
constants, the difference ``d_r`` to the previous refinement, and the observed
order ``log2(d_(r/2) / d_r)``; after the finest refinement it lists the
Richardson extrapolation with the observed order, the value rounded up in the
third decimal that the ``l^inf`` family uses, and, for ``C_P h``, the closed
forms ``2 s / pi``, ``2 s / pi`` and ``sqrt(2) s / pi``.

Outputs, under ``experiments/12_paper_poincare_constants_<time>/``:

* ``constants.csv``: every configuration and refinement;
* ``convergence.tex``: the convergence table, a LaTeX ``tabular``;
* ``constants.tex``: the table of the constants as in the paper
  (configuration, ``h / s``, ``||grad phi||_inf s``, ``c_2 h / s``,
  ``C_P h / s``), a LaTeX ``tabular``.

Run with:

    uv run python examples/12_paper_poincare_constants.py
"""

import csv
import math
import time
from pathlib import Path

from jax_fem.diagnostics import create_timestamped_output_directory

from vpinns_jax.test_spaces import LINF_CONFIGURATIONS, compute_linf_poincare_constants
from vpinns_jax.test_spaces.poincare_constants import round_up

# Closed forms of C_P h / s, and h / s and ||grad phi||_inf s in LaTeX.
CLOSED_FORMS = {
    "interior": (2.0 / math.pi, r"2/\pi"),
    "side": (2.0 / math.pi, r"2/\pi"),
    "corner": (math.sqrt(2.0) / math.pi, r"\sqrt2/\pi"),
}
DIAMETERS = {"interior": r"$2\sqrt2$", "side": r"$\sqrt5$", "corner": r"$\sqrt2$"}
DESCRIPTIONS = {
    "interior": r"interior, square of side $2s$",
    "side": r"side, $2s\times s$ rectangle",
    "corner": r"corner, square of side $s$",
}
CONSTANTS = {"weighted": "c_2 h / s", "unweighted": "C_P h / s"}


def extrapolate(values: list[float]) -> tuple[float, float]:
    """
    Observed order and Richardson extrapolation of the last three values.

    Parameters
    ----------
    values : list[float]
        Values on successive refinements by a factor two, at least three.

    Returns
    -------
    order : float
        ``log2(d_(r/2) / d_r)`` of the last two differences, ``nan`` if it is
        undefined.
    limit : float
        ``v_r + d_r / (2^order - 1)``, or ``v_r`` if the order is undefined.
    """
    previous = values[-2] - values[-3]
    last = values[-1] - values[-2]
    if previous <= 0.0 or last <= 0.0:
        return math.nan, values[-1]
    order = math.log2(previous / last)
    return order, values[-1] + last / (2.0**order - 1.0)


def write_convergence_table(
    path: Path,
    results: dict,
    summary: dict,
    refinements: tuple[int, ...],
    decimals: int,
) -> None:
    """
    LaTeX table of the constants on every refinement.

    Parameters
    ----------
    path : Path
        Output file.
    results : dict
        ``PoincareConstants`` by configuration and refinement.
    summary : dict
        Observed order, extrapolation, and rounded value by configuration and
        field.
    refinements : tuple[int, ...]
        Mesh cells per grid spacing.
    decimals : int
        Decimals of the rounded values.
    """
    configurations = list(LINF_CONFIGURATIONS)
    lines = [
        r"\begin{tabular}{|r|" + "cc|" * len(configurations) + "}",
        r"\hline",
        " & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c|}}{{{configuration}}}"
            for configuration in configurations
        )
        + r"\\",
        "$r$ & " + " & ".join([r"$c_2h/s$ & $C_Ph/s$"] * len(configurations)) + r"\\",
        r"\hline",
    ]
    for refinement in refinements:
        values = []
        for configuration in configurations:
            constants = results[configuration, refinement]
            values += [f"{constants.weighted:.6f}", f"{constants.unweighted:.6f}"]
        lines.append(f"{refinement} & " + " & ".join(values) + r"\\")
    lines.append(r"\hline")
    for label, index, digits in (
        ("observed order", 0, 2),
        ("extrapolated", 1, 6),
        ("rounded up", 2, decimals),
    ):
        values = []
        for configuration in configurations:
            for field in CONSTANTS:
                value = summary[configuration, field][index]
                values.append("--" if math.isnan(value) else f"{value:.{digits}f}")
        lines.append(f"{label} & " + " & ".join(values) + r"\\")
    closed = []
    for configuration in configurations:
        value, latex = CLOSED_FORMS[configuration]
        closed += ["--", f"${latex}={value:.6f}$"]
    lines += [
        "closed form & " + " & ".join(closed) + r"\\",
        r"\hline",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_constants_table(
    path: Path, results: dict, refinement: int, decimals: int
) -> None:
    """
    LaTeX table of the constants of every configuration, as in the paper.

    Parameters
    ----------
    path : Path
        Output file.
    results : dict
        ``PoincareConstants`` by configuration and refinement.
    refinement : int
        Refinement of the reported values.
    decimals : int
        Decimals of the rounded-up values used in ``rho``.
    """
    lines = [
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"\hline",
        r"configuration & \(h/s\) & \(\norm{\nabla\varphi}_{L^\infty}\,s\) & "
        r"\(c_2\,h/s\) & \(C_P\,h/s\) & \(c_2\,h/s\) used\\",
        r"\hline",
    ]
    for configuration in LINF_CONFIGURATIONS:
        constants = results[configuration, refinement]
        _, latex = CLOSED_FORMS[configuration]
        lines.append(
            f"{DESCRIPTIONS[configuration]} & {DIAMETERS[configuration]} & 1 & "
            f"{constants.weighted:.4f} & "
            f"{constants.unweighted:.4f} \\(\\approx {latex}\\) & "
            f"{round_up(constants.weighted, decimals):.{decimals}f}\\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(refinements: tuple[int, ...] = (4, 8, 16, 32, 64, 128), decimals: int = 3):
    """
    Compute the constants on every refinement and write the tables.

    Parameters
    ----------
    refinements : tuple[int, ...]
        Mesh cells per grid spacing, doubling, at least three.
    decimals : int
        Decimals kept when the constants are rounded up.
    """
    output_directory = create_timestamped_output_directory(
        "experiments", "12_paper_poincare_constants"
    )
    results = {}
    rows = []
    for configuration in LINF_CONFIGURATIONS:
        for refinement in refinements:
            start = time.perf_counter()
            constants = compute_linf_poincare_constants(configuration, refinement)
            elapsed = time.perf_counter() - start
            results[configuration, refinement] = constants
            rows.append(
                {
                    "configuration": configuration,
                    "refinement": refinement,
                    "diameter": constants.diameter,
                    "c2_h": constants.weighted,
                    "c2": constants.weighted / constants.diameter,
                    "CP_h": constants.unweighted,
                    "CP": constants.unweighted / constants.diameter,
                    "time_s": elapsed,
                }
            )
            print(
                f"{configuration:8s} r = {refinement:4d}: c_2 h / s = "
                f"{constants.weighted:.6f}, C_P h / s = {constants.unweighted:.6f} "
                f"({elapsed:.1f}s)"
            )
    with open(output_directory / "constants.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    print("\nExtrapolation from the last three refinements:")
    for configuration in LINF_CONFIGURATIONS:
        for field, label in CONSTANTS.items():
            values = [getattr(results[configuration, r], field) for r in refinements]
            order, limit = extrapolate(values)
            summary[configuration, field] = (
                order,
                limit,
                round_up(values[-1], decimals),
            )
            closed = (
                f", closed form {CLOSED_FORMS[configuration][0]:.6f}"
                if field == "unweighted"
                else ""
            )
            print(
                f"{configuration:8s} {label}: order {order:.2f}, extrapolated "
                f"{limit:.6f}, used {round_up(values[-1], decimals):.{decimals}f}"
                f"{closed}"
            )

    write_convergence_table(
        output_directory / "convergence.tex", results, summary, refinements, decimals
    )
    write_constants_table(
        output_directory / "constants.tex", results, refinements[-1], decimals
    )
    print(
        f"Saved constants.csv, convergence.tex and constants.tex to {output_directory}/"
    )


if __name__ == "__main__":
    main()
