"""Proof of concept: interpolated VPINN for the Poisson ``sin(pi x) sin(pi y)`` problem.

Trial representation: **interpolated**, ``w_theta = Pi_h u_theta`` (math
reference Section 2.1 / Section 3 "Strong, on the interpolant"), built by
coordinate interpolation of the network at the P1 mesh's nodal
coordinates:

    w_theta = sum_{i in interior} u_theta(x_i) phi_i + sum_{i in boundary} g(x_i) phi_i

i.e. the network is only evaluated at *interior* node coordinates; boundary
node coordinates get the exact Dirichlet value directly, so the resulting
interpolant satisfies the boundary condition exactly, without any boundary
loss term.

Test space: the FE space V_h itself, restricted to interior DOFs (V_h subset
H^1_0). Because the trial function is now literally a P1 finite element
function, the discrete weak residual r_h(w_theta, psi_i) = a_h(w_theta,
psi_i) - l_h(psi_i) is *just* `stiffness @ w_dofs - load` restricted to
interior rows -- no custom quadrature code needed beyond jax-fem's existing
elliptic bilinear/linear forms.

Loss: Euclidean loss L_E = sum_i r_h(w_theta, psi_i)^2 (math reference
Section 2.3). This is a proof of concept only: no Riesz loss, no patch test
spaces, no a posteriori estimator, no adaptive refinement -- see
context/implementation_plan.md for the full library scope this exercises a
first slice of.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them

import optax
from jax_fem.assembly.matrix import assemble_bilinear_form
from jax_fem.assembly.vector import assemble_linear_form
from jax_fem.constraints.dirichlet import evaluate_dirichlet_dof_values
from jax_fem.diagnostics.errors import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_relative_l2_error,
)
from jax_fem.diagnostics.plotting import (
    create_timestamped_output_directory,
    plot_fem_error,
    plot_fem_error_3d,
    save_plot,
)
from jax_fem.element.lagrange_triangle import LagrangeTriangleP1
from jax_fem.forms.elliptic import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.mesh.gmsh import create_gmsh_unit_square_mesh
from jax_fem.problem.elliptic import evaluate_elliptic_coefficients
from jax_fem.problem.manufactured import create_poisson_sin_sin_problem
from jax_fem.reference_cell.triangle import ReferenceTriangle
from jax_fem.space.cell_basis import create_cell_basis
from jax_fem.space.finite_element_space import create_finite_element_space

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MLP(eqx.Module):
    """Scalar neural trial function u_theta: R^2 -> R (Equinox MLP, tanh)."""

    net: eqx.nn.MLP

    def __init__(self, width: int, depth: int, key: jax.Array):
        self.net = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, xy: jax.Array) -> jax.Array:
        return self.net(xy)[0]


def build_discretization(cells_per_side: int, quadrature_degree: int):
    """Build the mesh, P1 test space, problem data, and the constant FE operators.

    Returns the raw arrays needed by the (jit-compiled) training step
    -- ``FiniteElementSpace``/``CellBasis``/``EllipticProblem`` are plain
    dataclasses, not registered JAX pytrees, so they must stay outside any
    jitted function; only their array-valued attributes are passed in.
    """
    mesh = create_gmsh_unit_square_mesh(cells_per_side)
    element = LagrangeTriangleP1(ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    problem = create_poisson_sin_sin_problem(mesh)

    quadrature = element.reference_cell.create_quadrature(quadrature_degree)
    basis = create_cell_basis(space, quadrature)
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)

    stiffness = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    load = assemble_linear_form(elliptic_linear_form, basis, coefficients)

    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    is_dirichlet = (
        jnp.zeros((space.number_of_dofs,), dtype=bool).at[dirichlet_dofs].set(True)
    )
    free_dofs = jnp.nonzero(~is_dirichlet)[0]

    dof_coordinates = space.dof_coordinates[:, 0, :]

    return {
        "mesh": mesh,
        "space": space,
        "problem": problem,
        "basis": basis,
        "stiffness": stiffness,
        "load": load,
        "dirichlet_dofs": dirichlet_dofs,
        "dirichlet_values": dirichlet_values,
        "free_dofs": free_dofs,
        "dof_coordinates": dof_coordinates,
        "number_of_dofs": space.number_of_dofs,
    }


def interpolated_trial_dofs(
    model: MLP,
    dof_coordinates: jax.Array,
    free_dofs: jax.Array,
    dirichlet_dofs: jax.Array,
    dirichlet_values: jax.Array,
    number_of_dofs: int,
) -> jax.Array:
    """Coordinate interpolation of ``w_theta = Pi_h u_theta`` with a strong BC.

    Interior DOFs get the network's value at their coordinate; boundary
    DOFs get the exact Dirichlet value at their coordinate -- so the
    boundary condition holds exactly, by construction, with no boundary
    loss term.
    """
    interior_values = jax.vmap(model)(dof_coordinates[free_dofs])  # (N_free,)

    dof_values = jnp.zeros((number_of_dofs,))
    dof_values = dof_values.at[free_dofs].set(interior_values)
    dof_values = dof_values.at[dirichlet_dofs].set(dirichlet_values[:, 0])
    return dof_values[:, None]  # (N_dofs, 1)


def euclidean_residual_loss(
    model: MLP,
    dof_coordinates: jax.Array,
    free_dofs: jax.Array,
    dirichlet_dofs: jax.Array,
    dirichlet_values: jax.Array,
    number_of_dofs: int,
    stiffness,
    load: jax.Array,
) -> jax.Array:
    """L_E(w_theta) = mean_i r_h(w_theta, psi_i)^2 over interior test functions."""
    w_dofs = interpolated_trial_dofs(
        model,
        dof_coordinates,
        free_dofs,
        dirichlet_dofs,
        dirichlet_values,
        number_of_dofs,
    )
    residual = stiffness @ w_dofs - load  # r_h(w_theta, psi_i) for every i, (N_dofs, 1)
    free_residual = residual[free_dofs, 0]
    return jnp.mean(free_residual**2)


@eqx.filter_jit
def train_step(
    model: MLP,
    opt_state,
    optimizer: optax.GradientTransformation,
    dof_coordinates: jax.Array,
    free_dofs: jax.Array,
    dirichlet_dofs: jax.Array,
    dirichlet_values: jax.Array,
    number_of_dofs: int,
    stiffness,
    load: jax.Array,
):
    loss, grads = eqx.filter_value_and_grad(euclidean_residual_loss)(
        model,
        dof_coordinates,
        free_dofs,
        dirichlet_dofs,
        dirichlet_values,
        number_of_dofs,
        stiffness,
        load,
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def plot_training_loss(loss_history: jax.Array) -> Figure:
    """Log-scale Adam-step vs. Euclidean-loss curve."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(jnp.arange(loss_history.shape[0]), loss_history, color="tab:blue")
    axis.set_yscale("log")
    axis.set_xlabel("Adam step")
    axis.set_ylabel(r"Euclidean loss $L_E(w_\theta)$")
    axis.set_title("Training loss")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    return figure


def main(
    cells_per_side: int = 32,
    quadrature_degree: int = 4,
    width: int = 32,
    depth: int = 3,
    num_steps: int = 10000,
    learning_rate: float = 1e-3,
    log_every: int = 200,
    seed: int = 0,
    output_directory: str = "experiments",
) -> None:
    disc = build_discretization(cells_per_side, quadrature_degree)

    model = MLP(width=width, depth=depth, key=jr.PRNGKey(seed))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_args = (
        disc["dof_coordinates"],
        disc["free_dofs"],
        disc["dirichlet_dofs"],
        disc["dirichlet_values"],
        disc["number_of_dofs"],
        disc["stiffness"],
        disc["load"],
    )

    print(
        f"Poisson sin(pi x) sin(pi y), unit square, {cells_per_side}x{cells_per_side} "
        f"P1 mesh ({disc['number_of_dofs']} DOFs, {disc['free_dofs'].shape[0]} free)."
    )
    start = time.perf_counter()
    loss_history = []
    for step in range(num_steps):
        model, opt_state, loss = train_step(model, opt_state, optimizer, *train_args)
        loss_history.append(loss)  # stays a device array; no host sync per step
        if step % log_every == 0 or step == num_steps - 1:
            print(f"  step {step:5d}  euclidean loss {float(loss):.6e}")
    elapsed = time.perf_counter() - start
    print(f"Trained {num_steps} Adam steps in {elapsed:.2f}s.")
    loss_history = jnp.stack(loss_history)

    final_dofs = interpolated_trial_dofs(
        model,
        disc["dof_coordinates"],
        disc["free_dofs"],
        disc["dirichlet_dofs"],
        disc["dirichlet_values"],
        disc["number_of_dofs"],
    )
    solution = FiniteElementFunction(space=disc["space"], dof_values=final_dofs)

    relative_l2 = compute_relative_l2_error(solution, disc["basis"], disc["problem"])
    h1_seminorm = compute_h1_seminorm_error(solution, disc["basis"], disc["problem"])
    energy = compute_energy_error(solution, disc["basis"], disc["problem"])

    print(f"Relative L2 error : {float(relative_l2):.6e}")
    print(f"H1 seminorm error : {float(h1_seminorm):.6e}")
    print(f"Energy norm error : {float(energy):.6e}")

    output_dir = create_timestamped_output_directory(
        output_directory, disc["problem"].name
    )
    save_plot(plot_training_loss(loss_history), output_dir / "training.png")
    save_plot(
        plot_fem_error_3d(solution, disc["basis"], disc["problem"], disc["mesh"]),
        output_dir / "3d.png",
    )
    save_plot(
        plot_fem_error(solution, disc["basis"], disc["problem"], disc["mesh"]),
        output_dir / "mesh_error.png",
    )
    print(f"Saved plots to {output_dir}/ (training.png, 3d.png, mesh_error.png)")


if __name__ == "__main__":
    main()
