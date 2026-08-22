"""Shared fixtures: small meshes/problems reused across the test suite."""

from __future__ import annotations

import jax.random as jr
import pytest
from jax_fem.mesh import create_gmsh_unit_square_mesh

from vpinns_jax.pde import create_poisson_sin_sin_problem


@pytest.fixture
def unit_square_mesh():
    return create_gmsh_unit_square_mesh(4)


@pytest.fixture
def sin_sin_problem(unit_square_mesh):
    return create_poisson_sin_sin_problem(unit_square_mesh)


@pytest.fixture
def prng_key():
    return jr.PRNGKey(0)
