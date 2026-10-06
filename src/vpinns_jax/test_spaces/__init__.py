"""
Discrete test spaces: conforming finite element spaces and patch spaces.
"""

from vpinns_jax.test_spaces.fe_space import FETestSpace, create_fe_test_space
from vpinns_jax.test_spaces.patch_family import (
    PatchFamily,
    create_linf_grid_family,
    create_vertex_star_family,
)
from vpinns_jax.test_spaces.patch_space import (
    PatchFunctions,
    PatchTestSpace,
    assemble_patch_matrix,
    assemble_patch_vector,
    create_patch_test_space,
)

__all__ = [
    "FETestSpace",
    "create_fe_test_space",
    "PatchFamily",
    "create_vertex_star_family",
    "create_linf_grid_family",
    "PatchFunctions",
    "PatchTestSpace",
    "create_patch_test_space",
    "assemble_patch_vector",
    "assemble_patch_matrix",
]
