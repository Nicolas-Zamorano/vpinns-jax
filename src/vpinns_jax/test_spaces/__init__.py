"""
Discrete test spaces: conforming finite element spaces and patch spaces.
"""

from vpinns_jax.test_spaces.bilinear_family import create_bilinear_family
from vpinns_jax.test_spaces.crisscross_family import create_crisscross_family
from vpinns_jax.test_spaces.crisscross_mesh import (
    create_crisscross_mesh,
    create_crisscross_unit_square_mesh,
)
from vpinns_jax.test_spaces.fe_space import FETestSpace, create_fe_test_space
from vpinns_jax.test_spaces.linf_vertex_family import (
    create_linf_vertex_family,
    linf_poincare_constants,
)
from vpinns_jax.test_spaces.meshfree_space import (
    MeshfreePatchFamily,
    MeshfreePatchTestSpace,
    create_meshfree_patch_family,
    create_meshfree_patch_test_space,
    subdivide_meshfree_family,
)
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
from vpinns_jax.test_spaces.poincare_constants import (
    LINF_CONFIGURATIONS,
    PoincareConstants,
    compute_linf_poincare_constants,
    compute_rectangle_poincare_constants,
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
    "MeshfreePatchFamily",
    "create_meshfree_patch_family",
    "MeshfreePatchTestSpace",
    "create_meshfree_patch_test_space",
    "subdivide_meshfree_family",
    "create_crisscross_family",
    "create_bilinear_family",
    "create_linf_vertex_family",
    "linf_poincare_constants",
    "create_crisscross_mesh",
    "create_crisscross_unit_square_mesh",
    "PoincareConstants",
    "LINF_CONFIGURATIONS",
    "compute_rectangle_poincare_constants",
    "compute_linf_poincare_constants",
]
