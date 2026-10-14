# `vpinns-jax`: Variational Physics-Informed Neural Networks in JAX

A modern, high-performance research framework for **Variational Physics-Informed Neural Networks (VPINNs)** and **Mesh-Free Variational PINNs (MF-VPINNs)** built on [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox), [Optax](https://github.com/google-deepmind/optax), and [`jax-fem`](/home/nicolas/Documents/Github/fem-jax).

---

## 1. Executive Summary & Core Philosophy

### What is `vpinns-jax`?
Classical Physics-Informed Neural Networks (PINNs) train neural networks by minimizing strong residuals of partial differential equations (PDEs) evaluated at scattered collocation points. While conceptually simple, strong-form PINNs require high-order automatic differentiation, struggle with discontinuous coefficients, lack a natural norm-equivalence to the true PDE error, and frequently suffer from optimization failures or spectral bias.

**`vpinns-jax`** formulates the training of neural networks within the **Petrov–Galerkin variational framework**:
1. **Trial functions $w_\theta$** are parameterized by deep neural networks (either directly as smooth global functions with strongly imposed boundary conditions, or through nodal finite element interpolation).
2. **Test functions $\psi_i$** span discrete test spaces—either standard conforming Finite Element (FE) spaces ($H_0^1$) or **Shepard-normalised partition-of-unity patch spaces**.
3. **Loss functions** are formulated not as arbitrary Euclidean sums of residuals, but as **dual-norm (Riesz) representations** ($r^T G^{-1} r$) and **residual error estimators** ($J = \eta_P^2 + \lambda \sum_i \eta_{R,i}^2$). These losses are provably equivalent to the true energy error $\|u - w\|_A$ uniformly under mesh refinement.
4. **Discrete Fracture Networks (DFNs)** are supported using embedded manifold meshes in $\mathbb{R}^3$, capturing discontinuous conormal fluxes and pressure kinks across fracture intersections.

### Core Architectural Principles
- **Built on `jax-fem`'s Three-Tier Assembly**: `vpinns-jax` delegates all integration, quadrature, and mesh topology to `jax-fem`. It adopts the 3-tier pipeline: *pointwise evaluation* $\to$ *cellwise integration* (`integrate_cellwise`) $\to$ *global sparse assembly* (`assemble_matrix`, `assemble_vector`).
- **No Facades or Compatibility Shims**: Submodules are imported directly without re-export layers or all-in-one wrappers. Callers explicitly compose the pipeline:
  $$\text{Problem / Mesh} \longrightarrow \text{Test Space} \longrightarrow \text{Evaluated Trial} \longrightarrow \text{Residual Vector} \longrightarrow \text{Loss Functional}$$
- **Full 64-bit Precision by Default**: Importing `vpinns_jax` automatically imports `jax_fem`, setting `jax_enable_x64 = True` before any JAX array is allocated.
- **Pure JIT & Equinox Workflows**: All network evaluations, residual assemblies, and optimizer steps are fully compatible with `eqx.filter_jit`.

---

## 2. Mathematical Framework

### 2.1 Advection–Diffusion–Reaction Problem
The package considers general second-order elliptic PDEs on a bounded domain $\Omega \subset \mathbb{R}^d$ with Dirichlet boundary $\Gamma = \partial\Omega$:
$$-\nabla \cdot (A(x) \nabla u) + \beta(x) \cdot \nabla u + c(x) u = f(x) \quad \text{in } \Omega, \qquad u = g \quad \text{on } \Gamma$$
where $A(x) \in \mathbb{R}^{d \times d}$ is symmetric positive definite, $\beta(x) \in \mathbb{R}^d$, $c(x) \ge 0$, and $c - \frac{1}{2}\nabla \cdot \beta \ge 0$.

The bilinear and linear forms are:
$$a(w, v) = \int_\Omega \left( \nabla v \cdot A \nabla w + v (\beta \cdot \nabla w) + c w v \right) dx, \qquad \ell(v) = \int_\Omega f v \, dx$$
The energy inner product and energy norm are defined by the symmetric part of the operator:
$$(w, v)_A = \int_\Omega \left( \nabla w \cdot A \nabla v + \left(c - \frac{1}{2}\nabla \cdot \beta\right) w v \right) dx, \qquad \|v\|_A = \sqrt{(v, v)_A}$$

---

### 2.2 Discrete Residuals and Riesz Dual Norms
Let $\{\psi_i\}_{i \in I_\Omega}$ be a basis of test functions vanishing on $\Gamma$. For any trial function $w$, the weak residual vector is:
$$r_i(w) = a(w, \psi_i) - \ell(\psi_i) \quad \text{for } i \in I_\Omega$$
Let $G$ be the Gram matrix of the test functions under the energy inner product:
$$G_{ij} = (\psi_i, \psi_j)_A$$

#### Loss Formulations:
1. **Euclidean Loss**:
   $$\mathcal{L}_{\text{Euclidean}}(w) = r^T r = \sum_{i \in I_\Omega} r_i(w)^2$$
   *Problem*: Lacks dimensional consistency. Under refinement $h \to 0$, $r^T r$ decays at an artificial rate and depends heavily on test function scaling.
2. **Decoupled Riesz Loss ($\eta_D^2$)**:
   $$\eta_D^2(w) = r^T D^{-1} r = \sum_{i \in I_\Omega} \frac{r_i(w)^2}{G_{ii}}, \qquad D = \operatorname{diag}(G)$$
   *Problem*: Treats each test function as if it spanned an independent subspace. It underweights smooth error modes by up to $h^{-2}$ under refinement, making it non-uniform in $h$.
3. **Coupled Riesz Dual-Norm Loss ($\eta_P^2$)**:
   $$\eta_P^2(w) = \|r\|_{V_h^*}^2 = r^T G^{-1} r = \|L^{-1} r\|_2^2$$
   where $G = L L^T$ is the Cholesky factorization of the energy Gram matrix.
   *Key Property*: Efficient with constant 1: $\eta_P(w) \le \|u - w\|_A$. For a conforming Galerkin trial $w \in V_h$, $\eta_P(w) = \|u_h - w\|_A$ exactly.

---

### 2.3 Shepard-Normalised Patch Test Spaces (MF-VPINNs)
In Mesh-Free VPINNs, test functions are defined on overlapping local subdomains (patches) $\{P_i\}_{i \in I_P}$.

#### The Partition of Unity and Shepard Normalisation:
- Each patch $P_i$ has a continuous raw weight function $\phi_i \in C(\bar{\Omega})$ with $0 \le \phi_i \le 1$ and $\operatorname{supp}(\phi_i) = \bar{P}_i$.
- Patches covering the boundary $\Gamma$ belong to $I_\Gamma$; interior patches belong to $I_\Omega = I_P \setminus I_\Gamma$.
- The covering function is:
  $$S(x) = \sum_{i \in I_P} \phi_i(x), \qquad 0 < c_{U2} \le S(x) \le c_{U1} < \infty \quad \forall x \in \bar{\Omega}$$
- **Shepard-normalised test functions**:
  $$\psi_i(x) = \frac{\phi_i(x)}{S(x)}$$
  By construction, $\{\psi_i\}$ forms a true partition of unity on $\bar{\Omega}$:
  $$\sum_{i \in I_P} \psi_i(x) = 1, \qquad \sum_{i \in I_P} \nabla \psi_i(x) = 0$$
- By the quotient rule, their gradients are:
  $$\nabla \psi_i = \frac{\nabla \phi_i}{S} - \frac{\phi_i \nabla S}{S^2}$$

#### Why Raw Weights Fail and Normalisation is Mandatory:
Without Shepard normalisation ($\psi_i = \phi_i$), the dual norm $\eta_P / \|\nabla e\|$ degenerates as $O(h)$ under refinement. With normalisation, $\eta_P / \|\nabla e\|$ settles at a constant ratio independent of $h$ (tested in `04_shepard_normalisation.py`).

#### The Strong Residual Estimator Term ($\eta_{R,i}$):
Because the patch space does not span all of $H_0^1$, the dual norm $\eta_P$ alone only bounds the projected error. The unobserved high-frequency components are captured by weighted local strong residuals:
$$\eta_{R,i}^2 = h_i^2 \|\phi_i^{1/2} \rho(w)\|_{L^2(P_i)}^2$$
where $h_i$ is the patch diameter, $\phi_i$ is the raw weight, and $\rho(w)$ is the strong PDE residual:
$$\rho(w) = f + \nabla \cdot (A \nabla w) - \beta \cdot \nabla w - c w$$
Because smooth neural networks have no edge jump discontinuities ($j_e \equiv 0$), edge residuals vanish identically.

#### Total Estimator Loss:
$$J(\theta) = \eta_P^2(w_\theta) + \lambda \sum_{i \in I_P} \eta_{R,i}^2(w_\theta)$$
This loss provides a provably reliable and efficient bound on the true energy error $\|u - w\|_A$.

---

### 2.4 Representation of Raw Weights via Integration Meshes
To ensure exact evaluation without heuristic numerical integration:
1. Every patch family is equipped with an **integration mesh** (a `TriangleMesh`) that refines the fan triangulation of every patch.
2. Each raw weight $\phi_i$ is continuous and affine on every triangle, meaning $\phi_i$ is a standard P1 finite element function stored through its nodal values.
3. As a result:
   - $S(x)$ is piecewise linear; its minimum $c_{U2}$ and maximum $c_{U1}$ are attained strictly at mesh vertices.
   - Any family that fails to cover the domain (i.e. $S = 0$ at any vertex) is detected and rejected at construction.
   - Quadrature of $\psi_i = \phi_i / S$ is rational and smooth inside each cell (no interior kinks).

Two families are implemented:
- **Vertex-Star Family**: Patches are the vertex stars of a triangulation; $\phi_z = \lambda_z$ (barycentric coordinates). Here $S \equiv 1$, $c_{U2} = c_{U1} = 1$, and the patch space coincides exactly with the standard P1 finite element test space.
- **$\ell^\infty$ Grid Family**: Patches are $\ell^\infty$ cones centered at grid vertices with spacing $s$: $\phi_c(x) = \max(0, 1 - \|x - c\|_\infty / s)$. The integration mesh is a criss-cross triangulation. On $[0, 1]^2$, covering bounds are exactly $1 \le S \le 2$. Arbitrary domains (such as L-shapes) are supported via boolean grid masks.

---

### 2.5 Trial Function Formulations

```
                    ┌─────────────────────────────────────────┐
                    │            Neural Network               │
                    │        u_theta(x) : R^d -> R            │
                    └────────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │     Direct Trial      │                       │  Interpolated Trial   │
     │  w = B u_theta + g_bar│                       │  w = Pi_h(u_theta)    │
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │                                               │
     • Smooth everywhere (C^k)                       • P1 FE function on mesh
     • Strongly satisfies BCs on Gamma               • Nodal values set at DOFs
     • Tested against FE or Patch spaces             • Tested against FE space
     • Used for planar domains (01, 03, 05)          • Mandatory for DFNs (02, 06)
```

1. **Direct Trial (`ConstrainedAnsatz`)**:
   $$w(x) = B(x) u_\theta(x) + \bar{g}(x)$$
   - $B(x)$ is an exact distance function vanishing on $\Gamma$ (`ExactDistanceFunction` for unit square, disk, or L-shape).
   - $\bar{g}(x)$ is a Dirichlet lift function.
   - Evaluated at quadrature points via vectorized automatic differentiation (`evaluate_network`, `evaluate_network_flux_divergence`).
2. **Interpolated Trial (`interpolate_network`)**:
   $$w = \sum_{j \in \text{free}} u_\theta(x_j) \phi_j + \sum_{j \in \text{Dirichlet}} g(x_j) \phi_j \in V_h$$
   - The network is evaluated at mesh nodal coordinates, and boundary values are overridden.
   - Residual simplifies to linear algebraic matrix-vector multiplication:
     $$r = K w - F$$
   - Essential for Discrete Fracture Networks (DFNs).

---

### 2.6 Discrete Fracture Networks (DFN) Limitation & Resolution
On a Discrete Fracture Network (DFN), 2D planar fractures intersect along 1D segments called traces. Physical pressure solutions exhibit sharp kinks across traces, exchanging normal fluxes $\pm 1/3$.

- **Failure of Direct Smooth Trials**: A smooth network $u_\theta$ on $\mathbb{R}^3$ has continuous gradients. At an X-crossing of fractures, the conormal fluxes from the two halves of a fracture cancel identically, so **a smooth direct trial exchanges zero flux across the trace**.
- **Success of Interpolated Trials**: The FE mesh on a DFN shares nodes along traces but allows discontinuous gradients across element boundaries. The interpolated trial $w = \Pi_h(u_\theta)$ can kink within one element of the trace, correctly capturing cross-fracture flux exchange.

---

## 3. Repository Architecture & Codebase Map

```
vpinns-jax/
├── context/
│   └── vpinns_migration_plan.md    # Detailed roadmap and technical specification
├── pyproject.toml                  # Packaging, uv build backend, dependencies
├── README.md                       # Quickstart summary
├── REPOSITORY_OVERVIEW.md          # Comprehensive architecture & theory guide (this file)
├── src/
│   └── vpinns_jax/                 # Main Python package
│       ├── __init__.py             # Imports jax_fem (sets x64), empty __all__
│       ├── models/                 # Neural trial functions & evaluation
│       │   ├── ansatz.py           # MultiLayerPerceptron, FourierFeatureMLP
│       │   ├── boundary.py         # ExactDistanceFunction, ConstrainedAnsatz, interpolate_network
│       │   └── evaluation.py       # evaluate_network, evaluate_network_flux_divergence
│       ├── test_spaces/            # Discrete test spaces & patch families
│       │   ├── fe_space.py         # FETestSpace, create_fe_test_space
│       │   ├── patch_family.py     # PatchFamily, vertex-star & l^inf grid families
│       │   └── patch_space.py      # PatchTestSpace, create_patch_test_space, assemble_patch_*
│       ├── losses/                 # Weak and strong loss functionals
│       │   ├── residual.py         # FE and patch residuals, strong residual terms
│       │   ├── riesz.py            # Cholesky-based Riesz loss, diagonal Riesz loss
│       │   ├── euclidean.py        # Plain r^T r loss
│       │   └── boundary_residual.py# Boundary mismatch L^2 penalty
│       ├── solvers/                # Optimization routines
│       │   └── trainer.py          # Adam, L-BFGS (Optax), unified train, early stopping
│       └── analysis/               # Error functionals, monitoring, and plotting
│           ├── errors.py           # compute_network_error, compute_solution_norm
│           ├── monitoring.py       # TrainingRecord, create_training_monitor, robustness
│           └── plotting.py         # Okabe-Ito plots, 3D solution surfaces, DFN cell errors
├── examples/                       # Runnable demonstration scripts
│   ├── 01_vpinns_poisson_direct.py # Direct trial with FE space (Euclidean vs. Riesz)
│   ├── 02_vpinns_interpolated.py   # Interpolated trial with FE space
│   ├── 03_mf_vpinns_patch.py       # MF-VPINN with Shepard patch space (4 losses compared)
│   ├── 04_shepard_normalisation.py # Fixed-error verification of Shepard normalisation
│   ├── 05_convergence_study.py     # Uniform refinement study of MF-VPINN estimator loss
│   └── 06_dfn_interpolated.py      # DFN two-fracture benchmark with interpolated trial
├── tests/                          # Complete pytest suite (85 tests)
│   ├── conftest.py                 # Shared fixtures (meshes, problems, FE spaces)
│   ├── test_dfn.py                 # DFN interpolation, Galerkin equivalence, trace flux
│   ├── test_errors.py              # compute_network_error against jax-fem functionals
│   ├── test_evaluation.py          # Vectorized evaluation, gradients, flux divergence
│   ├── test_fe_space.py            # FE operator shapes and partitions
│   ├── test_losses.py              # Residuals, Galerkin orthogonality, Riesz duality
│   ├── test_models.py              # MLP, FourierFeatureMLP, boundary enforcement
│   ├── test_monitoring.py          # TrainingRecord and monitor behavior
│   ├── test_patch_family.py        # Vertex-star & l^inf grid geometry and covering bounds
│   ├── test_patch_losses.py        # Patch residual, residual terms, Remark table values
│   ├── test_patch_space.py         # Partition of unity, vertex-star equivalence
│   ├── test_plotting.py            # Plotting utilities and formatting
│   └── test_solvers.py             # Adam and Optax L-BFGS convergence
├── experiments/                    # Timestamped outputs from example runs (plots & logs)
├── figures/                        # Generated vector diagrams and publication figures
├── ml_table.py                     # Multilevel criss-cross hierarchy condition number table
├── plot_nvb.py                     # Newest-vertex bisection (NVB) refinement visualization
└── plot_refinement.py              # Nested cell patch refinement visualization
```

---

## 4. Module Deep Dive

### 4.1 `models/`
- **`ansatz.py`**:
  - `MultiLayerPerceptron`: Wraps `eqx.nn.MLP` with single scalar output `()` for an input `(d,)`. Defaults to `tanh` activations for smooth second derivatives.
  - `FourierFeatureMLP`: Maps $x \mapsto [\cos(2\pi B x), \sin(2\pi B x)]$ before passing through an MLP, mitigating the spectral bias of standard MLPs on high-frequency components.
- **`boundary.py`**:
  - `ExactDistanceFunction`: Computes distance functions $B(x) \ge 0$ that vanish exactly on $\partial\Omega$. Supports `"unit_square"` ($x(1-x)y(1-y)$), `"circle"` ($1 - \|x\|^2$), and `"l_shape"` (signed polygonal edge distance).
  - `ConstrainedAnsatz`: Composes $w(x) = B(x) u_\theta(x) + \bar{g}(x)$ so Dirichlet boundary conditions are satisfied by construction.
  - `interpolate_network`: Nodal interpolant $\Pi_h(u_\theta) \in V_h$. Evaluates network at `space.dof_coordinates` and overwrites Dirichlet DOFs.
- **`evaluation.py`**:
  - `evaluate_network_values(network, points)`: Evaluates network values at row-vector points `(..., 1, d)` using `jnp.vectorize(..., signature="(d)->()")`.
  - `evaluate_network(network, points)`: Evaluates values and gradients via `jax.value_and_grad`. Returns a `FiniteElementFunctionEvaluation` matching `jax-fem`'s trial basis protocol.
  - `evaluate_network_flux_divergence(network, diffusion, points)`: Computes $\nabla \cdot (A(x) \nabla w(x)) = \operatorname{tr}(\mathcal{J}_x [A(x) \nabla w(x)])$ via forward-mode AD (`jax.jacfwd` and `jnp.trace`). Correctly differentiates variable diffusion tensors $A(x)$.

### 4.2 `test_spaces/`
- **`fe_space.py`**:
  - `FETestSpace`: Immutable dataclass holding `basis`, `coefficients`, `dirichlet_dofs`, `dirichlet_values`, `free_dofs`, `stiffness`, `load`, and the energy `gram` matrix.
  - `create_fe_test_space`: Eagerly evaluates Dirichlet conditions, partitions DOFs into free and Dirichlet, and assembles the global operators.
- **`patch_family.py`**:
  - `PatchFamily`: Dataclass containing the integration mesh, `cells_to_patches` $(K, M)$, `cell_patch_mask`, `local_weights` $(K, 1, M, 3)$, `patch_diameters`, `boundary_patch_mask`, `interior_patches`, and `covering_bounds` $(c_{U2}, c_{U1})$.
  - `create_vertex_star_family`: Patches are vertex stars of a triangle mesh. $S \equiv 1$, covering bounds $(1, 1)$.
  - `create_linf_grid_family`: $\ell^\infty$ cones on a criss-cross mesh for any union of grid squares. Covering bounds on unit square are $(1.0, 2.0)$.
  - `_create_patch_family`: Validates continuity, rejects unlisted supports, checks covering $S > 0$, and computes patch diameters.
- **`patch_space.py`**:
  - `PatchFunctions`: Evaluated patch values $(K, Q, M, 1)$ and gradients $(K, Q, M, d)$.
  - `PatchTestSpace`: Stores patch test functions $\psi_i$, raw weights $\phi_i$, and the assembled energy Gram matrix $P$.
  - `create_patch_test_space`: Applies Shepard quotient rule to compute values and gradients of $\psi_i = \phi_i / S$, assembling the global energy Gram matrix via `assemble_patch_matrix`.
  - `assemble_patch_vector` / `assemble_patch_matrix`: Custom sparse assembly using BCOO with `cells_to_patches` as the DOF map.

### 4.3 `losses/`
- **`residual.py`**:
  - `compute_fe_residual(fe_test_space, trial)`: Assembles $a(w, \psi_i) - \ell(\psi_i)$ for all DOFs.
  - `compute_interpolated_fe_residual(fe_test_space, dof_values)`: Computes $K w - F$.
  - `compute_patch_residual(patch_test_space, trial)`: Assembles weak residual against all patch test functions.
  - `compute_patch_residual_terms(patch_test_space, trial, flux_div)`: Computes weighted strong residual terms $\eta_{R,i}^2 = h_i^2 \|\phi_i^{1/2} \rho(w)\|^2$.
- **`riesz.py`**:
  - `restrict_to_free(gram, free_indices)`: Extracts dense submatrix $G_{ff}$ or $P_{I_\Omega, I_\Omega}$.
  - `riesz_loss(free_residual, factor)`: Evaluated as $\|L^{-1} r\|^2$ via `jax.scipy.linalg.solve_triangular`.
  - `diagonal_riesz_loss(free_residual, gram_diagonal)`: Decoupled loss $r^T D^{-1} r$.
- **`euclidean.py`**:
  - `euclidean_loss(residual)`: $r^T r$.
- **`boundary_residual.py`**:
  - `boundary_residual_loss`: Facet-integrated $\int_\Gamma (w - g)^2 \, ds$ for penalty-based enforcement.

### 4.4 `solvers/`
- **`trainer.py`**:
  - `train_adam`: Adam optimizer with cosine/exponential/constant schedules, early stopping, and step monitoring.
  - `train_lbfgs`: Optax-based L-BFGS implementation with line search.
  - `train`: Unified two-stage optimization pipeline (Adam pretraining followed by L-BFGS fine-tuning).
  - Returns a structured `TrainingHistory` containing loss histories and monitor evaluations.

### 4.5 `analysis/`
- **`errors.py`**:
  - `compute_network_error`: Evaluates network error against exact solutions using `jax-fem` functionals (`l2_error_functional`, `h1_seminorm_error_functional`, `energy_error_functional`).
  - `compute_solution_norm`: Computes $\|u\|_A$ by passing a zero function.
- **`monitoring.py`**:
  - `create_training_monitor`: Creates a callback evaluated during training that records:
    - Loss $L$ and named constituent terms
    - Energy error $\|u - w\|_A$
    - Relative loss $\sqrt{L} / \|u\|_A$
    - Relative energy error $\|u - w\|_A / \|u\|_A$
    - **Robustness ratio**: $\sqrt{L} / \|u - w\|_A$ (a reliable estimator keeps this bounded across all training iterations).
- **`plotting.py`**:
  - Publication-quality plotting using the **Okabe–Ito colorblind-safe palette**, distinct linestyles, and major/minor grids.
  - `plot_network_solution`: Draws 3D surfaces of $u$, $w_\theta$, and $|u - w_\theta|$ at full resolution.
  - `plot_cell_errors`: Cellwise $L^2$ and energy error maps on planar domains or across fractures of a DFN.
  - `save_plot`: Standardized figure exporter saving at 500 dpi.

---

## 5. Walkthrough of Examples

| Example | Method | Trial Space | Test Space | Key Focus |
| :--- | :--- | :--- | :--- | :--- |
| `01_vpinns_poisson_direct.py` | Direct VPINN | Direct ($B u_\theta$) | Conforming P1 FE | Euclidean vs. Riesz loss; Adam vs. L-BFGS |
| `02_vpinns_interpolated.py` | Interpolated VPINN | Nodal P1 ($\Pi_h u_\theta$) | Conforming P1 FE | Matrix-vector residual $Kw - F$; cellwise error tracking |
| `03_mf_vpinns_patch.py` | MF-VPINN | Direct ($B u_\theta$) | Shepard $\ell^\infty$ Patch Space | Comparison of 4 losses: $\eta_D^2$, $\eta_P^2$, $\eta_D^2 + \lambda\eta_R^2$, $\eta_P^2 + \lambda\eta_R^2$ |
| `04_shepard_normalisation.py` | Analysis (no training) | Exact error $e = \sin(\pi x)\sin(\pi y)$ | Raw vs. Shepard Patches | Proof that normalisation is required for $h$-uniformity |
| `05_convergence_study.py` | Convergence Study | Direct ($B u_\theta$) | Shepard $\ell^\infty$ Patch Space | Uniform $h$-refinement ($n=4 \dots 32$); estimator vs. error |
| `06_dfn_interpolated.py` | DFN Benchmark | Nodal P1 on DFN | Embedded P1 FE on DFN | Cross-trace flux exchange ($\pm 1/3$) on intersecting fractures |

---

## 6. Implementation Plan & Migration History

As documented in [`context/vpinns_migration_plan.md`](context/vpinns_migration_plan.md), the repository was systematically redesigned and migrated from an earlier prototype onto the modern `jax-fem` three-tier architecture:

### Phase A: Migration onto `jax-fem` Primitives (Complete)
- Removed all legacy facades (`pde/`, `estimators/`, `adaptive_loop.py`).
- Implemented vectorized network evaluation (`evaluate_network`, `evaluate_network_flux_divergence`) using JAX signature vectorization.
- Replaced custom assembly with `jax-fem` forms and assemblers.
- Implemented Cholesky-factorized Riesz loss in `losses/riesz.py`.
- Replaced jaxopt with Optax L-BFGS in `solvers/trainer.py`.

### Phase B: Shepard Patch Test Spaces (Complete)
- Implemented `PatchFamily` with P1 raw weights on integration meshes (`patch_family.py`).
- Implemented `create_vertex_star_family` and `create_linf_grid_family` with boundary and covering validation.
- Implemented `PatchTestSpace` with Shepard quotient rule and custom BCOO sparse matrix assembly (`patch_space.py`).
- Implemented weighted strong residual terms $\eta_{R,i}^2$ in `losses/residual.py`.
- Implemented monitoring and publication plotting (Okabe-Ito, 500 dpi, 3D uniform surfaces).
- Verified mathematical equivalence and reproduces the note's Remark on normalisation (`04_shepard_normalisation.py`).

### Phase C: Discrete Fracture Networks (Complete)
- Verified interpolated trial against embedded P1 FE test space on intersecting fractures (`test_dfn.py`, `06_dfn_interpolated.py`).
- Correctly reproduced trace flux exchange ($\pm 0.31$ numerical vs. $\pm 0.319$ Galerkin vs. $\pm 1/3$ exact).
- Documented the fundamental limitation of direct smooth trials on DFNs.

### Open & Deferred Work:
- **Adaptive Mesh Refinement (AMR)**: Localizing $\eta_P^2$ into cellwise indicators via the Riesz representative $\Phi_P = \sum_i (P^{-1}r)_i \psi_i$.
- **Sharp Loss Constants**: Offline eigenvalue calculation of $\|\Pi_P\|$ replacing the hyperparameter $\lambda$.
- **Arbitrary Polygonal Patches**: Triangulated polygon overlays for non-grid patch families.

---

## 7. How to Run & Verify

The repository uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### Running Tests
Execute the entire test suite (85 tests):
```bash
uv run pytest
```

### Running Examples
```bash
# 01. Direct VPINN for Poisson problem
uv run python examples/01_vpinns_poisson_direct.py

# 02. Interpolated VPINN for Poisson problem
uv run python examples/02_vpinns_interpolated.py

# 03. Mesh-Free VPINN with Shepard patch space (compares 4 loss functionals)
uv run python examples/03_mf_vpinns_patch.py

# 04. Shepard normalisation demonstration (no training)
uv run python examples/04_shepard_normalisation.py

# 05. Uniform convergence study across patch refinements
uv run python examples/05_convergence_study.py

# 06. Discrete Fracture Network benchmark
uv run python examples/06_dfn_interpolated.py
```
All training histories, error logs, and 500-dpi publication figures are automatically saved under timestamped subdirectories in `experiments/`.
