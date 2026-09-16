# Tangent Blow-Ups

Geometry processing on **non-manifold point clouds** via iterated tangent
(Nash) blow-ups.

Given a point cloud with tangent information (unit tangents for curves,
normals for surfaces), the library lifts each point into the product space
`R^n x Gr(d, n)` — position × tangent plane — using the Chordal–Sasaki
embedding. Points on different sheets that pass through the same location
(self-intersections, tangential contacts, creases) become *separated* in the
lifted space, so ordinary graph-based tools — k-NN graphs, Laplacians, the heat
method — behave correctly across singularities. Lifting again (level 2, 3, …)
exposes curvature and higher-order invariants.

What you can compute with it:

- **Geodesic distances** on non-manifold curves/surfaces (lifted heat method)
- **Graph Laplacians** and spectral clustering that respect sheet structure
- **Curvature invariants** (shape operator, principal / Gaussian / mean
  curvature, Codazzi tensor) directly from the blow-up
- **Reconstruction** of the lifted manifold (neural Poisson implicit fit,
  B-spline curves)

---

## Installation

Requires Python ≥ 3.11. The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone <this repo>
cd tangent-blowups
uv sync            # creates .venv with numpy, scipy, scikit-learn, torch, matplotlib
                   # and installs tangent_blowups in editable mode
uv run pytest      # sanity check
```

Or with plain pip into an existing environment:

```bash
pip install -e .
```

Core dependencies are `numpy` (≥ 1.26, 2.x supported), `scipy`,
`scikit-learn`, `torch` (used by the neural Poisson reconstructor and the
optional CUDA path of the heat method) and `matplotlib` (for
`tangent_blowups.viz`).

Some example scripts need extra packages that are **not** declared in
`pyproject.toml`; install them yourself if you want to run those scripts:

| Package            | Used by                                                |
| ------------------ | ------------------------------------------------------ |
| `open3d`           | `examples/io/mesh_to_pc.py` (normal estimation for mesh → point-cloud conversion) |
| `polyscope`        | 3-D interactive viewers in `examples/geodesics/3D`, `examples/curvature`, `examples/surface_parameterization`, `examples/teaser` |
| `umap-learn`       | `examples/iterated_blowups/*` (UMAP + HDBSCAN visualisations) |
| `robust_laplacian` | `examples/geodesics/3D/pointcloud.py`, `scalar_field_bilateral_comparison.py` (baseline comparison) |
| `pytest`           | running `tests/`                                       |

---

## Quick start: geodesic distances

The main entry point is `tangent_blowups.pointcloud.lifted_heat_method`. It
implements the heat method of Crane et al. on the lifted point cloud: heat
diffusion → normalised gradient → Poisson solve, all with a graph Laplacian
built in the lifted metric.

### Curve in 2-D (points + unit tangents)

```python
import numpy as np
from tangent_blowups.testsupport import figure8, UniformCurve, sample
from tangent_blowups.pointcloud import lifted_heat_method

# Sample a self-intersecting figure-8 with tangents (or bring your own arrays).
s = sample(
    figure8(scale=2.0),
    UniformCurve(n=800, t_min=0.0, t_max=2 * np.pi, endpoint=False),
    with_tangents=True,
)
points, tangents = s.points, s.tangents      # (N, 2), (N, 2)

dist = lifted_heat_method(points, tangents, source_index=0, k=20, alpha=1.0)
dist = np.maximum(dist - dist[0], 0.0)       # anchor at source, clip noise
```

Because the two lobes of the figure-8 have different tangent directions at the
crossing, the lifted graph does not short-cut across it: distances go *around*
the curve rather than jumping between lobes.

### Surface in 3-D (points + unit normals)

```python
import numpy as np
from tangent_blowups.testsupport import cube_surface, RandomSurface, sample
from tangent_blowups.pointcloud import lifted_heat_method

s = sample(
    cube_surface(scale=1.0),
    RandomSurface(n=3000, u_bounds=(0, 6), v_bounds=(0, 1),
                  rng=np.random.default_rng(0)),
    with_normals=True,
)
points, normals = s.points, s.normals        # (N, 3), (N, 3)

dist = lifted_heat_method(points, normals, source_index=0, k=20, alpha=1.0)
```

### Points only (estimate normals first)

```python
from tangent_blowups.pointcloud import estimate_normals_pca, lifted_heat_method

normals = estimate_normals_pca(points, k=20)          # local PCA, (N, 3)
dist = lifted_heat_method(points, normals, source_index=0, k=20)
```

### Many sources on the same cloud (precompute once)

```python
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.pointcloud import lifted_heat_method, precompute_heat_method

level0 = BlowUpLevel.from_normals(points, normals)   # or .from_point_tangents(points, frames)
level1 = level0.lift(k=20, alpha=1.0)                # Chordal–Sasaki lift

cache = precompute_heat_method(level1, k=20)         # Laplacian, mass matrix, heat operator, ...
d_a = lifted_heat_method(level1, source_index=0,   k=20, _precomputed=cache)
d_b = lifted_heat_method(level1, source_index=100, k=20, _precomputed=cache)
```

### How the second argument is interpreted

When you pass raw arrays, `subspace_basis` is interpreted as:

| ambient dim `n` | `subspace_basis` shape | meaning                                  |
| --------------- | ---------------------- | ---------------------------------------- |
| 3               | `(N, 3)`               | **surface normals** → tangent 2-frames   |
| any             | `(N, n)`  (n ≠ 3)      | unit **tangent** of a curve (d = 1)      |
| any             | `(N, n, d)`            | explicit orthonormal tangent `d`-frame   |

So for a *curve in 3-D*, pass the tangents as `(N, 3, 1)` (or build a
`BlowUpLevel.from_point_tangents(points, tangents)` yourself) — a plain
`(N, 3)` array would be treated as normals.

### Important parameters

| Parameter              | Default    | Meaning                                                                 |
| ---------------------- | ---------- | ----------------------------------------------------------------------- |
| `k`                    | `20`       | k-NN size for the graph and the curvature regression inside `lift()`.   |
| `alpha`                | `1.0`      | Weight of the tangent-plane term in the lifted metric. `alpha=0` gives the plain Euclidean heat method (no lifting). |
| `kernel`               | `"product"`| `"product"`: fixed-bandwidth spatial × angular Gaussian (PD). `"self-tuning"`: Zelnik-Manor adaptive bandwidth on the full embedding. |
| `sigma_x`, `sigma_u`   | `None`     | Product-kernel bandwidths; `None` auto-estimates from median k-NN distances. |
| `t`, `t_scale`         | `None`, `1.0` | Diffusion time; `None` auto-estimates, `t_scale` multiplies it.       |
| `diffusion_steps`      | `1`        | Number of implicit diffusion steps (spreads heat farther without raising `t`). |
| `laplacian_normalized` | `False`    | Use the symmetric-normalised Laplacian instead of `L = D − W`.          |
| `device`               | `None`     | `"cuda"` uses the torch sparse-solver path; `None` auto-detects.        |
| `return_intermediate`  | `False`    | Return `(phi, u, X, div)` instead of just distances.                    |
| `timings`              | `None`     | Pass a dict to receive per-stage wall-clock times.                      |

Runnable versions with plots: `examples/geodesics/2D/fig8.py`,
`examples/geodesics/3D/cube.py`.

---

## Core object: `BlowUpLevel`

`tangent_blowups.geometry.iterated_grassmann.BlowUpLevel` holds one level of
the blow-up for `N` points:

| Attribute      | Shape        | Meaning                                                              |
| -------------- | ------------ | -------------------------------------------------------------------- |
| `embedded`     | `(N, D)`     | Embedded position `Φ_i` at this level. First `n_orig` columns are always the original spatial coordinates. |
| `frame`        | `(N, D, d)`  | Orthonormal tangent frame `U_i` of the lifted manifold.              |
| `projectors`   | `(N, D, D)`  | Property: `P_i = U_i U_iᵀ`.                                           |
| `N, D, d`      | ints         | Point count, ambient dim at this level, intrinsic dim.               |
| `n_orig`       | int          | Original spatial dimension `n`.                                      |
| `level`        | int          | 0 for raw data; +1 per `lift()`.                                     |
| `curvature_ops`| `(N, n_comp, d, d)` or `None` | Shape-operator estimate of the level below; populated on the object returned by `lift()`, `None` at level 0. |

Constructors and methods:

```python
BlowUpLevel.from_point_tangents(points, frames)   # frames (N, n, d) or (N, n) for curves
BlowUpLevel.from_normals(points, normals)         # hypersurfaces only: d = n - 1
level.lift(k=20, alpha=1.0, lam=0.0)              # one blow-up step -> new BlowUpLevel
level.embedding_vector(alpha)                     # (N, D + D^2) Chordal-Sasaki coordinates
level.distance_matrix(...)                        # pairwise lifted distances

iterated_blowup(points, frames, num_levels, k=20, alpha=1.0)   # -> [level0, level1, ...]
ambient_dim_sequence(n, num_levels)                            # D_0 = n, D_l = D_{l-1} + D_{l-1}^2
```

The ambient dimension grows quickly (`n=3` → `D = 3, 12, 156, …`), so in
practice level 1 is used for geodesics/Laplacians and level 2 for
curvature-gradient invariants.

---

## Other things you can compute

### Graph Laplacians and spectral clustering

```python
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.clustering import spectral_embedding, spectral_clustering
from tangent_blowups.solvers.eigen import laplacian_spectrum

L, W, D = lifted_laplacian(level1, kernel="product", k=20)   # sparse CSR
labels  = spectral_clustering(L, n_clusters=6)                # e.g. the 6 faces of a cube
evals, emb = spectral_embedding(L, n_components=4)
evals, evecs = laplacian_spectrum(L, k=10)                     # shift-invert ARPACK
```

Lower-level affinity builders in `geometry.kernels`: `product_affinity`,
`lifted_affinity` (self-tuning), `uniform_affinity`, `affinity_to_laplacian`,
`estimate_product_bandwidths`, plus the discrete `lifted_gradient` /
`lifted_divergence` operators used by the heat method.
`pointcloud.laplacian` has the raw-array equivalents
(`pointcloud_laplacian`, `bilateral_pointcloud_laplacian`,
`lifted_pointcloud_laplacian`).

Rule of thumb from the examples: transverse crossings (figure-8) separate at
level 0–1; *tangential* contacts (line + parabola, plane + paraboloid) only
separate at level 2, where curvature enters the metric.

### Curvature invariants

```python
from tangent_blowups.geometry.iterated_grassmann import (
    iterated_blowup, extract_level0, extract_level1, extract_level2,
)

levels = iterated_blowup(points, frames, num_levels=2, k=20, alpha=1.0)
inv0 = extract_level0(levels[0])           # points, tangent_frame, normal_frame
inv1 = extract_level1(levels[1])           # shape_operator, mean/total/principal/gaussian curvature
inv2 = extract_level2(levels[0], levels[1], inv1, k=20)   # curvature_gradient (Codazzi tensor)
```

Principal / Gaussian curvatures are filled in for hypersurfaces
(`codim == 1`); otherwise they are `None`. Accuracy against closed-form
ground truth is checked in `tests/test_curvature_analytic.py`.

### Reconstruction

- `tangent_blowups.reconstruction.implicit.NeuralPoissonReconstructor` /
  `PoissonConfig` — learns `F: R^D → R^{D−d}` whose zero set is the lifted
  manifold (Poisson manifold reconstruction, torch). See
  `examples/reconstruction/implicit/`.
- `tangent_blowups.reconstruction.explicit.bspline.fit_bspline_curve` — fits a
  B-spline through a lifted curve and projects it back to `R^n`. See
  `examples/reconstruction/explicit/fig8_bspline.py`.

### Synthetic data (`tangent_blowups.testsupport`)

Parametric curves (`circle`, `figure8`, `lissajous`, `tangent_parabola_line`,
…), surfaces (`whitney_umbrella`, `klein_bottle`, `cube_surface`,
`plane_paraboloid_tangent`, `boy_surface`, …), sampling strategies
(`UniformCurve`, `RandomSurface`, `ChebyshevGrid`, …) and modifiers
(`add_point_noise`, `jitter_normals`, `flip_orientations`, …). All produce a
`Sample` dataclass with `points`, `params`, optional `tangents` / `normals`,
and an optional `singular_mask`.

Note: `UniformSurface` / grid samplers return `(Nu, Nv, 3)` arrays; random
samplers return flat `(N, 3)`. Reshape with `.reshape(-1, 3)` before passing to
the algorithms.

### I/O

```python
from tangent_blowups.io.save import save_pointcloud
from tangent_blowups.io.load import load_pointcloud, load_mesh

save_pointcloud("cloud.npz", points=points, normals=normals)   # also .npy (points only)
s = load_pointcloud("cloud.npz")                                # -> Sample
s = load_mesh("model.stl")                                      # STL / OBJ vertices (+ normals)
```

`io.adapters.stl_mesh_to_oriented_point_cloud` samples an oriented point
cloud from a triangle mesh with optional sharp-feature detection; the CLI in
`examples/io/mesh_to_pc.py` wraps it.

### Visualisation (`tangent_blowups.viz`, matplotlib)

`viz.nash_coordinates.visualize_components` / `visualize_quiver` (projector
components of a lift), `viz.laplacian.visualize_laplacian_eigenpairs`,
`viz.dist.visualize_distance_comparison` / `visualize_product_metric_comparison`.

---

## Package layout

```
src/tangent_blowups/
├── geometry/
│   ├── iterated_grassmann.py   BlowUpLevel, lift(), iterated_blowup, extract_level{0,1,2}
│   ├── kernels.py              affinities, lifted_laplacian, lifted_gradient / _divergence
│   ├── weight_config.py        WeightConfig for regression weights
│   ├── projectors.py           chordal / geodesic distances, principal angles
│   └── _gpu_ops.py             torch implementations used when device="cuda"
├── pointcloud/
│   ├── geodesic_heat.py        lifted_heat_method, precompute_heat_method
│   ├── laplacian.py            pointcloud_laplacian, lifted_pointcloud_laplacian, ...
│   ├── neighbors.py            knn_edges, radius_edges
│   └── tangent_estimation.py   estimate_normals_pca, estimate_tangents_pca, ...
├── clustering/spectral.py      spectral_embedding, spectral_clustering
├── solvers/
│   ├── eigen.py                laplacian_spectrum
│   ├── linalg.py               normalize_vectors, ensure_orthonormal
│   └── gpu_sparse.py           torch PCG / sparse solve
├── reconstruction/
│   ├── implicit/neural_poisson.py
│   └── explicit/bspline.py
├── io/                         load.py, save.py, adapters.py
├── viz/                        nash_coordinates.py, laplacian.py, dist.py
└── testsupport/                synthetic curves / surfaces / samplers / modifiers
```

---

## Examples

Every script under `examples/` is standalone (`python path/to/script.py`);
most open a matplotlib or polyscope window.

| Directory                          | What it shows                                                        |
| ---------------------------------- | -------------------------------------------------------------------- |
| `geodesics/2D`, `geodesics/3D`     | Lifted heat method on figure-8, cube, scanned clouds; `alpha` ablation; noise robustness |
| `laplacian/`, `kernel/`            | Laplacian eigenpairs and affinity kernels at different blow-up levels |
| `clustering/`                      | Spectral clustering of sheets (line + parabola, cube, crossing planes, fault surfaces) |
| `iterated_blowups/`                | Level 0 / 1 / 2 separation of tangential intersections (UMAP + HDBSCAN) |
| `curvature/`                       | Blow-up curvature vs. analytic ground truth, jet fitting and CNC baselines |
| `reconstruction/`                  | Neural Poisson (implicit) and B-spline (explicit) reconstruction     |
| `blowups/`, `metrics/`, `bounds/`  | Visualising the lift itself, lifted distance fields, cross-sheet bounds |
| `surface_parameterization/`, `occluding_contours/`, `teaser/` | Application demos |
| `sampling/`, `io/`                 | Sampling strategies; mesh → point-cloud conversion and dataset export |

Scripts whose name is `thingi10k.py`, `pointcloud.py`, `geo_faults_rift.py`
or that live in `examples/curvature/` read external datasets from a top-level
`data/` directory (git-ignored). Generate them with the scripts in
`examples/io/` (`mesh_to_pc.py`, `sample_surfaces_to_txt.py`,
`load_scec_faults.py`) after downloading the source meshes.

`presentation_figures/` contains the scripts that produced the figures for the
accompanying talk; they use the same public API.

---

## Tests

```bash
pytest                         # 18 curvature-accuracy tests, ~10 s
pytest -k sphere -v            # single shape
```

---

## License

MIT — see `LICENSE`.
