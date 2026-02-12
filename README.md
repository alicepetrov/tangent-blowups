# Tangent Blow-Ups for Reconstructing Nonmanifold Surfaces

---

### Getting Started

#TODO

---

### Project Structure

    src/tbu_recon/
    ├── __init__.py
    ├── _version.py
    ├── config.py                 # Shared config objects for pipelines
    │
    ├── geometry/                 # Core geometric primitives + algebra
    │   ├── __init__.py
    │   ├── grassmann.py          # Grassmannian points, metrics, exp/log, geodesics
    │   ├── orientations.py       # Orientation handling, sign flips, double cover tricks
    │   ├── projectors.py         # P = UU^T representations, principal angles, distances
    │   └── kernels.py            # (Optional) geometric kernels on G(k,n), heat kernels, etc.
    │
    ├── pointcloud/               # Anything that takes a point cloud as input
    │   ├── __init__.py
    │   ├── neighbors.py          # kNN / radius graph backends; abstracts sklearn/faiss/pykdtree
    │   ├── denoise.py            # (Optional) filtering / outlier handling
    │   ├── tangent_estimation.py # local PCA / robust PCA / MLS; returns tangent planes
    │   └── lifts.py              # grassmannian lifts (oriented + unoriented) = star feature
    │
    ├── fields/                   # “Fields” living over points (tangent plane field, normals, etc.)
    │   ├── __init__.py
    │   ├── tangent_field.py      # container + ops (smooth, compare, transport)
    │   └── smooth.py             # smoothing / regularization on graphs
    │
    ├── reconstruction/           # High-level reconstruction pipelines
    │   ├── __init__.py
    │   ├── explicit/             # mesh/complex output
    │   │   ├── __init__.py
    │   │   ├── crust.py          # placeholders for explicit methods
    │   │   ├── atlas.py          # charts, stitching, param patches
    │   │   └── postprocess.py
    │   ├── implicit/             # implicit field output (UDF/indicator/level set)
    │   │   ├── __init__.py
    │   │   ├── udf.py            # UDF construction
    │   │   ├── poisson.py        # poisson-style variants
    │   │   └── extract.py        # marching cubes / dual contouring adapters
    │   ├── nonmanifold/          # logic specific to nonmanifold branching (maybe edges/corners?)
    │   │   ├── __init__.py
    │   │   ├── blowups.py        # tangent blow-up computation, clustering sheets, etc. 
    │   │   └── topology.py
    │   └── pipelines.py          # user-facing orchestration: fit->lift->blowup->reconstruct
    │
    ├── solvers/                  # Optimization / linear algebra that you may swap
    │   ├── __init__.py
    │   ├── linalg.py             # sparse wrappers, eigensolvers, svd backends
    │
    ├── io/                       # Pure IO/adapters (keep dependencies contained)
    │   ├── __init__.py
    │   ├── load.py               # ply/obj/npz/etc
    │   ├── save.py
    │   └── adapters.py           # open3d/trimesh/polyscope conversions (optional deps)
    │
    ├── viz/                      # Optional: plotting, polyscope, napari, etc.
    │   ├── __init__.py
    │   └── debug.py
    │   └── nash_coordinates.py
    │
    └── _testsupport/             # synthetic datasets + fixtures if helpful
        ├── __init__.py           # Exposes key functions
        ├── geom_types.py         # Data containers (dataclasses)
        ├── 2D_curves.py          # Pure math generators for curves
        ├── 3D_surfaces.py        # Pure math generators for surfaces
        └── modifiers.py          # Noise, jitter, and transforms
        └── samplers.py           # Sampling strategies

---

# Test Support

The `tangent_blowups.testsupport` package provides synthetic geometry and
sampling utilities used by tests, examples, and algorithm prototyping.

It is intended to be the "front door" for generating:

- Parametric curves/surfaces with optional tangents and normals
- Sampled point sets using reusable sampling strategies
- Controlled perturbations (noise, jitter, orientation flips)

## Quick Usage

### 1) Sample a curve uniformly

```python
import numpy as np
from tangent_blowups.testsupport import circle, sample, UniformCurve

gt = circle(radius=2.0)
curve_strategy = UniformCurve(n=256, t_min=0.0, t_max=2.0 * np.pi, endpoint=True)

s = sample(
    gt,
    curve_strategy,
    with_tangents=True,
    with_normals=True,
)

print(s.points.shape)    # (256, 2)
print(s.tangents.shape)  # (256, 2)
print(s.normals.shape)   # (256, 2)
```

### 2) Sample a surface and add controlled corruption

```python
import numpy as np
from tangent_blowups.testsupport import (
    RandomSurface,
    add_point_noise,
    flip_orientations,
    sample,
    whitney_umbrella,
)

gt = whitney_umbrella(scale=1.0)
surface_strategy = RandomSurface(
    n=3000,
    u_bounds=(-2.0, 2.0),
    v_bounds=(-2.0, 2.0),
    rng=np.random.default_rng(7),
)

raw = sample(gt, surface_strategy, with_tangents=True, with_normals=True)
noisy = add_point_noise(raw, sigma=0.01)
augmented = flip_orientations(noisy, flip_probability=0.2, coupled=True)
```

### 3) Detect likely self-intersections during evaluation

```python
import numpy as np
from tangent_blowups.testsupport import figure8, UniformCurve

curve = figure8(scale=2.0)
t = UniformCurve(n=1000, t_min=0.0, t_max=2.0 * np.pi)(curve)[0]

s = curve.evaluate(
    t,
    with_tangents=True,
    with_normals=True,
    singularity_tol=0.05,
)

if s.singular_mask is not None:
    print("Detected singular points:", int(s.singular_mask.sum()))
```

## Run End-to-End Example Scripts

The repository includes plotting demos that iterate through available strategies:

```bash
python examples/2D_sampling.py
python examples/3D_sampling.py
```

These scripts show sampling behavior and highlight detected singular regions.

---
