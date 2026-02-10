# Tangent Blow-Ups for Reconstructing Nonmanifold Surfaces

---

**TODO; TEMP DEV NOTES**

Keep the public API small and stable. In src/tbu_recon/__init__.py, re-export only the “front door”:

    from .pointcloud.lifts import grassmann_lift, oriented_grassmann_lift
    from .reconstruction.pipelines import reconstruct_implicit, reconstruct_explicit
    from .reconstruction.nonmanifold.blowups import tangent_blowups

Conventions:

- Avoid boolean flags like oriented=True. Prefer explicit functions/types, e.g. GrassmannLift, OrientedGrassmannLift

Keep clean dependency rules
- geometry/ must not import from anywhere else.
- pointcloud/ may import geometry/.
- reconstruction/ may import pointcloud/, fields/, solvers/.
- io/ and viz/ can depend on optional heavy libs; keep them at the edge.

---

### Getting Started

#TODO

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
    │
    └── _testsupport/             # synthetic datasets + fixtures if helpful
        ├── __init__.py           # Exposes key functions
        ├── geom_types.py         # Data containers (dataclasses)
        ├── 2D_curves.py          # Pure math generators for curves
        ├── 3D_surfaces.py        # Pure math generators for surfaces
        └── modifiers.py          # Noise, jitter, and transforms
        └── samplers.py           # Sampling strategies

# Test Support

#TODO