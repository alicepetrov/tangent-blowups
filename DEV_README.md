# DEV NOTES

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

## Implementation Tracker

Icons:

☐ = Not started

✅ = Done

🟨 = In progress

❌ = Cancelled


| Module Path   | Purpose                               | Status | Notes |
| ------------- | ------------------------------------- | ------ | ----- |
| `__init__.py` | Package init                          | ☐      |       |
| `_version.py` | Version management                    | ☐      |       |
| `config.py`   | Shared pipeline configuration objects | ☐      |       |

---

### `geometry/` — Core geometric primitives + algebra

| Module                     | Purpose                                               | Status | Notes                           |
| -------------------------- | ----------------------------------------------------- | ------ | ------------------------------- |
| `geometry/grassmann.py`    | Grassmannian points, metrics, exp/log, geodesics      | 🟨      | Core theoretical backbone       |
| `geometry/orientations.py` | Orientation handling, sign flips, double cover tricks | ☐      | Important for consistent lifts  |
| `geometry/projectors.py`   | (P = UU^T), principal angles, subspace distances      | 🟨      | Needed for metrics + clustering |
| `geometry/kernels.py`      | Geometric kernels on (G(k,n)), heat kernels           | ☐      | Optional / research extension   |

---

### `pointcloud/` — Point cloud input processing

| Module                             | Purpose                                  | Status | Notes                       |
| ---------------------------------- | ---------------------------------------- | ------ | --------------------------- |
| `pointcloud/neighbors.py`          | kNN / radius graph backends              | ☐      | Wrap sklearn/faiss/pykdtree |
| `pointcloud/denoise.py`            | Filtering / outlier handling             | ☐      | Optional                    |
| `pointcloud/tangent_estimation.py` | PCA / robust PCA / MLS                   | ☐      | Required for lift           |
| `pointcloud/lifts.py`              | Grassmannian lifts (oriented/unoriented) | ☐      | ⭐ Core feature              |

---

### `fields/` — Fields over points

| Module                    | Purpose                                           | Status | Notes                |
| ------------------------- | ------------------------------------------------- | ------ | -------------------- |
| `fields/tangent_field.py` | Field container + smoothing / compare / transport | ☐      | Abstraction layer    |
| `fields/smooth.py`        | Graph smoothing / regularization                  | ☐      | Needed for denoising |

---

### `solvers/`

| Module              | Purpose                                     | Status | Notes                 |
| ------------------- | ------------------------------------------- | ------ | --------------------- |
| `solvers/linalg.py` | Sparse wrappers, eigensolvers, SVD backends | 🟨      | Keep backend-agnostic |

---

### `io/`

| Module           | Purpose                           | Status | Notes         |
| ---------------- | --------------------------------- | ------ | ------------- |
| `io/load.py`     | Load ply/obj/npz                  | ☐      |               |
| `io/save.py`     | Save meshes / fields              | ☐      |               |
| `io/adapters.py` | open3d/trimesh/polyscope adapters | ☐      | Optional deps |

---

### `viz/`

| Module         | Purpose                       | Status | Notes    |
| -------------- | ----------------------------- | ------ | -------- |
| `viz/debug.py` | Debug visualization utilities | ☐      | Optional |
| `viz/nash_coordinates.py` | Plotting lifted nash coordinates | ✅      | Optional |

---

### `testsupport/` — Synthetic datasets

| Module                        | Purpose                     | Status | Notes |
| ----------------------------- | --------------------------- | ------ | ----- |
| `testsupport/geom_types.py`  | Dataclass containers        | ✅      |       |
| `testsupport/2D_curves.py`   | Curve generators            | 🟨      |   Add more self intersecting curves    |
| `testsupport/3D_surfaces.py` | Surface generators          | 🟨      |   Add more self intersecting surfaces    |
| `testsupport/modifiers.py`   | Noise / jitter / transforms | ✅      |       |
| `testsupport/samplers.py`    | Sampling strategies         | ✅      |       |

---
