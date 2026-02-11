# DEV NOTES


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
| `geometry/grassmann.py`    | Grassmannian points, metrics, exp/log, geodesics      | ☐      | Core theoretical backbone       |
| `geometry/orientations.py` | Orientation handling, sign flips, double cover tricks | ☐      | Important for consistent lifts  |
| `geometry/projectors.py`   | (P = UU^T), principal angles, subspace distances      | ☐      | Needed for metrics + clustering |
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

### `reconstruction/` — High-level pipelines

| Module                        | Purpose                                          | Status | Notes           |
| ----------------------------- | ------------------------------------------------ | ------ | --------------- |
| `reconstruction/pipelines.py` | Orchestration: fit → lift → blowup → reconstruct | ☐      | User-facing API |

#### `explicit/`

| Module                    | Purpose                                 | Status | Notes                 |
| ------------------------- | --------------------------------------- | ------ | --------------------- |
| `explicit/crust.py`       | Explicit surface reconstruction methods | ☐      | Placeholder initially |
| `explicit/atlas.py`       | Chart stitching / param patches         | ☐      | Research-heavy        |
| `explicit/postprocess.py` | Cleanup / refinement                    | ☐      |                       |

#### `implicit/`

| Module                | Purpose                                   | Status | Notes |
| --------------------- | ----------------------------------------- | ------ | ----- |
| `implicit/udf.py`     | UDF construction                          | ☐      |       |
| `implicit/poisson.py` | Poisson-style variants                    | ☐      |       |
| `implicit/extract.py` | Marching cubes / dual contouring adapters | ☐      |       |

#### `nonmanifold/`

| Module                    | Purpose                                       | Status | Notes          |
| ------------------------- | --------------------------------------------- | ------ | -------------- |
| `nonmanifold/blowups.py`  | Tangent blow-up computation, sheet clustering | ☐      | ⭐ Core novelty |
| `nonmanifold/topology.py` | Topological logic                             | ☐      |                |

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

---

### `_testsupport/` — Synthetic datasets

| Module                        | Purpose                     | Status | Notes |
| ----------------------------- | --------------------------- | ------ | ----- |
| `_testsupport/geom_types.py`  | Dataclass containers        | ✅      |       |
| `_testsupport/2D_curves.py`   | Curve generators            | 🟨      |   Add more self intersecting curves    |
| `_testsupport/3D_surfaces.py` | Surface generators          | 🟨      |   Add more self intersecting surfaces    |
| `_testsupport/modifiers.py`   | Noise / jitter / transforms | ✅      |       |
| `_testsupport/samplers.py`    | Sampling strategies         | ✅      |       |

---
