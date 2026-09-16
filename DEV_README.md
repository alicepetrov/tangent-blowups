# Developer notes

Notes for modifying the library. For usage, see `README.md`.

---

## Setup

```bash
uv sync                       # installs runtime + dev deps (pytest, ruff) into .venv
uv run pytest                 # 18 tests, ~10 s
uv run ruff check src         # lint (line-length 88, target py311)
```

The build backend is hatchling (`[build-system]` in `pyproject.toml`), so
`uv sync` installs `tangent_blowups` from `src/` in editable mode; `pip
install -e .` also works. Dev tools live in `[dependency-groups] dev`.
`uv.lock` is committed; update it with `uv lock` after changing
`pyproject.toml`.

numpy is constrained `>=1.26`; the test suite and the README snippets are
verified on both 1.26 and 2.4.

Known dependency quirk (left as-is; change deliberately): `open3d`,
`polyscope`, `umap-learn` and `robust_laplacian` are used by some examples
but are not declared anywhere (they are heavy and `src/` never imports them).
See the table in `README.md`.

---

## Layout and import rules

Actual dependency direction between subpackages (verified from the imports):

```
solvers        <- nothing internal            (linalg, eigen, gpu_sparse)
geometry       <- solvers                     (iterated_grassmann, kernels, weight_config, projectors, _gpu_ops)
testsupport    <- solvers                     (synthetic data; geom_types.Sample is the shared container)
pointcloud     <- geometry, solvers, testsupport
clustering     <- solvers
reconstruction <- geometry
io             <- solvers, testsupport        (returns Sample)
viz            <- geometry, solvers           (matplotlib only)
```

Keep it acyclic. `geometry` must stay free of `pointcloud`/`io`/`viz`
imports; heavy optional deps (matplotlib, torch, open3d, polyscope) stay at
the edges.

`torch` is imported **lazily** on every CPU path. It is only imported at
module level in `geometry/_gpu_ops.py`, `solvers/gpu_sparse.py` and
`reconstruction/implicit/neural_poisson.py`, and those modules are only
imported when `device="cuda"` is resolved or the reconstructor is used. Keep
it that way so the numpy/scipy path works without CUDA.

Empty `__init__.py` files (`tangent_blowups`, `io`, `reconstruction`, `viz`)
are intentional package markers. The other `__init__.py` files re-export the
public API of their subpackage; add new public names there and to the
module's `__all__`.

---

## Core data flow

```
points (N, n) + tangents/normals
   └─ BlowUpLevel.from_point_tangents / from_normals      level 0, D = n
        └─ .lift(k, alpha, lam)                            level 1, D = n + n^2
             ├─ embedded = [x, sqrt(alpha/2) vec(P^(0))]
             ├─ frame    = tangent d-frame of the lifted manifold
             └─ curvature_ops = shape-operator estimate B_i (set on the result)
                  └─ .lift(...)                            level 2, D = D_1 + D_1^2
```

- `embedded[:, :n_orig]` are always the original spatial coordinates, at every
  level. `product_affinity` relies on this to get pure spatial distances
  (`n_orig` is propagated through `lift()`).
- `_proj_blocks` on a `BlowUpLevel` records `(start_col, n_cols, scale)` for
  each vectorised projector block appended by `lift()`. Kernels use it to
  recover per-level angular distances from `embedded`.
- Neighbourhoods for curvature regression are built in the **Chordal–Sasaki
  embedding**, not in position space: `lift()` queries a k-d tree on
  `self.embedding_vector(alpha)` (current position + projectors), and
  `extract_level2` uses `level1.embedded`. This is what keeps sheets apart at
  tangential contacts.

### QR sign convention

`from_point_tangents` and `lift()` orthonormalise frames with `np.linalg.qr`
and then flip the first column if `det(R[:d, :d]) < 0`. Householder QR is
otherwise free to return `(-Q, -R)`, which produced random tangent sign flips
along curves and broke any loss that depends on frame orientation (neural
Poisson fit targets). Do not remove this.

### Heat method (`pointcloud/geodesic_heat.py`)

`lifted_heat_method` = `_coerce_to_level` → affinity (`product_affinity` or
`lifted_affinity`) → `affinity_to_laplacian` → lumped mass matrix → solve
`(M + tL) u = δ` → `lifted_gradient` → normalise → `_edge_divergence` → solve
`L φ = −div` with a Dirichlet anchor. `precompute_heat_method` caches
everything up to the solves; `_precomputed=` reuses it.

The function still accepts legacy keyword arguments (`radius`,
`subspace_metric`, `include_self`, `subspace_kind`, `normalized`) that are
ignored or aliased. They exist so older example scripts keep running.

### Windows / stdout encoding

Python stdout on Windows is cp1252. Do not put box-drawing or Greek
characters (`═`, `→`, `κ`, `λ`, …) in `print()` calls — they are fine in
docstrings and matplotlib strings.

---

## Examples and data

- Every script under `examples/` and `presentation_figures/` is standalone
  and imports only the public API (plus a couple of private helpers such as
  `geodesic_heat._normals_to_tangent_frames`; keep those stable or update the
  callers).
- Scripts that need real data resolve a top-level `data/` directory relative
  to the repo root (`Path(__file__).resolve().parents[k] / "data"`). That
  directory is git-ignored. Producers live in `examples/io/`:
  `mesh_to_pc.py` (Thingi10K / 3D-scans STL+OBJ → `.npz`),
  `sample_surfaces_to_txt.py` (parametric surfaces → `x y z nx ny nz u v`
  text, used by `examples/curvature/analytic*.py`), `load_scec_faults.py`.
- `examples/curvature/{cnc,jet}/*.txt` are pre-computed baseline curvatures
  from external tools (CNC, jet fitting) and are committed on purpose.
- Polyscope writes `.polyscope.ini` / `imgui.ini` next to any script that
  opens a viewer; both are git-ignored.

---

## Tests

`tests/test_curvature_analytic.py` compares level-1 curvature from the
blow-up against closed-form Gaussian/mean curvature on manifold (sphere,
monkey saddle, Klein bottle) and non-manifold (Whitney umbrella, tangent
spheres, plane–paraboloid, tangent cylinders) shapes. There is no coverage of
the heat method, Laplacians or reconstruction beyond the example scripts.
