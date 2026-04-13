"""
Lifted spectral clustering on Thingi10k point clouds.

Pipeline:
    BlowUpLevel.from_normals(pts, nrm)
        -> lift(alpha)                            (level-1 Chordal-Sasaki)
        -> lifted_laplacian(self-tuning product)  (eigenmap kernel)
        -> spectral_embedding                     (Laplacian eigenmaps)
        -> KMeans                                 (cluster the embedding)

Polyscope shows one point cloud with multiple scalar/color quantities — toggle
between the cluster labels and each eigenmap from the panel selector.

Usage:
    python thingi10k.py -p klein_bottle_two
    python thingi10k.py -p ship --rotate y --n-clusters 8
    python thingi10k.py                                # list available clouds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
from matplotlib.colors import hsv_to_rgb
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN

from tangent_blowups.clustering import spectral_embedding
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "thingi10k_pointcloud"
N_MAX = 100_000


# -- Rotation ---------------------------------------------------------------
def _rotation_matrix(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def _parse_rotation(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "x":
        return _rotation_matrix(90, np.array([0.0, 1.0, 0.0]))
    if spec == "y":
        return _rotation_matrix(-90, np.array([1.0, 0.0, 0.0]))
    parts = spec.split(",")
    if len(parts) == 4:
        return _rotation_matrix(
            float(parts[0]), np.array([float(p) for p in parts[1:]]),
        )
    raise ValueError(f"Invalid --rotate spec: {spec!r}")


# -- Data loading -----------------------------------------------------------
def _available() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.npz"))


def _load(name: str, *, rotate: np.ndarray | None = None):
    pc = load_pointcloud(DATA_DIR / f"{name}.npz")
    pts = np.asarray(pc.points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(pc.normals, dtype=float).reshape(-1, 3)
    valid = (
        np.isfinite(pts).all(1)
        & np.isfinite(nrm).all(1)
        & (np.linalg.norm(nrm, axis=1) > 1e-8)
    )
    pts, nrm = pts[valid], normalize_vectors(nrm[valid])
    if len(pts) > N_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N_MAX, replace=False)
        pts, nrm = pts[idx], nrm[idx]
    if rotate is not None:
        pts = pts @ rotate.T
        nrm = nrm @ rotate.T
    return pts, nrm


# -- Color helpers ----------------------------------------------------------
def _segment_colors(labels: np.ndarray) -> np.ndarray:
    """Distinct golden-ratio HSV color per cluster id."""
    phi = (1 + np.sqrt(5)) / 2
    n_clusters = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    colors = np.full((labels.size, 3), 0.7)
    for c in range(n_clusters):
        colors[labels == c] = hsv_to_rgb([(c * phi) % 1.0, 0.65, 0.85])
    return colors


# -- Per-segment spectral parameterization ----------------------------------
def _segment_spectral_uv(
    pts: np.ndarray,
    nrm: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment 2D Laplacian eigenmap parameterization.

    For each cluster id: build a level-0 BlowUpLevel on the segment, build a
    self-tuning product Laplacian on it (no lift -> pure spatial self-tuning
    Gaussian), and take the 2 smallest non-trivial eigenvectors as (u, v).
    Each segment's uv is independently rescaled per-axis to [-0.5, 0.5] so
    every segment shows a comparable checker tile density.

    Noise points (label = -1) and segments smaller than ~max(8, k+1) points
    get uv = (0, 0) and valid[i] = False.
    """
    N = len(pts)
    uv = np.zeros((N, 2), dtype=float)
    valid = np.zeros(N, dtype=bool)
    for c in np.unique(labels):
        if c < 0:
            continue
        mask = labels == c
        n_seg = int(mask.sum())
        k_seg = min(k, n_seg - 1)
        if n_seg < max(8, k + 1) or k_seg < 3:
            continue
        seg_level = BlowUpLevel.from_normals(pts[mask], nrm[mask])
        L_seg, _, _ = lifted_laplacian(
            seg_level, kernel="product", self_tuning=True,
            k=k_seg, normalized=True,
        )
        try:
            _, emb = spectral_embedding(L_seg, n_components=2)
        except Exception:
            # Disconnected segment / ARPACK failure -- skip rather than crash.
            continue
        lo = emb.min(axis=0)
        hi = emb.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        uv[mask] = (emb - lo) / span - 0.5
        valid[mask] = True
    return uv, valid


def _uv_checker(
    uv: np.ndarray, valid: np.ndarray, n_cells: float,
) -> np.ndarray:
    """RGB checker pattern from per-point (u, v). Invalid points -> gray."""
    cell = 1.0 / max(n_cells, 1e-12)
    cu = np.floor(uv[:, 0] / cell).astype(int)
    cv = np.floor(uv[:, 1] / cell).astype(int)
    check = (cu + cv) % 2
    light = np.array([0.95, 0.95, 0.98])
    dark = np.array([0.30, 0.40, 0.75])
    rgb = np.where(check[:, None], dark, light)
    rgb[~valid] = [0.7, 0.7, 0.7]
    return rgb


# -- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Blow-up spectral clustering + eigenmap viewer.",
    )
    parser.add_argument("--pointcloud", "-p", type=str, default=None,
                        help="Cloud name (omit to list).")
    parser.add_argument("--n-clusters", "-n", type=int, default=10,
                        help="Number of k-means clusters / eigenmap dimension. "
                             "DBSCAN/HDBSCAN ignore the cluster count but still "
                             "use this as the embedding dimension.")
    parser.add_argument("--method", "-m", type=str, default="dbscan",
                        choices=["kmeans", "dbscan", "hdbscan"],
                        help="Clusterer to run on the eigenmap embedding.")
    parser.add_argument("--eps", type=float, default=0.2,
                        help="DBSCAN eps (eigenmap-space radius).")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="DBSCAN min_samples.")
    parser.add_argument("--min-cluster-size", type=int, default=25,
                        help="HDBSCAN min_cluster_size.")
    parser.add_argument("--k", type=int, default=20,
                        help="k-NN neighborhood size.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Chordal-Sasaki lift parameter.")
    parser.add_argument("--num-levels", type=int, default=1,
                        help="Number of iterated blow-up levels.")
    parser.add_argument("--rotate", "-r", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cells", type=float, default=12.0,
                        help="Checker cells per segment side (default 12).")
    parser.add_argument("--seg-k", type=int, default=15,
                        help="k-NN size for the per-segment Laplacian "
                             "(default 15; clamped to segment size - 1).")
    args = parser.parse_args()

    available = _available()
    if args.pointcloud is None:
        print("Available point clouds:")
        for name in available:
            print(f"  {name}")
        print("\nUsage: python thingi10k.py -p <name>")
        return
    if args.pointcloud not in available:
        print(f"Unknown: {args.pointcloud!r}. Available: {', '.join(available)}")
        return

    rotate = _parse_rotation(args.rotate) if args.rotate else None
    pts, nrm = _load(args.pointcloud, rotate=rotate)
    print(f"Loaded {len(pts)} points from {args.pointcloud}")

    # --- Build the iterated blow-up ---------------------------------------
    print(f"Building level-{args.num_levels} blow-up (alpha={args.alpha})...")
    level = BlowUpLevel.from_normals(pts, nrm)
    for _ in range(args.num_levels):
        level = level.lift(k=args.k, alpha=args.alpha)
    print(f"  Embedded dim = {level.D}")

    # --- Self-tuning product Laplacian ------------------------------------
    print("Building self-tuning product Laplacian...")
    L, _, _ = lifted_laplacian(
        level, kernel="product", self_tuning=True,
        k=args.k, normalized=True,
    )
    print(f"  L: shape={L.shape}, nnz={L.nnz}")

    # --- Eigenmaps + clustering -------------------------------------------
    print(f"Computing {args.n_clusters} eigenmaps...")
    evals, embedding = spectral_embedding(L, n_components=args.n_clusters)
    print("  eigenvalues:", " ".join(f"{e:.5f}" for e in evals))

    print(f"Running {args.method} on eigenmap embedding...")
    if args.method == "kmeans":
        labels = KMeans(
            n_clusters=args.n_clusters, n_init=10, random_state=args.seed,
        ).fit_predict(embedding)
    elif args.method == "dbscan":
        labels = DBSCAN(
            eps=args.eps, min_samples=args.min_samples,
        ).fit_predict(embedding)
    else:  # hdbscan
        labels = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
        ).fit_predict(embedding)

    n_found = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    n_noise = int((labels < 0).sum())
    counts = np.bincount(labels[labels >= 0], minlength=n_found).tolist()
    noise_str = f", noise={n_noise}" if n_noise else ""
    print(f"  {n_found} clusters{noise_str}, sizes: {counts}")

    # --- Per-segment spectral parameterization + checker ------------------
    print("Computing per-segment spectral parameterization...")
    uv_seg, uv_valid = _segment_spectral_uv(pts, nrm, labels, k=args.seg_k)
    n_param = int(uv_valid.sum())
    n_seg_param = int(np.unique(labels[uv_valid]).size) if n_param else 0
    print(f"  parameterized {n_param} / {len(pts)} points "
          f"across {n_seg_param} segments")
    checker_rgb = _uv_checker(uv_seg, uv_valid, args.n_cells)

    # --- Polyscope: 5 side-by-side panels ---------------------------------
    # Layout (left -> right):
    #   eigenmap_0 | eigenmap_1 | eigenmap_2 | segmentation | checker
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    bbox_x = float(pts[:, 0].max() - pts[:, 0].min())
    spacing = bbox_x * 1.4
    cluster_rgb = _segment_colors(labels)

    panels: list[tuple[str, str, np.ndarray, dict]] = [
        ("eigenmap_0",   "scalar", embedding[:, 0], {"cmap": "coolwarm"}),
        ("eigenmap_1",   "scalar", embedding[:, 1], {"cmap": "coolwarm"}),
        ("eigenmap_2",   "scalar", embedding[:, 2], {"cmap": "coolwarm"}),
        ("segmentation", "color",  cluster_rgb,     {}),
        ("checker",      "color",  checker_rgb,     {}),
    ]

    for i, (name, kind, values, opts) in enumerate(panels):
        offset = np.array([i * spacing, 0.0, 0.0])
        cloud = ps.register_point_cloud(name, pts + offset, radius=0.0025)
        if kind == "scalar":
            cloud.add_scalar_quantity(name, values, enabled=True, **opts)
        else:
            cloud.add_color_quantity(name, values, enabled=True, **opts)

    ps.show()


if __name__ == "__main__":
    main()
