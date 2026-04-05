"""
Sample point clouds from parametric surfaces and save as text files.

Two files are written per surface:

  {name}.txt          x y z nx ny nz u v   (with parametric coordinates)
  {name}_xyzn.txt     x y z nx ny nz       (positions + normals only, for CNC)

The u v columns store the parametric coordinates so that analytic ground-
truth curvatures can be evaluated on the exact same point set.

Usage::

    python sample_surfaces_to_txt.py                  # default 100000 points
    python sample_surfaces_to_txt.py -n 50000         # custom point count
    python sample_surfaces_to_txt.py -o /tmp/out      # custom output directory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tangent_blowups.testsupport import (
    klein_bottle,
    torus,
    whitney_umbrella,
    RandomSurface,
    sample,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_out_dir() -> Path:
    return _repo_root() / "data" / "sampled_surfaces"


def sample_and_save(name, surface, strategy, out_dir: Path) -> int:
    s = sample(surface, strategy, with_normals=True)
    pts = np.asarray(s.points, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(s.normals, dtype=np.float64).reshape(-1, 3)
    u = np.asarray(s.params[0], dtype=np.float64).ravel()
    v = np.asarray(s.params[1], dtype=np.float64).ravel()

    valid = np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
    valid &= np.linalg.norm(nrm, axis=1) > 1e-8
    pts, nrm, u, v = pts[valid], nrm[valid], u[valid], v[valid]

    # Full file with parametric coordinates
    full_path = out_dir / f"{name}.txt"
    combined = np.hstack([pts, nrm, u[:, None], v[:, None]])
    np.savetxt(full_path, combined, fmt="%.10g")
    print(f"  {name}: {len(combined)} points -> {full_path}")

    # Positions + normals only (for CNC input)
    xyzn_path = out_dir / f"{name}_xyzn.txt"
    xyzn = np.hstack([pts, nrm])
    np.savetxt(xyzn_path, xyzn, fmt="%.10g")
    print(f"  {name}: {len(xyzn)} points -> {xyzn_path}")

    return len(combined)


def main():
    parser = argparse.ArgumentParser(
        description="Sample parametric surfaces to text files"
    )
    parser.add_argument(
        "-n", "--num-points", type=int, default=100_000,
        help="Number of points to sample (default: 100000)",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=None,
        help="Output directory (default: data/sampled_surfaces/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    surfaces = {
        "whitney_umbrella": (
            whitney_umbrella(scale=1.0),
            RandomSurface(n=args.num_points, u_bounds=(-2.0, 2.0),
                          v_bounds=(-2.0, 2.0), rng=rng),
        ),
        "klein_bottle": (
            klein_bottle(radius=2.0, scale=1.0),
            RandomSurface(n=args.num_points, u_bounds=(0.0, 2 * np.pi),
                          v_bounds=(0.0, 2 * np.pi), rng=rng),
        ),
        "torus": (
            torus(R=2.0, r=1.0),
            RandomSurface(n=args.num_points, u_bounds=(0.0, 2 * np.pi),
                          v_bounds=(0.0, 2 * np.pi), rng=rng),
        ),
    }

    for name, (surface, strategy) in surfaces.items():
        sample_and_save(name, surface, strategy, out_dir)


if __name__ == "__main__":
    main()
