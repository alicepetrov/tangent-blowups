"""
Convert point cloud NPZ files to plain text.

Each output line has the form: x y z nx ny nz

Usage::

    python npz_to_txt.py                     # converts all NPZ files (thingi10k)
    python npz_to_txt.py -d threedscans      # converts all NPZ files (threedscans)
    python npz_to_txt.py -p ship coral       # converts only ship.npz and coral.npz
    python npz_to_txt.py -o /tmp/out         # write text files to a custom directory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DATASETS = {
    "thingi10k": "thingi10k_pointcloud",
    "threedscans": "threedscans_pointcloud",
}


def _pointcloud_dir(dataset: str) -> Path:
    return _repo_root() / "data" / DATASETS[dataset]


def convert(npz_path: Path, out_path: Path) -> int:
    """Convert a single NPZ to x y z nx ny nz text.  Returns point count."""
    with np.load(npz_path) as data:
        pts = np.asarray(data["points"], dtype=np.float64)
        nrm = np.asarray(data["normals"], dtype=np.float64)

    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if nrm.ndim > 2:
        nrm = nrm.reshape(-1, nrm.shape[-1])

    combined = np.hstack([pts, nrm])
    np.savetxt(out_path, combined, fmt="%.10g")
    return len(combined)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npz point clouds to x y z nx ny nz text files"
    )
    parser.add_argument(
        "-d", "--dataset", choices=list(DATASETS), default="threedscans",
        help="Dataset to convert (default: threedscans).",
    )
    parser.add_argument(
        "-p", "--pointclouds", nargs="+", default=None,
        help="Names to convert (without extension). Omit to convert all.",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=None,
        help="Output directory (default: same as input directory).",
    )
    args = parser.parse_args()

    src = _pointcloud_dir(args.dataset)
    if args.pointclouds is None:
        npz_files = sorted(src.glob("*.npz"))
    else:
        npz_files = [src / f"{name}.npz" for name in args.pointclouds]

    out_dir = args.out_dir or src
    out_dir.mkdir(parents=True, exist_ok=True)

    for npz in npz_files:
        if not npz.exists():
            print(f"  SKIP (not found): {npz.name}")
            continue
        txt = out_dir / f"{npz.stem}.txt"
        n = convert(npz, txt)
        print(f"  {npz.stem}: {n} points -> {txt}")


if __name__ == "__main__":
    main()
