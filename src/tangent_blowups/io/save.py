from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..testsupport.geom_types import Sample


def _serialize_params(payload: dict, params) -> None:
    if params is None:
        payload["params_kind"] = np.array("none", dtype="U4")
        return

    if isinstance(params, tuple):
        payload["params_kind"] = np.array("tuple", dtype="U5")
        payload["params_count"] = np.array([len(params)], dtype=np.int64)
        for idx, value in enumerate(params):
            payload[f"params_{idx}"] = np.asarray(value)
        return

    payload["params_kind"] = np.array("array", dtype="U5")
    payload["params"] = np.asarray(params)


def save_pointcloud(
    path: str | Path,
    sample: Optional["Sample"] = None,
    *,
    points: Optional[np.ndarray] = None,
    params=None,
    tangents: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    singular_mask: Optional[np.ndarray] = None,
    singular_indices: Optional[np.ndarray] = None,
    compress: bool = True,
) -> Path:
    """
    Save a point cloud to disk.

    Supports:
        - .npz (points + optional fields)
        - .npy (points only)
    """
    if sample is not None:
        if points is None:
            points = sample.points
        if params is None:
            params = sample.params
        if tangents is None:
            tangents = sample.tangents
        if normals is None:
            normals = sample.normals
        if singular_mask is None:
            singular_mask = sample.singular_mask
        if singular_indices is None:
            singular_indices = sample.singular_indices

    if points is None:
        raise ValueError("points must be provided (or pass a Sample).")

    path = Path(path)
    suffix = path.suffix.lower()

    points = np.asarray(points, dtype=float)
    if suffix == ".npy":
        np.save(path, points)
        return path

    if suffix != ".npz":
        raise ValueError("Unsupported extension. Use '.npz' or '.npy'.")

    payload: dict[str, np.ndarray] = {"points": points}
    if tangents is not None:
        payload["tangents"] = np.asarray(tangents, dtype=float)
    if normals is not None:
        payload["normals"] = np.asarray(normals, dtype=float)
    if singular_mask is not None:
        payload["singular_mask"] = np.asarray(singular_mask, dtype=bool)
    if singular_indices is not None:
        payload["singular_indices"] = np.asarray(singular_indices, dtype=int)

    _serialize_params(payload, params)

    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)

    return path


__all__ = ["save_pointcloud"]
