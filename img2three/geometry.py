"""Turn a depth map into 3D geometry: point clouds and 2.5D meshes."""
from __future__ import annotations

import numpy as np


def intrinsics(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    """Pinhole camera matrix from a horizontal field of view."""
    fx = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    return np.array([[fx, 0, (width - 1) / 2], [0, fx, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project every pixel to 3D. Returns (H, W, 3) in glTF convention (x right, y up, camera looks down -z)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    return np.stack([x, -y, -depth], axis=-1).astype(np.float32)


def grid_faces(h: int, w: int) -> np.ndarray:
    """Two counter-clockwise (camera-facing) triangles per pixel quad."""
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])


def build_mesh(depth: np.ndarray, colors: np.ndarray, K: np.ndarray, edge_threshold: float = 0.04):
    """Build a vertex-colored mesh from a depth map.

    Triangles spanning a depth discontinuity (relative depth jump > edge_threshold)
    are dropped so foreground objects don't get "stretched skin" to the background.
    """
    import trimesh

    h, w = depth.shape
    verts = unproject(depth, K).reshape(-1, 3)
    faces = grid_faces(h, w)
    z = depth.reshape(-1)[faces]
    keep = z.max(1) / z.min(1) - 1.0 < edge_threshold
    rgba = np.concatenate([colors.reshape(-1, 3), np.full((h * w, 1), 255, np.uint8)], axis=1)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces[keep], vertex_colors=rgba, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def build_point_cloud(depth: np.ndarray, colors: np.ndarray, K: np.ndarray):
    import trimesh

    return trimesh.PointCloud(unproject(depth, K).reshape(-1, 3), colors=colors.reshape(-1, 3))
