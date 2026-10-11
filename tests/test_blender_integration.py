"""Opt-in, model-free tests against a real Blender installation.

Set BLENDER_TEST_PATH to the executable to enable these tests. Optionally set
BLENDER_TEST_ENGINE to CYCLES (default) or BLENDER_EEVEE_NEXT / eevee.
"""
import os

import numpy as np
from PIL import Image
import pytest
import trimesh

from img2three import geometry
from img2three.blender import BlenderOptions, render_mesh


@pytest.fixture
def real_blender_options():
    executable = os.environ.get("BLENDER_TEST_PATH")
    if not executable:
        pytest.skip("Set BLENDER_TEST_PATH to opt into real Blender rendering tests")
    engine = os.environ.get("BLENDER_TEST_ENGINE", "CYCLES").upper()
    if engine == "EEVEE":
        engine = "BLENDER_EEVEE_NEXT"
    return BlenderOptions(executable=executable, engine=engine, resolution=64, samples=4, timeout=180)


@pytest.mark.parametrize("width,height", [(16, 8), (8, 16)], ids=["landscape", "portrait"])
def test_depth_preserves_source_orientation_framing_and_colors(tmp_path, real_blender_options, width, height):
    colors = np.empty((height, width, 3), dtype=np.uint8)
    quadrants = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
    colors[:height // 2, :width // 2] = quadrants[0]
    colors[:height // 2, width // 2:] = quadrants[1]
    colors[height // 2:, :width // 2] = quadrants[2]
    colors[height // 2:, width // 2:] = quadrants[3]
    mesh = geometry.build_mesh(np.full((height, width), 2.0, dtype=np.float32),
                               colors, geometry.intrinsics(width, height, 55.0))
    mesh_path = tmp_path / "quadrants.glb"
    mesh.export(mesh_path)

    files = render_mesh(mesh_path, tmp_path / "out", real_blender_options,
                        prefix="depth", mode="depth", fov=55.0, aspect_ratio=width / height)

    expected_size = (64, 32) if width > height else (32, 64)
    with Image.open(files["render"]) as image:
        assert image.size == expected_size
        pixels = np.asarray(image.convert("RGB"))
    render_width, render_height = expected_size
    centers = [(render_width // 4, render_height // 4),
               (3 * render_width // 4, render_height // 4),
               (render_width // 4, 3 * render_height // 4),
               (3 * render_width // 4, 3 * render_height // 4)]
    for (x, y), expected in zip(centers, quadrants):
        np.testing.assert_allclose(pixels[y, x], expected, atol=10,
                                   err_msg=f"Source color changed at rendered pixel ({x}, {y})")
    assert files["blend"].read_bytes().startswith(b"BLENDER")


def test_object_renders_colored_mesh_and_full_turntable(tmp_path, real_blender_options):
    mesh = trimesh.creation.box(extents=(0.7, 1.2, 2.0))
    # Distinct corner colors ensure the four views cannot collapse to one GIF frame.
    normalized = (mesh.vertices - mesh.bounds[0]) / mesh.extents
    mesh.visual.vertex_colors = np.column_stack((40 + normalized * 180,
                                                 np.full(len(mesh.vertices), 255))).astype(np.uint8)
    mesh_path = tmp_path / "z_up_box.glb"
    mesh.export(mesh_path)

    files = render_mesh(mesh_path, tmp_path / "out", real_blender_options,
                        mode="object", turntable=True, n_views=4)

    with Image.open(files["render"]) as image:
        assert image.size == (64, 64)
        assert np.asarray(image.convert("RGB")).std() > 5
    with Image.open(files["turntable"]) as animation:
        assert animation.n_frames == 4
        assert animation.size == (64, 64)
    with Image.open(files["views"]) as sheet:
        assert sheet.size == (64 * 6, 64)
    assert files["blend"].read_bytes().startswith(b"BLENDER")
