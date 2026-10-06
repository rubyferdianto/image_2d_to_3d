"""Model-free tests for geometry and view synthesis."""
import numpy as np

from img2three import geometry, stereo
from img2three.depth import disparity_to_depth


def test_disparity_to_depth_endpoints():
    d = disparity_to_depth(np.array([1.0, 0.0]), near=2.0, far=8.0)
    np.testing.assert_allclose(d, [2.0, 8.0], rtol=1e-6)


def test_unproject_center_pixel_on_axis():
    K = geometry.intrinsics(5, 5, 90)
    pts = geometry.unproject(np.full((5, 5), 3.0, np.float32), K)
    np.testing.assert_allclose(pts[2, 2], [0, 0, -3], atol=1e-6)
    assert pts[0, 4, 0] > 0 and pts[0, 4, 1] > 0  # top-right pixel -> +x, +y


def test_faces_face_camera():
    K = geometry.intrinsics(4, 4, 60)
    mesh = geometry.build_mesh(np.full((4, 4), 2.0, np.float32), np.zeros((4, 4, 3), np.uint8), K)
    assert len(mesh.faces) == 2 * 3 * 3
    assert (mesh.face_normals[:, 2] > 0.99).all()  # normals point back toward the camera (+z)


def test_mesh_cuts_depth_discontinuity():
    depth = np.full((4, 4), 1.0, np.float32)
    depth[:, 2:] = 5.0  # wall-like jump between column 1 and 2
    mesh = geometry.build_mesh(depth, np.zeros((4, 4, 3), np.uint8), geometry.intrinsics(4, 4, 60))
    assert len(mesh.faces) == 2 * 3 * 2  # the middle column of quads is removed


def test_render_view_near_pixels_move_more_and_occlude():
    img = np.zeros((1, 10, 3), np.uint8)
    disp = np.zeros((1, 10), np.float32)
    img[0, 4] = 255
    disp[0, 4] = 1.0  # one bright, near pixel
    out = stereo.render_view(img, disp, shift=3, convergence=0.0)
    assert out[0, 7].sum() == 765 and out[0, 4].sum() < 765
    np.testing.assert_array_equal(stereo.render_view(img, disp, 0.0), img)


def test_anaglyph_channels():
    left = np.full((2, 2, 3), 10, np.uint8)
    right = np.full((2, 2, 3), 200, np.uint8)
    np.testing.assert_array_equal(stereo.anaglyph(left, right)[0, 0], [10, 200, 200])
