"""Depth-image-based rendering (DIBR): synthesize new viewpoints, stereo pairs, anaglyphs, wiggle GIFs."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def render_view(image: np.ndarray, disparity: np.ndarray, shift: float, convergence: float = 0.5) -> np.ndarray:
    """Re-render the image from a horizontally shifted camera.

    shift: max pixel displacement (positive moves the camera left). Pixels at
    `convergence` disparity stay put (zero parallax plane). Occlusions are resolved
    with a disparity z-buffer (nearer wins); disoccluded holes are inpainted.
    """
    h, w = disparity.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xt = np.rint(xs + shift * (disparity - convergence)).astype(np.int64)
    valid = (xt >= 0) & (xt < w)
    ys, xs, xt, d = ys[valid], xs[valid], xt[valid], disparity[valid]

    zbuf = np.full((h, w), -1.0, np.float32)
    np.maximum.at(zbuf, (ys, xt), d)
    win = d >= zbuf[ys, xt]

    out = np.zeros_like(image)
    out[ys[win], xt[win]] = image[ys[win], xs[win]]
    holes = (zbuf < 0).astype(np.uint8)
    return cv2.inpaint(out, holes, 3, cv2.INPAINT_TELEA) if holes.any() else out


def stereo_pair(image: np.ndarray, disparity: np.ndarray, strength: float = 0.03, convergence: float = 0.5):
    """Left/right views. strength = max parallax as a fraction of image width."""
    s = strength * image.shape[1] / 2
    return render_view(image, disparity, s, convergence), render_view(image, disparity, -s, convergence)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.hstack([left, right])


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Red/cyan anaglyph: red channel from the left eye, green+blue from the right."""
    return np.dstack([left[..., 0], right[..., 1], right[..., 2]])


def wiggle_frames(image: np.ndarray, disparity: np.ndarray, strength: float = 0.03, frames: int = 16,
                  convergence: float = 0.5) -> list[Image.Image]:
    """Camera swaying side to side — the "3D photo" effect."""
    amp = strength * image.shape[1] / 2
    return [Image.fromarray(render_view(image, disparity, amp * np.sin(2 * np.pi * i / frames), convergence))
            for i in range(frames)]


def save_gif(frames: list[Image.Image], path, fps: int = 20) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
