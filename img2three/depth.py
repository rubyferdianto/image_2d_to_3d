"""Monocular depth estimation with Depth Anything V2 (Hugging Face transformers)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

MODELS = {
    # Relative (affine-invariant inverse depth). Small is Apache-2.0; Base/Large are CC-BY-NC-4.0.
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    # Metric depth in meters.
    "metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}


@dataclass
class DepthResult:
    depth: np.ndarray       # (H, W) float32, distance along camera z (meters if metric, else relative units)
    disparity: np.ndarray   # (H, W) float32 in [0, 1], 1 = nearest
    is_metric: bool


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load(model_key: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    repo = MODELS[model_key]
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo).to(_device()).eval()
    return processor, model


def normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def disparity_to_depth(disp01: np.ndarray, near: float = 1.0, far: float = 10.0) -> np.ndarray:
    """Map normalized disparity to depth so that disp=1 -> near and disp=0 -> far."""
    inv = disp01 * (1.0 / near - 1.0 / far) + 1.0 / far
    return (1.0 / inv).astype(np.float32)


def estimate_depth(image: Image.Image, model_key: str = "small", near: float = 1.0, far: float = 10.0) -> DepthResult:
    import torch

    processor, model = _load(model_key)
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    pred = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    raw = pred[0]["predicted_depth"].float().cpu().numpy()

    if model_key.startswith("metric"):
        depth = np.maximum(raw, 1e-3).astype(np.float32)
        return DepthResult(depth=depth, disparity=normalize(1.0 / depth), is_metric=True)

    disp = normalize(raw)  # relative models predict inverse depth
    return DepthResult(depth=disparity_to_depth(disp, near, far), disparity=disp, is_metric=False)
