"""Exercise renderer selection without downloading or running neural models."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import trimesh

from img2three import cli, object3d, pipeline
from img2three.blender import BlenderOptions
from img2three.depth import DepthResult


@pytest.fixture
def input_image():
    return Image.new("RGB", (16, 8), (120, 80, 40))


@pytest.fixture
def depth_model(monkeypatch):
    def estimate(image, *_args):
        shape = (image.height, image.width)
        return DepthResult(np.full(shape, 2.0, np.float32),
                           np.full(shape, 0.5, np.float32), False)

    model = Mock(side_effect=estimate)
    monkeypatch.setattr(pipeline, "estimate_depth", model)
    return model


def test_depth_blender_builds_mesh_even_when_only_depth_requested(tmp_path, monkeypatch, input_image, depth_model):
    monkeypatch.setattr(pipeline, "find_blender", lambda _path: "resolved-blender")
    expected = {"render": tmp_path / "depth_render.png", "blend": tmp_path / "depth.blend"}

    def render(mesh_path, out_dir, opts, **kwargs):
        assert Path(mesh_path).is_file()
        assert len(trimesh.load(mesh_path).geometry) == 1
        assert out_dir == tmp_path
        assert opts.executable == "resolved-blender"
        assert opts.resolution == 96
        assert kwargs["prefix"] == "depth"
        assert kwargs["mode"] == "depth"
        assert kwargs["fov"] == 65.0
        assert kwargs["aspect_ratio"] == 2.0
        return expected

    render_mock = Mock(side_effect=render)
    monkeypatch.setattr(pipeline, "render_mesh", render_mock)
    files = pipeline.convert(input_image, tmp_path, pipeline.Options(
        outputs=("depth",), renderer="blender", fov=65.0,
        blender=BlenderOptions(resolution=96)))

    depth_model.assert_called_once()
    render_mock.assert_called_once()
    assert files["mesh"].is_file()
    assert files["depth"].is_file()
    assert files["render"] == expected["render"]
    assert files["blend"] == expected["blend"]


def test_native_depth_does_not_require_blender(tmp_path, monkeypatch, input_image, depth_model):
    discover = Mock(side_effect=AssertionError("native mode must not look for Blender"))
    render = Mock(side_effect=AssertionError("native mode must not render in Blender"))
    monkeypatch.setattr(pipeline, "find_blender", discover)
    monkeypatch.setattr(pipeline, "render_mesh", render)
    files = pipeline.convert(input_image, tmp_path, pipeline.Options(outputs=("depth",)))
    assert set(files) == {"depth", "depth16"}
    assert all(path.is_file() for path in files.values())


def test_missing_blender_fails_before_depth_inference(tmp_path, monkeypatch, input_image, depth_model):
    monkeypatch.setattr(pipeline, "find_blender", Mock(side_effect=FileNotFoundError("missing Blender")))
    with pytest.raises(FileNotFoundError, match="missing Blender"):
        pipeline.convert(input_image, tmp_path, pipeline.Options(renderer="blender"))
    depth_model.assert_not_called()


@pytest.fixture
def object_model(monkeypatch):
    # reconstruct only needs these contexts when the neural renderer is bypassed.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    model = Mock(return_value="scene-codes")
    model.extract_mesh.return_value = [trimesh.creation.box()]
    model.render.side_effect = AssertionError("Blender must render the extracted mesh")
    monkeypatch.setattr(object3d, "_load", lambda: (model, "cpu"))
    monkeypatch.setattr(object3d, "prepare", lambda image, *_args: image.convert("RGB"))
    return model


@pytest.mark.parametrize("turntable", [False, True])
def test_object_blender_uses_mesh_and_always_produces_still(tmp_path, monkeypatch, input_image, object_model, turntable):
    monkeypatch.setattr(object3d, "find_blender", lambda _path: "resolved-blender")
    expected = {"render": tmp_path / "object_render.png", "blend": tmp_path / "object.blend"}
    if turntable:
        expected.update(turntable=tmp_path / "object_turntable.gif", views=tmp_path / "object_views.jpg")

    def render(mesh_path, out_dir, opts, **kwargs):
        assert Path(mesh_path).is_file()
        assert opts.executable == "resolved-blender"
        assert kwargs == {"prefix": "object", "mode": "object", "turntable": turntable, "n_views": 8}
        return expected

    render_mock = Mock(side_effect=render)
    monkeypatch.setattr(object3d, "render_mesh", render_mock)
    files = object3d.reconstruct(input_image, tmp_path, renderer="blender", turntable=turntable, n_views=8)

    render_mock.assert_called_once()
    object_model.render.assert_not_called()
    assert all(files[key].is_file() for key in ("input", "mesh", "obj"))
    assert all(files[key] == path for key, path in expected.items())


def test_missing_blender_fails_before_object_model_load(tmp_path, monkeypatch, input_image):
    load = Mock(side_effect=AssertionError("must check Blender before loading TripoSR"))
    monkeypatch.setattr(object3d, "_load", load)
    monkeypatch.setattr(object3d, "find_blender", Mock(side_effect=FileNotFoundError("missing Blender")))
    with pytest.raises(FileNotFoundError, match="missing Blender"):
        object3d.reconstruct(input_image, tmp_path, renderer="blender")
    load.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["depth", "object"])
def test_unknown_renderer_rejected_before_inference(tmp_path, input_image, entrypoint):
    with pytest.raises(ValueError, match="renderer"):
        if entrypoint == "depth":
            pipeline.convert(input_image, tmp_path, pipeline.Options(renderer="unknown"))
        else:
            object3d.reconstruct(input_image, tmp_path, renderer="unknown")


def test_cli_passes_blender_settings_to_both_pipelines(tmp_path, monkeypatch):
    depth = Mock(return_value={})
    objects = Mock(return_value={})
    monkeypatch.setattr(cli, "convert", depth)
    monkeypatch.setattr(object3d, "reconstruct", objects)
    monkeypatch.setattr(cli, "find_blender", lambda _path: "resolved-blender")

    cli.main(["photo.png", "--out", str(tmp_path), "--mode", "both", "--renderer", "blender",
              "--blender-engine", "eevee", "--render-size", "96", "--render-samples", "8",
              "--no-turntable"])

    opts = depth.call_args.args[2]
    assert opts.renderer == "blender"
    assert opts.blender.engine == "BLENDER_EEVEE_NEXT"
    assert opts.blender.resolution == 96
    assert opts.blender.samples == 8
    assert opts.blender.executable == "resolved-blender"
    assert objects.call_args.kwargs["renderer"] == "blender"
    assert objects.call_args.kwargs["blender"] == opts.blender
    assert objects.call_args.kwargs["turntable"] is False


@pytest.mark.parametrize("flag,value", [("--render-size", "0"), ("--render-samples", "-1"),
                                         ("--render-size", "2.5")])
def test_cli_rejects_invalid_render_settings(flag, value):
    with pytest.raises(SystemExit) as error:
        cli.main(["photo.png", flag, value])
    assert error.value.code == 2


def test_cli_missing_blender_is_actionable_parser_error(monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_blender", Mock(side_effect=FileNotFoundError("Set BLENDER_PATH")))
    with pytest.raises(SystemExit) as error:
        cli.main(["photo.png", "--renderer", "blender"])
    assert error.value.code == 2
    assert "Set BLENDER_PATH" in capsys.readouterr().err
